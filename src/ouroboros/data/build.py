"""Corpus assembly: turn raw sources into span-annotated training documents."""

from __future__ import annotations

import collections
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from ..labels import Humanizer, Provenance
from .documents import Document, Span, iter_json_lines, write_jsonl
from .humanize import EVASION_ATTACKS, INCIDENTAL_ARTIFACTS, apply_transforms
from .soft_ngrams import Embedder, NullEmbedder, SoftNGramConfig, label_edits
from .sources import RaidSubset, group_by_source, read_raid
from .splice import splice


@dataclass
class BuildSpec:
    raid_csv: str
    out_dir: str = "data/corpus"
    n_human: int = 24000
    n_ai: int = 24000
    n_humanized: int = 7000
    n_spliced: int = 9000
    #: RAID holds a fixed pool of human source documents, so the AI strata are
    #: trimmed relative to whatever human pool the scan actually returned.
    ai_per_human: float = 1.0
    humanized_per_human: float = 0.5
    #: JSONL of {"source": ..., "target": ...} pairs produced by `ouroboros edit`.
    edited_pairs: str | None = None
    #: A second pairs file, e.g. the diverse portfolio from `ouroboros api-edit`.
    extra_edited_pairs: str | None = None
    #: Comma-separated JSONL files of extra human/AI text from `hf-fetch`.
    hf_samples: str | None = None
    #: Cap on Soft N-Grams labeling (0 = no cap).
    max_edited_pairs: int = 0
    #: Fraction of pre-2024-generator AI text to keep. Those models dominate
    #: the public corpora but write nothing like current ones, and training on
    #: them teaches tics ("Furthermore", "It is essential to understand") that
    #: today's models have dropped.
    legacy_ai_keep: float = 1.0
    #: Cap the human pool at this multiple of the AI pool. A large human
    #: surplus is not free: it shifts the model's prior towards `human`, which
    #: shows up as false negatives on registers the AI side does not cover.
    human_per_ai: float = 0.0
    #: Resample the human pool towards the report's Figure 2 domain mix. Our raw
    #: pool is dominated by encyclopaedic and web text, while the report's is
    #: mostly creative, scientific and reference writing; a detector only sees
    #: the registers it was shown.
    match_domain_mix: bool = False
    #: JSONL of ready-made Documents (already carrying spans).
    extra_documents: str | None = None
    #: Heterogeneous mixed documents spliced from the HuggingFace pool.
    n_hf_spliced: int = 8000
    #: Cap the HuggingFace AI pool at this multiple of its human pool, so the
    #: AI class cannot drown out the human class the FPR depends on.
    hf_ai_per_human: float = 1.0
    #: Fraction of AI documents that also receive one of our own evasion attacks.
    own_attack_rate: float = 0.12
    #: Fraction of documents that receive incidental (non-humanizer) corruption.
    incidental_rate: float = 0.06
    min_words: int = 50
    eval_fraction: float = 0.06
    calibration_fraction: float = 0.06
    #: Final held-out split. Never used for fitting or for model selection.
    test_fraction: float = 0.08
    domain_mix_min_keep: float = 0.5
    seed: int = 0
    cache_dir: str | None = None
    embedder: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    splits: dict[str, float] = field(default_factory=dict)


def _word_count(text: str) -> int:
    return len(text.split())


def _content_key(text: str, prefix_chars: int = 600) -> str:
    """Whitespace- and case-insensitive fingerprint of a document.

    Public corpora overlap heavily -- the same Wikipedia article turns up in
    several of them -- and keying the split on a positional index let identical
    text land on both sides of the train/test boundary. Deduplicating on content
    removes the leak at the source.
    """
    normalised = " ".join(text.split()).lower()[:prefix_chars]
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=12).hexdigest()


def _splitter(source_id: str, spec: BuildSpec) -> str:
    """Deterministic split by source document, so no source leaks across splits.

    Four splits: `train` fits the model, `calibration` fits the CRF penalties and
    the unary calibrator, `eval` is the development set watched during training,
    and `test` is touched exactly once, for the final report.
    """
    h = random.Random(f"{spec.seed}:{source_id}").random()
    if h < spec.eval_fraction:
        return "eval"
    if h < spec.eval_fraction + spec.calibration_fraction:
        return "calibration"
    if h < spec.eval_fraction + spec.calibration_fraction + spec.test_fraction:
        return "test"
    return "train"


def build_corpus(spec: BuildSpec, progress: bool = True) -> dict[str, int]:
    rng = random.Random(spec.seed)
    subset: RaidSubset = read_raid(
        spec.raid_csv,
        n_human=spec.n_human,
        n_ai=spec.n_ai,
        n_humanized=spec.n_humanized,
        seed=spec.seed,
        progress=progress,
        cache_dir=spec.cache_dir,
    )
    print(f"[build] raid subset: {subset.summary()}")
    n_human_available = len(subset.human)
    subset.ai = subset.ai[: int(spec.ai_per_human * n_human_available)]
    subset.humanized = subset.humanized[: int(spec.humanized_per_human * n_human_available)]
    print(f"[build] balanced subset: {subset.summary()}")

    buckets: dict[str, list[Document]] = {"train": [], "eval": [], "calibration": [], "test": []}

    seen_content: set[str] = set()
    counters = {"short": 0, "duplicate": 0, "kept": 0}

    def add(doc: Document, source_id: str) -> None:
        if _word_count(doc.text) < spec.min_words:
            counters["short"] += 1
            return
        key = _content_key(doc.text)
        if key in seen_content:
            counters["duplicate"] += 1
            return
        seen_content.add(key)
        doc.validate()
        # Fall back to the content fingerprint when no real source id groups the
        # document, so identical text can never straddle two splits.
        buckets[_splitter(source_id or key, spec)].append(doc)
        counters["kept"] += 1

    # --- pure human and pure AI ------------------------------------------------
    for sample in subset.human:
        doc = Document.uniform(
            f"human-{sample.id}",
            sample.text,
            Provenance.HUMAN,
            Humanizer.HUMAN,
            f"raid/{sample.domain}",
            domain=sample.domain,
        )
        if rng.random() < spec.incidental_rate:
            # Incidental artifacts must NOT be labeled humanization (Section 3.4).
            doc = apply_transforms(
                doc, [rng.choice(list(INCIDENTAL_ARTIFACTS))], rng, suffix="-inc"
            )
        add(doc, sample.source_id)

    for sample in subset.ai:
        doc = Document.uniform(
            f"ai-{sample.id}",
            sample.text,
            Provenance.AI,
            Humanizer.AI_GENERATED,
            f"raid/{sample.domain}",
            domain=sample.domain,
            generator=sample.model,
        )
        roll = rng.random()
        if roll < spec.own_attack_rate:
            names = rng.sample(list(EVASION_ATTACKS), rng.randint(1, 2))
            doc = apply_transforms(doc, names, rng, int(Humanizer.HUMANIZED_AI), "-evade")
        elif roll < spec.own_attack_rate + spec.incidental_rate:
            doc = apply_transforms(
                doc, [rng.choice(list(INCIDENTAL_ARTIFACTS))], rng, suffix="-inc"
            )
        add(doc, sample.source_id)

    # --- RAID's own adversarial attacks ---------------------------------------
    for sample in subset.humanized:
        add(
            Document.uniform(
                f"humanized-{sample.id}",
                sample.text,
                Provenance.AI,
                Humanizer.HUMANIZED_AI,
                f"raid/{sample.domain}",
                domain=sample.domain,
                generator=sample.model,
                attack=sample.attack,
            ),
            sample.source_id,
        )

    # --- heterogeneous mixed text (Section 3.2) --------------------------------
    human_by_source = group_by_source(subset.human)
    ai_by_source = group_by_source(subset.ai)
    shared = sorted(set(human_by_source) & set(ai_by_source))
    rng.shuffle(shared)
    made = 0
    bar = tqdm(total=spec.n_spliced, desc="splice", disable=not progress)
    for source_id in shared:
        if made >= spec.n_spliced:
            break
        human_text = rng.choice(human_by_source[source_id]).text
        ai_text = rng.choice(ai_by_source[source_id]).text
        doc = splice(
            human_text,
            ai_text,
            rng,
            doc_id=f"splice-{source_id}-{made}",
            source=f"splice/{rng.choice(ai_by_source[source_id]).domain}",
        )
        if doc is None:
            continue
        add(doc, source_id)
        made += 1
        bar.update(1)
    bar.close()

    # --- documents that already carry ground-truth spans -----------------------
    if spec.extra_documents:
        n_docs = 0
        for path in str(spec.extra_documents).split(","):
            path = path.strip()
            if not path or not Path(path).exists():
                if path:
                    print(f"[build] document shard missing, skipped: {path}")
                continue
            for payload in iter_json_lines(path):
                doc = Document.from_json(payload)
                add(doc, doc.id)
                n_docs += 1
        print(f"[build] added {n_docs} documents with ground-truth spans")

    # --- extra human and AI text from public HuggingFace corpora ---------------
    if spec.hf_samples:
        n_hf = _add_hf_samples(spec, rng, add, progress)
        print(f"[build] added {n_hf} documents from HuggingFace corpora")

    # --- homogeneous mixed text: AI-edited human documents ---------------------
    pair_files: list[str] = []
    for group in (spec.edited_pairs, spec.extra_edited_pairs):
        if group:
            pair_files.extend(f.strip() for f in str(group).split(",") if f.strip())
    if pair_files:
        n_edited = _add_edited_pairs(spec, pair_files, add, progress, rng)
        print(f"[build] labeled {n_edited} AI-edited pairs from {len(pair_files)} file(s)")

    print(f"[build] kept {counters['kept']}, dropped {counters['duplicate']} duplicates "
          f"and {counters['short']} short documents")

    out_dir = Path(spec.out_dir)
    counts: dict[str, int] = {}
    for split, docs in buckets.items():
        rng.shuffle(docs)
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", docs)
    (out_dir / "build_spec.json").write_text(json.dumps(spec.__dict__, indent=2))
    return counts


def _add_edited_pairs(
    spec: BuildSpec, pair_files: list[str], add, progress: bool, rng: random.Random
) -> int:
    embedder = Embedder(spec.embedder) if spec.embedder else NullEmbedder()
    sn_cfg = SoftNGramConfig()
    pairs: list[dict] = []
    for path in pair_files:
        if not Path(path).exists():
            print(f"[build] pairs file missing, skipped: {path}")
            continue
        chunk = list(iter_json_lines(path))
        print(f"[build]   {len(chunk)} pairs from {path}")
        pairs.extend(chunk)
    rng.shuffle(pairs)
    if spec.max_edited_pairs and len(pairs) > spec.max_edited_pairs:
        print(f"[build] capping {len(pairs)} pairs to {spec.max_edited_pairs}")
        pairs = pairs[: spec.max_edited_pairs]
    kept = [p for p in pairs if _word_count(p.get("target", "")) >= spec.min_words]
    print(f"[build] labeling {len(kept)} pairs with Soft N-Grams")
    results = label_edits(
        [(p["source"], p["target"]) for p in kept], embedder, sn_cfg, progress=progress
    )

    made = 0
    for pair, (labeled, f_ai) in zip(kept, results):
        if not labeled:
            continue
        target = pair["target"]
        spans: list[Span] = []
        for lc in labeled:
            if spans and spans[-1].label == lc.label:
                spans[-1] = Span(spans[-1].start, lc.end, lc.label)
            else:
                spans.append(Span(lc.start, lc.end, lc.label))
        doc = Document(
            id=f"edited-{pair.get('id', made)}",
            text=target,
            spans=spans,
            humanizer=int(Humanizer.MIXED_AUTHORSHIP),
            source=pair.get("source_name", "edited"),
            meta={
                "f_ai": f_ai,
                "instruction": pair.get("instruction", ""),
                "generator": pair.get("generator", "local"),
                "tier": pair.get("tier", "local"),
            },
        )
        add(doc, str(pair.get("source_id", pair.get("id", made))))
        made += 1
    return made


def _add_hf_samples(spec: BuildSpec, rng: random.Random, add, progress: bool) -> int:
    """Add HuggingFace human/AI documents, plus splices from topic-matched pairs."""
    from .hf_sources import AI as HF_AI
    from .hf_sources import HUMAN as HF_HUMAN
    from .hf_sources import read_samples

    samples: list = []
    for path in str(spec.hf_samples).split(","):
        path = path.strip()
        if not path:
            continue
        if not Path(path).exists():
            print(f"[build] hf samples file missing, skipped: {path}")
            continue
        chunk = read_samples(path)
        print(f"[build]   {len(chunk)} samples from {path}")
        samples.extend(chunk)
    human = [s for s in samples if s.label == HF_HUMAN]
    ai = [s for s in samples if s.label == HF_AI]
    if spec.legacy_ai_keep < 1.0:
        legacy_markers = (
            "gpt-3.5", "gpt-4-0314", "gpt-4\n", "gpt2", "davinci", "bloomz", "opt_",
            "flan_t5", "gpt_neox", "gpt_j", "llama-chat", "mistral-chat", "mpt",
            "claude-evol", "dolly", "cohere", "t0_", "dmitva-ai",
        )
        kept, dropped = [], 0
        for sample in ai:
            g = (sample.generator or "").lower()
            if any(m in g for m in legacy_markers) and rng.random() > spec.legacy_ai_keep:
                dropped += 1
                continue
            kept.append(sample)
        print(f"[build] dropped {dropped} legacy-generator AI samples "
              f"(keep={spec.legacy_ai_keep})")
        ai = kept
    rng.shuffle(ai)
    cap = int(spec.hf_ai_per_human * len(human))
    # Keep every topic-matched AI half: those pairs feed the splicer.
    paired = [s for s in ai if s.pair_id]
    unpaired = [s for s in ai if not s.pair_id][: max(0, cap - len(paired))]
    ai = paired + unpaired
    if spec.match_domain_mix:
        before = len(human)
        human = resample_to_domain_mix(human, PAPER_DOMAIN_MIX, rng,
                                       min_keep=spec.domain_mix_min_keep)
        mix = collections.Counter(domain_of(x.domain) for x in human)
        total = max(1, sum(mix.values()))
        print(f"[build] domain mix {before} -> {len(human)} human: "
              + ", ".join(f"{d}={mix[d]/total:.1%}" for d in PAPER_DOMAIN_MIX))

    if spec.human_per_ai > 0 and len(human) > spec.human_per_ai * len(ai):
        keep = int(spec.human_per_ai * len(ai))
        rng.shuffle(human)
        # Keep the pool balanced across sources rather than truncating whichever
        # ones happen to sort first.
        by_source: dict[str, list] = defaultdict(list)
        for sample in human:
            by_source[sample.domain.split("/")[0]].append(sample)
        balanced: list = []
        pools = {k: v for k, v in by_source.items() if v}
        while pools and len(balanced) < keep:
            for key in list(pools):
                if not pools[key]:
                    del pools[key]
                    continue
                balanced.append(pools[key].pop())
                if len(balanced) >= keep:
                    break
        print(f"[build] capped human pool {len(human)} -> {len(balanced)} "
              f"({spec.human_per_ai}x the AI pool)")
        human = balanced

    samples = human + ai
    rng.shuffle(samples)
    print(f"[build] hf pool: {len(human)} human, {len(ai)} ai (cap {cap})")
    made = 0
    pairs: dict[str, dict[str, str]] = {}

    for i, sample in enumerate(tqdm(samples, desc="hf-docs", disable=not progress)):
        label = Provenance.HUMAN if sample.label == HF_HUMAN else Provenance.AI
        humanizer = Humanizer.HUMAN if sample.label == HF_HUMAN else Humanizer.AI_GENERATED
        doc = Document.uniform(
            f"hf-{i}",
            sample.text,
            label,
            humanizer,
            sample.domain,
            domain=sample.domain,
            generator=sample.generator,
        )
        roll = rng.random()
        if sample.label == HF_AI and roll < spec.own_attack_rate:
            names = rng.sample(list(EVASION_ATTACKS), rng.randint(1, 2))
            doc = apply_transforms(doc, names, rng, int(Humanizer.HUMANIZED_AI), "-evade")
        elif roll < spec.own_attack_rate + spec.incidental_rate:
            doc = apply_transforms(doc, [rng.choice(list(INCIDENTAL_ARTIFACTS))], rng, suffix="-inc")

        # Keep a topic-matched pair together; otherwise let `add` fall back to
        # the content fingerprint rather than a positional index.
        add(doc, sample.pair_id or "")
        made += 1

        if sample.pair_id:
            pairs.setdefault(sample.pair_id, {})[sample.label] = sample.text

    # Topic-matched human/AI pairs make ideal heterogeneous mixed documents.
    complete = [k for k, v in pairs.items() if HF_HUMAN in v and HF_AI in v]
    rng.shuffle(complete)
    spliced = 0
    for pair_id in complete:
        if spliced >= spec.n_hf_spliced:
            break
        doc = splice(
            pairs[pair_id][HF_HUMAN],
            pairs[pair_id][HF_AI],
            rng,
            doc_id=f"hf-splice-{pair_id}",
            source="splice/hf",
        )
        if doc is None:
            continue
        add(doc, pair_id)
        spliced += 1
        made += 1

    # Most sources have no explicit pairing, so also splice *within a language*:
    # a French human passage with a French AI passage. Without this the mixed
    # class would be English-only, and the detector would have no idea what a
    # half-AI French document looks like.
    def lang_of(sample) -> str:
        parts = sample.domain.split("/")
        return parts[-1] if len(parts) > 1 else parts[0]

    by_lang_h: dict[str, list[str]] = defaultdict(list)
    by_lang_a: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.pair_id:
            continue
        (by_lang_h if sample.label == HF_HUMAN else by_lang_a)[lang_of(sample)].append(sample.text)

    cross = 0
    shared_langs = sorted(set(by_lang_h) & set(by_lang_a))
    budget = max(0, spec.n_hf_spliced - spliced)
    per_lang = max(1, budget // max(1, len(shared_langs))) if shared_langs else 0
    for lang in shared_langs:
        humans, ais = by_lang_h[lang], by_lang_a[lang]
        for i in range(min(per_lang, len(humans), len(ais))):
            doc = splice(
                rng.choice(humans),
                rng.choice(ais),
                rng,
                doc_id=f"hf-xsplice-{lang}-{i}",
                source=f"splice/{lang}",
            )
            if doc is None:
                continue
            add(doc, f"xsplice-{lang}-{i}")
            cross += 1
            made += 1
    print(
        f"[build] spliced {spliced} paired + {cross} within-language heterogeneous documents "
        f"across {len(shared_langs)} language/domain groups"
    )
    return made


#: Report Figure 2: approximate share of human source text by category.
PAPER_DOMAIN_MIX = {
    "creative": 0.222,
    "scientific": 0.180,
    "reference": 0.159,
    "reviews": 0.106,
    "social": 0.104,
    "web": 0.082,
    "news": 0.066,
    "essays": 0.049,
    "professional": 0.031,
}

#: Which of our sources feeds each of those categories.
SOURCE_DOMAIN = {
    "cosmo_stories": "creative", "wp": "creative", "roct": "creative",
    "mage": "creative", "hswag": "creative",
    "cosmo_stanford": "scientific", "cosmo_openstax": "scientific",
    "abstracts": "scientific", "sci_gen": "scientific", "pubmed": "scientific",
    "wikipedia": "reference", "cosmo_khanacademy": "reference",
    "cosmo_wikihow": "reference", "squad": "reference", "aya": "reference",
    "yelp": "reviews", "reviews": "reviews", "amazon": "reviews",
    "sharechat": "social", "wildchat": "social", "wildchat48": "social",
    "reddit": "social", "eli5": "social", "cmv": "social", "tldr": "social",
    "ultrachat": "social", "openhermes": "social", "french_instruct": "social",
    "fineweb": "web", "detection_pile": "web", "cosmopedia": "web",
    "cosmo_web_samples_v1": "web", "manus_frontier": "web",
    "news": "news", "xsum": "news", "coling": "news", "industry_news": "news",
    "dmitva": "essays", "essays": "essays", "mgt_multi": "essays",
    "industry_finance": "professional", "industry_law": "professional",
    "industry_education": "professional", "professional": "professional",
}


def domain_of(source: str) -> str:
    """Best-effort mapping of a sample's source onto a Figure 2 category."""
    head = source.split("/")[0].lower()
    if head in SOURCE_DOMAIN:
        return SOURCE_DOMAIN[head]
    for key, value in SOURCE_DOMAIN.items():
        if key in source.lower():
            return value
    return "web"


def resample_to_domain_mix(
    samples: list, target: dict[str, float], rng: random.Random, min_keep: float = 0.0
) -> list:
    """Subsample so the category shares approach ``target``.

    Only ever drops: the mix is reached by trimming over-represented categories
    to whatever the scarcest category can support, never by duplicating, which
    would just teach the same documents twice.

    ``min_keep`` bounds how much may be thrown away: when a category is nearly
    empty, matching the mix exactly would shrink the whole pool to that
    category's size. The budget is then raised until at least this fraction of
    the pool survives -- over-represented categories are still trimmed first,
    the scarce ones are kept whole.
    """
    from collections import defaultdict

    by_domain: dict[str, list] = defaultdict(list)
    for sample in samples:
        by_domain[domain_of(sample.domain)].append(sample)
    if not by_domain:
        return samples

    # Largest total that keeps every category at or below what it actually has.
    budget = min(
        (len(by_domain[d]) / share for d, share in target.items() if by_domain.get(d)),
        default=0.0,
    )
    if budget <= 0:
        return samples

    def kept(b: float) -> int:
        return sum(min(len(by_domain.get(d, [])), int(round(sh * b))) for d, sh in target.items())

    floor = min_keep * len(samples)
    if kept(budget) < floor:
        lo, hi = budget, budget
        while kept(hi) < floor and hi < 1e9:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if kept(mid) < floor else (lo, mid)
        budget = hi

    out: list = []
    for domain, share in target.items():
        pool = by_domain.get(domain, [])
        if not pool:
            continue
        keep = min(len(pool), int(round(share * budget)))
        rng.shuffle(pool)
        out.extend(pool[:keep])
    rng.shuffle(out)
    return out
