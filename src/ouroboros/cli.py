"""Ouroboros command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(add_completion=False, help="Open reimplementation of Pangram 4 (arXiv:2607.27183)")


@app.command("build-data")
def build_data(
    raid_csv: str = typer.Option(..., help="Path to RAID train.csv"),
    out_dir: str = typer.Option("data/corpus"),
    n_human: int = 24000,
    n_ai: int = 24000,
    n_humanized: int = 7000,
    n_spliced: int = 9000,
    test_fraction: float = 0.08,
    ai_per_human: float = 1.0,
    hf_ai_per_human: float = 1.0,
    match_domain_mix: bool = typer.Option(False, help="Resample human pool to the report's Figure 2 mix"),
    domain_mix_min_keep: float = typer.Option(0.5, help="Never keep less than this fraction of the human pool when matching the mix"),
    human_per_ai: float = typer.Option(0.0, help="Cap human pool at N x the AI pool (0 = no cap)"),
    legacy_ai_keep: float = typer.Option(1.0, help="Keep this fraction of pre-2024-generator AI text"),
    humanized_per_human: float = 0.5,
    edited_pairs: Optional[str] = typer.Option(None, help="JSONL from `ouroboros edit`"),
    extra_edited_pairs: Optional[str] = typer.Option(None, help="JSONL from `ouroboros api-edit`"),
    hf_samples: Optional[str] = typer.Option(None, help="Comma-separated JSONL files from `hf-fetch`"),
    extra_documents: Optional[str] = typer.Option(None, help="JSONL of Documents with spans"),
    max_edited_pairs: int = typer.Option(0, help="Cap on Soft N-Grams labeling (0 = no cap)"),
    embedder: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: Optional[str] = typer.Option(None, help="Cache dir for the RAID subset scan"),
    seed: int = 0,
):
    """Assemble the span-annotated training corpus."""
    from .data.build import BuildSpec, build_corpus

    spec = BuildSpec(
        raid_csv=raid_csv,
        out_dir=out_dir,
        n_human=n_human,
        n_ai=n_ai,
        n_humanized=n_humanized,
        n_spliced=n_spliced,
        test_fraction=test_fraction,
        ai_per_human=ai_per_human,
        hf_ai_per_human=hf_ai_per_human,
        legacy_ai_keep=legacy_ai_keep,
        human_per_ai=human_per_ai,
        match_domain_mix=match_domain_mix,
        domain_mix_min_keep=domain_mix_min_keep,
        humanized_per_human=humanized_per_human,
        edited_pairs=edited_pairs,
        extra_edited_pairs=extra_edited_pairs,
        hf_samples=hf_samples,
        extra_documents=extra_documents,
        max_edited_pairs=max_edited_pairs,
        embedder=embedder,
        cache_dir=cache_dir,
        seed=seed,
    )
    counts = build_corpus(spec)
    rprint({"written": counts, "out_dir": out_dir})


@app.command("hf-fetch")
def hf_fetch(
    out: str = typer.Option("data/hf_samples.jsonl"),
    recipes: str = typer.Option(
        "mage,coling,detection_pile,dmitva,cosmopedia,wildchat,ultrachat,openhermes",
        help="Comma-separated recipe keys",
    ),
    n_human: int = typer.Option(7000, help="Human documents to draw per recipe"),
    n_ai: int = typer.Option(5000, help="AI documents to draw per recipe"),
    min_chars: int = 500,
    max_rows: int = 400000,
    seed: int = 0,
):
    """Sample additional human and AI text from public HuggingFace corpora."""
    from .data.hf_sources import RECIPES, sample_recipe, write_samples

    keys = [k.strip() for k in recipes.split(",") if k.strip()]
    unknown = [k for k in keys if k not in RECIPES]
    if unknown:
        raise typer.BadParameter(f"unknown recipes {unknown}; available: {sorted(RECIPES)}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("")  # fresh file; write_samples appends per recipe
    totals: dict[str, dict[str, int]] = {}
    for key in keys:
        recipe = RECIPES[key]
        try:
            samples = sample_recipe(
                recipe, n_human=n_human, n_ai=n_ai, min_chars=min_chars,
                max_rows=max_rows, seed=seed,
            )
        except Exception as exc:  # a single unavailable dataset must not abort the run
            rprint(f"[yellow]skipped {key}: {type(exc).__name__}: {str(exc)[:140]}[/yellow]")
            continue
        write_samples(out, samples, key)
        counts = {"human": 0, "ai": 0}
        for sample in samples:
            counts[sample.label] += 1
        totals[key] = counts
        rprint(f"  {key}: {counts}")

    grand = {
        "human": sum(v["human"] for v in totals.values()),
        "ai": sum(v["ai"] for v in totals.values()),
    }
    rprint({"per_recipe": totals, "total": grand, "out": out})


@app.command("scan")
def scan_cmd(
    raid_csv: str = typer.Option(..., help="Path to RAID train.csv"),
    cache_dir: str = typer.Option(..., help="Where to cache the sampled subset"),
    n_human: int = 24000,
    n_ai: int = 24000,
    n_humanized: int = 7000,
    seed: int = 0,
):
    """Sample RAID once and cache it, so later steps skip the 12 GB CSV pass."""
    from .data.sources import read_raid

    subset = read_raid(
        raid_csv,
        n_human=n_human,
        n_ai=n_ai,
        n_humanized=n_humanized,
        seed=seed,
        cache_dir=cache_dir,
    )
    rprint(subset.summary())


@app.command("mirror")
def mirror(
    raid_csv: str = typer.Option(..., help="RAID csv to draw human documents from"),
    out: str = typer.Option("data/mirrors.jsonl"),
    n: int = 2000,
    client: str = typer.Option("local:Qwen/Qwen2.5-1.5B-Instruct"),
    cache_dir: Optional[str] = typer.Option(None),
    n_human: int = typer.Option(24000, help="Must match `build-data` to reuse its cached scan"),
    n_ai: int = 24000,
    n_humanized: int = 7000,
    seed: int = 0,
):
    """Generate synthetic mirrors of human documents (report Section 2.2)."""
    from .data.generation import build_client
    from .data.mirror import synthetic_mirror
    from .data.sources import read_raid

    subset = read_raid(
        raid_csv,
        n_human=n_human,
        n_ai=n_ai,
        n_humanized=n_humanized,
        seed=seed,
        cache_dir=cache_dir,
    )
    samples = subset.human[:n]
    llm = build_client(client)
    mirrors = synthetic_mirror(
        llm, [s.text for s in samples], [s.title for s in samples]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i, m in enumerate(mirrors):
            fh.write(json.dumps({"id": f"mirror-{i}", **m}, ensure_ascii=False) + "\n")
    rprint(f"wrote {len(mirrors)} mirrors (from {len(samples)} sources) -> {out}")


@app.command("edit")
def edit(
    raid_csv: str = typer.Option(..., help="RAID csv to draw human documents from"),
    out: str = typer.Option("data/edited_pairs.jsonl"),
    n: int = 3000,
    client: str = typer.Option("local:Qwen/Qwen2.5-1.5B-Instruct"),
    max_source_chars: int = 3000,
    batch_size: int = typer.Option(24, help="Generation batch size"),
    cache_dir: Optional[str] = typer.Option(None),
    n_human: int = typer.Option(24000, help="Must match `build-data` to reuse its cached scan"),
    n_ai: int = 24000,
    n_humanized: int = 7000,
    seed: int = 0,
):
    """Generate AI-edited (homogeneous mixed) text from human documents."""
    from .data.edits import generate_edits
    from .data.generation import build_client
    from .data.sources import read_raid

    subset = read_raid(
        raid_csv,
        n_human=n_human,
        n_ai=n_ai,
        n_humanized=n_humanized,
        seed=seed,
        cache_dir=cache_dir,
    )
    samples = subset.human[:n]
    llm = build_client(client, batch_size=batch_size) if client.startswith("local:") else build_client(client)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("")  # streamed to disk chunk by chunk below
    pairs = generate_edits(
        llm,
        [s.text[:max_source_chars] for s in samples],
        [s.source_id for s in samples],
        seed=seed,
        out_path=out,
    )
    rprint(f"wrote {len(pairs)} edited pairs -> {out}")


@app.command("api-edit")
def api_edit(
    raid_csv: str = typer.Option(..., help="RAID csv to draw human documents from"),
    out: str = typer.Option("data/api_edited_pairs.jsonl"),
    n: int = typer.Option(9000, help="How many human documents to rewrite"),
    skip: int = typer.Option(0, help="Skip this many human documents (avoid overlap)"),
    cache_dir: Optional[str] = typer.Option(None),
    n_human: int = 24000,
    n_ai: int = 24000,
    n_humanized: int = 7000,
    budget: float = typer.Option(2.40, help="Hard total spend cap in USD"),
    max_source_chars: int = 2600,
    max_tokens: int = 900,
    workers: int = 12,
    report_out: Optional[str] = typer.Option(None, help="Where to write the spend report"),
    seed: int = 0,
):
    """Rewrite human documents with a diverse portfolio of models via OpenRouter.

    This is report Section 2.3 (AI-assisted text) run across many generators:
    each human document is handed to a randomly chosen model with a randomly
    chosen edit instruction, from light copyediting to a full rewrite. The
    resulting pairs are labeled clause-by-clause by the Soft N-Grams labeler in
    `build-data`.
    """
    import random as _random

    from .data.edits import EDIT_INSTRUCTIONS, PROMPT
    from .data.openrouter import DEFAULT_TIER_BUDGETS, DEFAULT_TIER_VOLUME, run_portfolio
    from .data.sources import read_raid

    subset = read_raid(
        raid_csv,
        n_human=n_human,
        n_ai=n_ai,
        n_humanized=n_humanized,
        seed=seed,
        cache_dir=cache_dir,
    )
    samples = subset.human[skip : skip + n]
    if not samples:
        raise typer.BadParameter("no human documents left after --skip")

    rng = _random.Random(seed)
    tasks = []
    for sample in samples:
        intensity, instruction = rng.choice(EDIT_INSTRUCTIONS)
        tasks.append(
            {
                "id": sample.id,
                "source_id": sample.source_id,
                "source": sample.text[:max_source_chars],
                "instruction": instruction,
                "intensity": intensity,
                "domain": sample.domain,
            }
        )

    scale = budget / sum(DEFAULT_TIER_BUDGETS.values())
    tier_budgets = {k: v * scale for k, v in DEFAULT_TIER_BUDGETS.items()}

    def prompt_fn(task, model):
        return PROMPT.format(instruction=task["instruction"], document=task["source"])

    def record_fn(task, model, text):
        text = text.strip()
        # Reject non-answers and models that echoed the prompt back.
        if len(text) < 300 or text == task["source"].strip():
            return None
        return {
            "id": f'{task["id"]}-{model.id.replace("/", "_")}',
            "source_id": task["source_id"],
            "source": task["source"],
            "target": text,
            "instruction": task["instruction"],
            "intensity": task["intensity"],
            "generator": model.id,
            "tier": model.tier,
            "domain": task["domain"],
            "source_name": "api-edited",
        }

    report = run_portfolio(
        tasks,
        prompt_fn,
        record_fn,
        out_path=out,
        tier_budgets=tier_budgets,
        tier_volume=DEFAULT_TIER_VOLUME,
        max_tokens=max_tokens,
        workers=workers,
        seed=seed,
    )
    rprint(report)
    if report_out:
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(report_out).write_text(json.dumps(report, indent=2))
        rprint(f"spend report -> {report_out}")


@app.command("edit-hf")
def edit_hf(
    hf_samples: str = typer.Option(..., help="JSONL from `ouroboros hf-fetch`"),
    out: str = typer.Option("data/ml_edited_pairs.jsonl"),
    recipes: str = typer.Option(
        "wikipedia_fr,fineweb_fr,french_instruct,aya,wikipedia_es,wikipedia_de,wikipedia_it,wikipedia_pt",
        help="Only use human samples coming from these recipes",
    ),
    n: int = typer.Option(3000),
    client: str = typer.Option("local:Qwen/Qwen2.5-1.5B-Instruct"),
    batch_size: int = 24,
    max_source_chars: int = 2200,
    max_minutes: float = typer.Option(45.0, help="Stop generating after this long"),
    seed: int = 0,
):
    """Generate AI-edited (co-written) pairs in languages other than English.

    Without this the `ai-assisted` class is English-only: the heterogeneous
    splicer can mix languages, but homogeneous co-writing needs real
    (source, edited) pairs, and those only exist where we generate them.
    """
    import random as _random
    import time as _time

    from .data.edits import EDIT_INSTRUCTIONS, PROMPT_KEEP_LANGUAGE, EditConfig, generate_edits
    from .data.generation import build_client
    from .data.hf_sources import HUMAN, read_samples

    wanted = {r.strip() for r in recipes.split(",") if r.strip()}
    samples = [
        s
        for s in read_samples(hf_samples)
        if s.label == HUMAN and s.domain.split("/")[0] in wanted
    ]
    if not samples:
        raise typer.BadParameter(f"no human samples from {sorted(wanted)}")

    rng = _random.Random(seed)
    rng.shuffle(samples)
    samples = samples[:n]
    by_recipe: dict[str, int] = {}
    for s in samples:
        key = s.domain.split("/")[0]
        by_recipe[key] = by_recipe.get(key, 0) + 1
    rprint({"sources": by_recipe, "total": len(samples)})

    llm = build_client(client, batch_size=batch_size) if client.startswith("local:") else build_client(client)
    # Chunked generation with a wall-clock budget: this runs on the same GPU the
    # training needs next, so it must stop on time rather than when it is done.
    import ouroboros.data.edits as edits_mod

    edits_mod.PROMPT = PROMPT_KEEP_LANGUAGE

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("")
    deadline = _time.time() + max_minutes * 60
    texts = [s.text[:max_source_chars] for s in samples]
    ids = [f"ml-{i}" for i in range(len(samples))]
    written = 0
    chunk = 240
    for start in range(0, len(texts), chunk):
        if _time.time() > deadline:
            rprint(f"[yellow]time budget reached after {written} pairs[/yellow]")
            break
        pairs = generate_edits(
            llm,
            texts[start : start + chunk],
            ids[start : start + chunk],
            cfg=EditConfig(max_tokens=640, min_chars=250),
            seed=seed + start,
            progress=True,
            out_path=out,
            chunk_size=chunk,
        )
        written += len(pairs)
    rprint(f"wrote {written} multilingual edited pairs -> {out}")


@app.command("hf-pairs")
def hf_pairs(
    out: str = typer.Option("data/hf_pairs.jsonl"),
    recipes: str = typer.Option("chatgpt_paraphrase,mgt_pairs"),
    n: int = typer.Option(40000, help="Pairs per recipe"),
    min_chars: int = 300,
    max_rows: int = 600000,
    seed: int = 0,
):
    """Harvest (source, AI-rewritten) pairs from the Hub for Soft N-Grams labeling."""
    from .data.hf_sources import PAIR_RECIPES, sample_pairs

    keys = [k.strip() for k in recipes.split(",") if k.strip()]
    unknown = [k for k in keys if k not in PAIR_RECIPES]
    if unknown:
        raise typer.BadParameter(f"unknown pair recipes {unknown}; have {sorted(PAIR_RECIPES)}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out, "w", encoding="utf-8") as fh:
        for key in keys:
            try:
                pairs = sample_pairs(key, n=n, min_chars=min_chars, max_rows=max_rows, seed=seed)
            except Exception as exc:
                rprint(f"[yellow]skipped {key}: {type(exc).__name__}: {str(exc)[:120]}[/yellow]")
                continue
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            total += len(pairs)
            rprint(f"  {key}: {len(pairs)} pairs")
    rprint({"total_pairs": total, "out": out})


@app.command("hf-boundary")
def hf_boundary(
    out: str = typer.Option("data/boundary_docs.jsonl"),
    n: int = typer.Option(20000),
    min_words: int = 60,
    seed: int = 0,
):
    """Harvest documents that already carry a real human->AI change point."""
    from .data.hf_sources import boundary_documents

    docs = boundary_documents(n=n, min_words=min_words, seed=seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    rprint({"documents": len(docs), "out": out})


@app.command("train")
def train_cmd(
    config: str = typer.Option(..., help="YAML config for stage 1"),
    stage2_config: Optional[str] = typer.Option(None, help="YAML config for stage 2"),
    skip_stage1: bool = False,
    override: list[str] = typer.Option([], "--set", help="key.path=value override"),
):
    """Run the two-stage training procedure (report Section 4.2)."""
    from .config import load_config
    from .train.stage import run_stages

    cfg1 = load_config(config, override)
    cfg2 = load_config(stage2_config, override) if stage2_config else None
    ckpt = run_stages(cfg1, cfg2, skip_stage1=skip_stage1)
    rprint(f"checkpoint: {ckpt}")


@app.command("calibrate")
def calibrate_cmd(
    run_dir: str = typer.Option(..., help="Training run directory"),
    shards: list[str] = typer.Option(..., help="Calibration JSONL shard(s)"),
    limit: Optional[int] = typer.Option(600),
    target_fpr: float = 0.005,
):
    """Fit the global calibrator and the CRF penalties on held-out data."""
    from .data.documents import read_shards
    from .infer.fit_calibration import fit_calibrator
    from .infer.predict import Predictor

    predictor = Predictor.from_run(run_dir)
    docs = read_shards(shards)
    if limit:
        docs = docs[:limit]
    calibrator, info = fit_calibrator(predictor, docs, target_fpr=target_fpr)
    calibrator.save(Path(run_dir) / "calibrator.json")
    rprint(info)


@app.command("eval")
def eval_cmd(
    run_dir: str = typer.Option(...),
    shards: list[str] = typer.Option(...),
    limit: Optional[int] = typer.Option(None),
    out: Optional[str] = typer.Option(None),
    humanizer: bool = typer.Option(True, help="Also evaluate the humanizer probe"),
):
    """Evaluate a trained checkpoint (report Section 5)."""
    from .data.documents import read_shards
    from .eval.run_eval import evaluate_corpus, evaluate_humanizer, save_report
    from .infer.predict import Predictor

    predictor = Predictor.from_run(run_dir)
    docs = read_shards(shards)
    if limit:
        docs = docs[:limit]
    report = evaluate_corpus(predictor, docs)
    if humanizer:
        report["humanizer"] = evaluate_humanizer(predictor, docs, limit=limit or 1500)

    table = Table(title="Document-level (Section 5.1 rule)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in report["document"].items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    for key, value in report.get("ranking", {}).items():
        table.add_row(key, f"{value:.4f}")
    for key in ("token_accuracy", "token_macro_f1"):
        if key in report["token"]:
            table.add_row(key, f"{report['token'][key]:.4f}")
    for key, value in report.get("boundary", {}).items():
        table.add_row(key, f"{value:.4f}")
    if "humanizer" in report:
        table.add_row("humanizer_accuracy", f"{report['humanizer']['accuracy']:.4f}")
    rprint(table)

    if out:
        save_report(report, out)
        rprint(f"report -> {out}")


@app.command("eval-segment")
def eval_segment_cmd(
    run_dir: str = typer.Option(...),
    shards: list[str] = typer.Option(...),
    limit: Optional[int] = typer.Option(1000),
    out: Optional[str] = typer.Option(None),
):
    """Document-level detection using the segment head alone (works after stage 1)."""
    from .data.documents import read_shards
    from .eval.run_eval import evaluate_segment_head, save_report
    from .infer.predict import Predictor

    predictor = Predictor.from_run(run_dir)
    report = evaluate_segment_head(predictor, read_shards(shards), limit=limit)

    table = Table(title=f"Segment head only — {run_dir}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in report.items():
        if isinstance(value, dict):
            for k2, v2 in value.items():
                table.add_row(f"{key}.{k2}", f"{v2:.4f}" if isinstance(v2, float) else str(v2))
        else:
            table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    rprint(table)
    if out:
        save_report(report, out)


@app.command("mine")
def mine_cmd(
    run_dir: str = typer.Option(..., help="Checkpoint to mine with"),
    shards: list[str] = typer.Option(..., help="Reserved pool to run inference over"),
    out: str = typer.Option("data/hard_cases.jsonl"),
    limit: Optional[int] = typer.Option(None),
    margin: float = typer.Option(
        0.0, help="Also keep near-misses: human scored above / AI scored below this"
    ),
):
    """Mine hard negatives and hard positives (report Section 4.2, active learning)."""
    from .data.documents import read_shards
    from .eval.mine import mine_hard_cases, save_mined
    from .infer.predict import Predictor

    predictor = Predictor.from_run(run_dir)
    docs = read_shards(shards)
    if limit:
        docs = docs[:limit]
    hard, stats = mine_hard_cases(predictor, docs, margin=margin)
    save_mined(out, hard, stats)

    table = Table(title=f"Active learning — {run_dir}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in stats.as_dict().items():
        table.add_row(key, str(value))
    rprint(table)
    rprint(f"{len(hard)} hard cases -> {out}")


@app.command("predict")
def predict_cmd(
    run_dir: str = typer.Option(...),
    text: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None),
    show_segments: bool = True,
):
    """Score a document and print its authorship segments."""
    from .infer.predict import Predictor

    if not text and not file:
        raise typer.BadParameter("pass --text or --file")
    content = text or Path(file).read_text(encoding="utf-8")

    predictor = Predictor.from_run(run_dir)
    prediction = predictor.predict(content)
    rprint(f"[bold]{prediction.summary()}[/bold]")
    rprint({"humanizer": {k: round(v, 4) for k, v in predictor.humanizer(content).items()}})

    if show_segments:
        table = Table(title="Segments")
        table.add_column("label")
        table.add_column("conf", justify="right")
        table.add_column("chars", justify="right")
        table.add_column("text")
        for segment in prediction.segments:
            excerpt = content[segment.start : segment.end].replace("\n", " ")
            if len(excerpt) > 90:
                excerpt = excerpt[:87] + "..."
            table.add_row(
                segment.label_name,
                f"{segment.confidence:.3f}",
                f"{segment.start}-{segment.end}",
                excerpt,
            )
        rprint(table)


if __name__ == "__main__":
    app()
