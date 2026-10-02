"""Corpus readers.

The report trains on in-house data built by synthetic mirroring (Section 2.2).
We cannot reproduce that corpus, so the default open recipe uses RAID
(Dugan et al.), which already pairs every human document with AI generations
produced from the *same* source prompt across many generators and domains --
the same topic-invariance property synthetic mirroring is designed to give --
and additionally ships twelve adversarial attack variants, which supervise the
humanizer head of Section 3.4.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm.auto import tqdm

RAID_COLUMNS = ["id", "source_id", "model", "attack", "domain", "title", "generation"]

#: RAID attacks that are deliberate detector evasion (report Section 3.4).
RAID_EVASION_ATTACKS = {
    "alternative_spelling",
    "article_deletion",
    "homoglyph",
    "insert_paragraphs",
    "lower_upper",
    "upper_lower",
    "misspelling",
    "number",
    "paraphrase",
    "perplexity_misspelling",
    "synonym",
    "whitespace",
    "zero_width_space",
}


@dataclass
class RaidSample:
    id: str
    source_id: str
    model: str
    attack: str
    domain: str
    title: str
    text: str


@dataclass
class RaidSubset:
    human: list[RaidSample] = field(default_factory=list)
    ai: list[RaidSample] = field(default_factory=list)
    humanized: list[RaidSample] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {"human": len(self.human), "ai": len(self.ai), "humanized": len(self.humanized)}


class _Reservoir:
    """Fixed-capacity uniform reservoir over a stream."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def offer(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = item


def read_raid(
    path: str | Path,
    n_human: int = 20000,
    n_ai: int = 20000,
    n_humanized: int = 6000,
    min_chars: int = 400,
    chunksize: int = 200_000,
    seed: int = 0,
    progress: bool = True,
    cache_dir: str | Path | None = None,
) -> RaidSubset:
    """Single streaming pass over RAID with per-(stratum, domain) reservoirs.

    RAID's CSV is ordered by domain and by attack, so a prefix read would be
    badly biased; reservoir sampling gives a uniform draw in one pass. The
    resulting subset is cached, because that pass reads ~12 GB of CSV and both
    the edit-generation and corpus-build steps need the same sample.
    """
    cache_path = None
    if cache_dir is not None:
        key = f"{Path(path).name}-{n_human}-{n_ai}-{n_humanized}-{min_chars}-{seed}"
        cache_path = Path(cache_dir) / f"raid-subset-{key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as fh:
                subset = pickle.load(fh)
            print(f"[raid] loaded cached subset {subset.summary()} from {cache_path}")
            return subset

    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], _Reservoir] = {}
    # Domain count is known after the first chunk; we allocate lazily with a
    # generous per-domain capacity and trim to the requested totals at the end.
    per_domain = {"human": n_human, "ai": n_ai, "humanized": n_humanized}

    reader = pd.read_csv(
        path,
        usecols=RAID_COLUMNS,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        engine="c",
    )
    bar = tqdm(reader, desc="scan raid", unit="chunk") if progress else reader
    for chunk in bar:
        text = chunk["generation"].str.strip()
        chunk = chunk[text.str.len() >= min_chars]
        if chunk.empty:
            continue
        is_human = chunk["model"] == "human"
        no_attack = chunk["attack"] == "none"
        strata = pd.Series("drop", index=chunk.index)
        strata[is_human & no_attack] = "human"
        strata[~is_human & no_attack] = "ai"
        strata[~is_human & chunk["attack"].isin(RAID_EVASION_ATTACKS)] = "humanized"
        chunk = chunk[strata != "drop"]
        if chunk.empty:
            continue
        for stratum, row in zip(strata[chunk.index], chunk.itertuples(index=False)):
            key = (stratum, row.domain)
            res = reservoirs.get(key)
            if res is None:
                res = reservoirs[key] = _Reservoir(per_domain[stratum], rng)
            res.offer(
                RaidSample(
                    id=row.id,
                    source_id=row.source_id,
                    model=row.model,
                    attack=row.attack,
                    domain=row.domain,
                    title=row.title,
                    text=row.generation.strip(),
                )
            )

    subset = RaidSubset()
    for stratum, target in (("human", n_human), ("ai", n_ai), ("humanized", n_humanized)):
        pools = {k[1]: v.items for k, v in reservoirs.items() if k[0] == stratum}
        bucket = _balanced_take(pools, target, rng)
        getattr(subset, stratum).extend(bucket)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(subset, fh)
        print(f"[raid] cached subset -> {cache_path}")
    return subset


def _balanced_take(pools: dict[str, list], total: int, rng: random.Random) -> list:
    """Round-robin across domains so no domain dominates the final sample."""
    out: list = []
    pools = {k: list(v) for k, v in pools.items() if v}
    for v in pools.values():
        rng.shuffle(v)
    while pools and len(out) < total:
        for key in list(pools):
            if not pools[key]:
                del pools[key]
                continue
            out.append(pools[key].pop())
            if len(out) >= total:
                break
    return out


def group_by_source(samples: list[RaidSample]) -> dict[str, list[RaidSample]]:
    grouped: dict[str, list[RaidSample]] = defaultdict(list)
    for s in samples:
        grouped[s.source_id].append(s)
    return dict(grouped)


def iter_jsonl_texts(path: str | Path, field: str = "text") -> Iterator[str]:
    import json

    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)[field]
