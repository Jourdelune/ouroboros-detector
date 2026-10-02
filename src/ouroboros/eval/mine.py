"""Offline active learning: mine the cases the current checkpoint gets wrong.

Report Section 4.2 describes one round of this for *hard negatives* -- human
text the model calls AI -- which is how a false-positive rate reaches the
1-in-24,000 range. We mine symmetrically and also collect *hard positives*: AI
text the model calls human. Those are what a miss on an unseen register looks
like, and they are invisible to aggregate metrics when the register is rare.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..config import PostprocessConfig
from ..data.documents import Document, write_jsonl
from ..eval.metrics import document_truth
from ..infer.postprocess import postprocess
from ..infer.predict import Predictor


@dataclass
class MineStats:
    seen: int = 0
    hard_negatives: int = 0
    hard_positives: int = 0
    hard_mixed: int = 0

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "hard_negatives": self.hard_negatives,
            "hard_positives": self.hard_positives,
            "hard_mixed": self.hard_mixed,
            "error_rate": round(
                (self.hard_negatives + self.hard_positives + self.hard_mixed)
                / max(1, self.seen),
                5,
            ),
        }


def mine_hard_cases(
    predictor: Predictor,
    docs: list[Document],
    cfg: PostprocessConfig | None = None,
    margin: float = 0.0,
    include_mixed: bool = True,
    progress: bool = True,
) -> tuple[list[Document], MineStats]:
    """Return the documents the model currently gets wrong, with a margin.

    ``margin`` widens the net beyond outright errors: a human document scored
    at f_AI just under the threshold is nearly a false positive and is worth
    training on too.
    """
    cfg = cfg or predictor.cfg.postprocess
    stats = MineStats()
    hard: list[Document] = []

    for doc in tqdm(docs, desc="mine", unit="doc", disable=not progress):
        agg, _ = predictor.observe(doc.text)
        if agg.features.shape[0] == 0:
            continue
        pred = postprocess(agg, predictor.calibrator, cfg)
        truth = document_truth(doc, cfg)
        stats.seen += 1

        f_ai = pred.weighted_ai_fraction
        wrong = pred.document_label != truth
        near_miss = False
        if truth == "human":
            near_miss = f_ai > margin
        elif truth == "ai":
            near_miss = f_ai < 1.0 - margin

        if not (wrong or near_miss):
            continue
        if truth == "human":
            stats.hard_negatives += 1
        elif truth == "ai":
            stats.hard_positives += 1
        else:
            if not include_mixed:
                continue
            stats.hard_mixed += 1

        doc.meta = {
            **doc.meta,
            "mined": True,
            "truth": truth,
            "predicted": pred.document_label,
            "f_ai": round(float(f_ai), 4),
        }
        hard.append(doc)

    return hard, stats


def save_mined(path: str | Path, docs: list[Document], stats: MineStats) -> None:
    n = write_jsonl(path, docs)
    report = Path(str(path).replace(".jsonl", "_stats.json"))
    report.write_text(json.dumps({**stats.as_dict(), "written": n}, indent=2))
