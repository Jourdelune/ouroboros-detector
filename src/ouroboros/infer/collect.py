"""Run sliding-window inference over a corpus and keep the per-token features.

Shared by calibration fitting and evaluation so both see identical observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from ..data.dataset import char_spans_to_token_labels
from ..data.documents import Document
from .predict import Predictor
from .windows import AggregatedObservations


@dataclass
class Observed:
    doc: Document
    agg: AggregatedObservations
    true_token_labels: np.ndarray


def collect(
    predictor: Predictor, docs: list[Document], progress: bool = True, limit: int | None = None
) -> list[Observed]:
    docs = docs[:limit] if limit else docs
    out: list[Observed] = []
    for doc in tqdm(docs, desc="observe", disable=not progress):
        agg, _ = predictor.observe(doc.text)
        if agg.features.shape[0] == 0:
            continue
        true_labels, _ = char_spans_to_token_labels(agg.offsets, doc)
        out.append(Observed(doc, agg, true_labels.astype(np.int64)))
    return out
