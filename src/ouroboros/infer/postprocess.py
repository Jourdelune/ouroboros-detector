"""Tokenwise post-processing (report Section 4.3).

Pipeline: calibrated unary potentials -> linear-chain CRF (Viterbi labels plus
forward-backward marginals) -> sentence majority voting -> minimum segment
length merging -> character-aligned segments, document fractions and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import PostprocessConfig
from ..data.clauses import split_sentences
from ..labels import PROVENANCE_NAMES, Provenance
from .calibration import Calibrator
from .crf import decode
from .windows import AggregatedObservations


@dataclass
class Segment:
    start: int
    end: int
    label: int
    confidence: float

    @property
    def label_name(self) -> str:
        return PROVENANCE_NAMES[self.label]


@dataclass
class DocumentPrediction:
    text: str
    segments: list[Segment]
    fractions: dict[str, float]
    weighted_ai_fraction: float
    document_label: str
    token_marginals: np.ndarray = field(repr=False, default=None)
    token_labels: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v:.3f}" for k, v in self.fractions.items())
        return f"{self.document_label} (f_AI={self.weighted_ai_fraction:.3f}; {parts})"


def sentence_majority_vote(
    labels: np.ndarray, offsets: list[tuple[int, int]], text: str
) -> np.ndarray:
    """Replace every token label in a sentence by that sentence's majority label.

    Tokens are assigned to sentences by character midpoint; ties are broken
    deterministically towards the lower class index (Section 4.3.3).
    """
    if len(labels) == 0:
        return labels
    sentences = split_sentences(text)
    if not sentences:
        return labels
    bounds = np.array([s.start for s in sentences], dtype=np.float64)
    mids = np.array([(a + b) / 2.0 for a, b in offsets], dtype=np.float64)
    assignment = np.clip(np.searchsorted(bounds, mids, side="right") - 1, 0, len(sentences) - 1)

    out = labels.copy()
    for sentence_idx in range(len(sentences)):
        mask = assignment == sentence_idx
        if not mask.any():
            continue
        counts = np.bincount(labels[mask], minlength=3)
        out[mask] = int(counts.argmax())  # argmax breaks ties towards class 0
    return out


def enforce_min_segments(labels: np.ndarray, min_tokens: int) -> np.ndarray:
    """Repeatedly merge the first run shorter than ``min_tokens`` into a neighbor."""
    if len(labels) == 0 or min_tokens <= 1:
        return labels
    labels = labels.copy()
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        target = next((r for r in runs if r[1] - r[0] < min_tokens), None)
        if target is None:
            return labels
        idx = runs.index(target)
        prev_run = runs[idx - 1] if idx > 0 else None
        next_run = runs[idx + 1] if idx + 1 < len(runs) else None
        if prev_run is None:
            label = next_run[2]
        elif next_run is None:
            label = prev_run[2]
        elif prev_run[2] == next_run[2]:
            label = prev_run[2]
        else:
            prev_len = prev_run[1] - prev_run[0]
            next_len = next_run[1] - next_run[0]
            if prev_len > next_len:
                label = prev_run[2]
            elif next_len > prev_len:
                label = next_run[2]
            else:
                label = min(prev_run[2], next_run[2])  # deterministic tie-break
        labels[target[0] : target[1]] = label


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, int(labels[start])))
            start = i
    return runs


def build_segments(
    labels: np.ndarray,
    marginals: np.ndarray,
    offsets: list[tuple[int, int]],
    text: str,
) -> list[Segment]:
    """Coalesce equal-label runs into a complete partition of the character sequence."""
    if len(labels) == 0:
        return [Segment(0, len(text), int(Provenance.HUMAN), 0.0)]
    confidences = marginals.max(axis=1)
    segments: list[Segment] = []
    for lo, hi, label in _runs(labels):
        start = offsets[lo][0] if not segments else segments[-1].end
        end = offsets[hi - 1][1]
        segments.append(Segment(start, end, label, float(confidences[lo:hi].mean())))
    segments[0] = Segment(0, segments[0].end, segments[0].label, segments[0].confidence)
    segments[-1] = Segment(segments[-1].start, len(text), segments[-1].label, segments[-1].confidence)
    return segments


def postprocess(
    agg: AggregatedObservations,
    calibrator: Calibrator,
    cfg: PostprocessConfig,
) -> DocumentPrediction:
    """Run the full Section 4.3 decoder on one document's aggregated observations."""
    unary = calibrator.unary(agg.features)
    labels, marginals = decode(
        unary, agg.mixed_logodds, calibrator.smoothness_lambda, calibrator.mixed_gamma
    )
    # Marginals are taken before the constrained steps, per Section 4.3.5.
    voted = sentence_majority_vote(labels, agg.offsets, agg.text)
    constrained = enforce_min_segments(voted, cfg.min_segment_tokens)
    segments = build_segments(constrained, marginals, agg.offsets, agg.text)

    total_chars = max(1, len(agg.text))
    counts = [0.0, 0.0, 0.0]
    for segment in segments:
        counts[segment.label] += segment.end - segment.start
    fractions = {PROVENANCE_NAMES[c]: counts[c] / total_chars for c in range(3)}
    f_ai = (0.5 * counts[1] + counts[2]) / total_chars

    if fractions["human"] >= cfg.human_fraction_threshold:
        document_label = "human"
    elif (fractions["ai-generated"] + fractions["ai-assisted"]) >= cfg.ai_fraction_threshold and (
        fractions["ai-generated"] >= cfg.ai_fraction_threshold
    ):
        document_label = "ai"
    else:
        document_label = "mixed"

    return DocumentPrediction(
        text=agg.text,
        segments=segments,
        fractions=fractions,
        weighted_ai_fraction=f_ai,
        document_label=document_label,
        token_marginals=marginals,
        token_labels=constrained,
    )
