"""Evaluation metrics following report Section 5.1.

A False Positive is a prediction of Mixed *or* AI when the truth is Human; a
False Negative is a prediction of Human *or* Mixed when the truth is AI. A
Mixed prediction therefore counts against both rates, which is stricter than
thresholding a binary logit.
"""

from __future__ import annotations

import numpy as np

from ..config import PostprocessConfig
from ..data.documents import Document
from ..labels import PROVENANCE_NAMES


def document_truth(doc: Document, cfg: PostprocessConfig) -> str:
    """Ground-truth document category under the Section 5.1 decision rule."""
    counts = doc.char_counts()
    total = max(1.0, sum(counts))
    fractions = [c / total for c in counts]
    if fractions[0] >= cfg.human_fraction_threshold:
        return "human"
    if fractions[2] >= cfg.ai_fraction_threshold:
        return "ai"
    return "mixed"


def binary_rates(truths: list[str], predictions: list[str]) -> dict[str, float]:
    fp = sum(1 for t, p in zip(truths, predictions) if t == "human" and p != "human")
    n_human = sum(1 for t in truths if t == "human")
    fn = sum(1 for t, p in zip(truths, predictions) if t == "ai" and p != "ai")
    n_ai = sum(1 for t in truths if t == "ai")
    mixed_true = [(t, p) for t, p in zip(truths, predictions) if t == "mixed"]
    mixed_acc = sum(1 for t, p in mixed_true if p == "mixed") / max(1, len(mixed_true))
    return {
        "fpr": fp / max(1, n_human),
        "fnr": fn / max(1, n_ai),
        "n_human": n_human,
        "n_ai": n_ai,
        "n_mixed": len(mixed_true),
        "mixed_accuracy": mixed_acc,
    }


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Rank-based AUROC; ``positives`` is a boolean mask of the AI class."""
    pos, neg = scores[positives], scores[~positives]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks over ties so identical scores do not inflate the score.
    combined = np.concatenate([pos, neg])
    for value in np.unique(combined):
        tie = combined == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(scores: np.ndarray, positives: np.ndarray, target_fpr: float) -> float:
    """Highest TPR achievable while keeping the FPR at or below the target."""
    neg = np.sort(scores[~positives])
    pos = scores[positives]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    k = int(np.floor(target_fpr * len(neg)))
    threshold = neg[-1] + 1e-9 if k == 0 else neg[len(neg) - k]
    return float((pos >= threshold).mean())


def token_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1 over the three provenance classes."""
    out: dict[str, float] = {}
    if len(true_labels) == 0:
        return {"token_accuracy": float("nan"), "token_macro_f1": float("nan")}
    out["token_accuracy"] = float((true_labels == pred_labels).mean())
    f1s = []
    for c in range(3):
        tp = float(((pred_labels == c) & (true_labels == c)).sum())
        fp = float(((pred_labels == c) & (true_labels != c)).sum())
        fn = float(((pred_labels != c) & (true_labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
        f1s.append(f1)
        out[f"token_f1_{PROVENANCE_NAMES[c]}"] = f1
    out["token_macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def boundary_metrics(
    true_labels: np.ndarray, pred_labels: np.ndarray, tolerance: int = 8
) -> dict[str, float]:
    """Precision/recall of authorship change points within a token tolerance."""
    true_bounds = np.flatnonzero(np.diff(true_labels) != 0)
    pred_bounds = np.flatnonzero(np.diff(pred_labels) != 0)
    if len(true_bounds) == 0 and len(pred_bounds) == 0:
        return {"boundary_f1": 1.0, "boundary_precision": 1.0, "boundary_recall": 1.0}
    matched_pred = sum(
        1 for b in pred_bounds if len(true_bounds) and np.min(np.abs(true_bounds - b)) <= tolerance
    )
    matched_true = sum(
        1 for b in true_bounds if len(pred_bounds) and np.min(np.abs(pred_bounds - b)) <= tolerance
    )
    precision = matched_pred / max(1, len(pred_bounds))
    recall = matched_true / max(1, len(true_bounds))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"boundary_f1": f1, "boundary_precision": precision, "boundary_recall": recall}
