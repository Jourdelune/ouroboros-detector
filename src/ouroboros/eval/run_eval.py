"""Evaluation driver: document-level rates, ranking metrics and span quality."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import PostprocessConfig
from ..data.documents import Document
from ..infer.collect import Observed, collect
from ..infer.postprocess import postprocess
from ..infer.predict import Predictor
from ..labels import Humanizer, HUMANIZER_NAMES
from .metrics import (
    auroc,
    binary_rates,
    boundary_metrics,
    document_truth,
    token_metrics,
    tpr_at_fpr,
)


def evaluate_corpus(
    predictor: Predictor,
    docs: list[Document],
    cfg: PostprocessConfig | None = None,
    limit: int | None = None,
    observations: list[Observed] | None = None,
) -> dict:
    cfg = cfg or predictor.cfg.postprocess
    observations = observations if observations is not None else collect(predictor, docs, limit=limit)

    truths: list[str] = []
    predictions: list[str] = []
    scores: list[float] = []
    token_true: list[np.ndarray] = []
    token_pred: list[np.ndarray] = []
    boundary: list[dict] = []
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_generator: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for item in observations:
        pred = postprocess(item.agg, predictor.calibrator, cfg)
        truth = document_truth(item.doc, cfg)
        truths.append(truth)
        predictions.append(pred.document_label)
        scores.append(pred.weighted_ai_fraction)
        by_source[item.doc.source.split("/")[0]].append((truth, pred.document_label))
        generator = item.doc.meta.get("generator")
        if generator and truth == "ai":
            by_generator[generator].append((truth, pred.document_label))

        token_true.append(item.true_token_labels)
        token_pred.append(pred.token_labels)
        if truth == "mixed":
            boundary.append(boundary_metrics(item.true_token_labels, pred.token_labels))

    scores_a = np.array(scores)
    binary_mask = np.array([t in ("human", "ai") for t in truths])
    positives = np.array([t == "ai" for t in truths])[binary_mask]

    report: dict = {"document": binary_rates(truths, predictions)}
    if binary_mask.any() and positives.any() and (~positives).any():
        report["ranking"] = {
            "auroc": auroc(scores_a[binary_mask], positives),
            "tpr@1%fpr": tpr_at_fpr(scores_a[binary_mask], positives, 0.01),
            "tpr@0.1%fpr": tpr_at_fpr(scores_a[binary_mask], positives, 0.001),
        }
    report["token"] = token_metrics(np.concatenate(token_true), np.concatenate(token_pred))
    if boundary:
        report["boundary"] = {
            k: float(np.mean([b[k] for b in boundary])) for k in boundary[0]
        }
    report["by_source"] = {
        source: binary_rates([t for t, _ in pairs], [p for _, p in pairs])
        for source, pairs in sorted(by_source.items())
    }
    # False negative rate per generator, the breakdown of report Table 3.
    report["fnr_by_generator"] = {
        generator: {
            "fnr": binary_rates([t for t, _ in pairs], [p for _, p in pairs])["fnr"],
            "n": len(pairs),
        }
        for generator, pairs in sorted(by_generator.items())
    }
    return report


def evaluate_humanizer(predictor: Predictor, docs: list[Document], limit: int | None = None) -> dict:
    """Confusion of the four-way humanizer probe (report Section 3.4)."""
    docs = [d for d in docs][:limit] if limit else docs
    confusion = np.zeros((4, 4), dtype=np.int64)
    for doc in docs:
        probs = predictor.humanizer(doc.text)
        pred = int(np.argmax([probs[name] for name in HUMANIZER_NAMES]))
        confusion[doc.humanizer, pred] += 1
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    per_class = {
        HUMANIZER_NAMES[c]: float(confusion[c, c] / max(1, confusion[c].sum())) for c in range(4)
    }
    return {"accuracy": accuracy, "recall_per_class": per_class, "confusion": confusion.tolist()}


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))


def evaluate_segment_head(
    predictor: Predictor, docs: list[Document], limit: int | None = None, batch_size: int = 8
) -> dict:
    """Document-level detection using the segment head alone.

    Stage 1 only trains the segment head, so this is the honest way to measure
    what it learned: score each document by the weighted AI fraction the 15-bucket
    head predicts, averaged over its sliding windows, and rank documents by it.
    It is also a useful baseline for stage 2 -- it isolates how much the tokenwise
    head and the CRF decoder actually add.
    """
    import numpy as np
    from tqdm.auto import tqdm

    from ..infer.windows import infer_document
    from ..labels import BUCKET_CENTERS

    docs = docs[:limit] if limit else docs
    scores: list[float] = []
    truths: list[str] = []
    cfg = predictor.cfg.postprocess

    for doc in tqdm(docs, desc="segment-head", unit="doc"):
        obs = infer_document(
            predictor.model, predictor.tokenizer, doc.text, predictor.cfg.data, predictor.device
        )
        if not obs.windows:
            continue
        # Weight each window by its token count: a 40-token tail window should
        # not count as much as a full 512-token one.
        weights = np.array([w.hi - w.lo for w in obs.windows], dtype=np.float64)
        per_window = np.array([w.segment_probs @ BUCKET_CENTERS for w in obs.windows])
        scores.append(float((per_window * weights).sum() / weights.sum()))
        truths.append(document_truth(doc, cfg))

    scores_a = np.array(scores)
    binary = np.array([t in ("human", "ai") for t in truths])
    positives = np.array([t == "ai" for t in truths])[binary]
    report: dict = {"n_docs": len(scores), "n_human": int((~positives).sum()), "n_ai": int(positives.sum())}

    if positives.any() and (~positives).any():
        s, y = scores_a[binary], positives
        report["auroc"] = auroc(s, y)
        report["tpr@10%fpr"] = tpr_at_fpr(s, y, 0.10)
        report["tpr@1%fpr"] = tpr_at_fpr(s, y, 0.01)
        report["tpr@0.1%fpr"] = tpr_at_fpr(s, y, 0.001)
        report["mean_f_ai_human"] = float(s[~y].mean())
        report["mean_f_ai_ai"] = float(s[y].mean())
        # Best single threshold on this split, for an interpretable error pair.
        best = None
        for thr in np.unique(np.round(s, 3)):
            fpr = float((s[~y] >= thr).mean())
            fnr = float((s[y] < thr).mean())
            if best is None or fpr + fnr < best[1] + best[2]:
                best = (float(thr), fpr, fnr)
        report["best_threshold"] = {"threshold": best[0], "fpr": best[1], "fnr": best[2]}

    mixed = scores_a[np.array([t == "mixed" for t in truths])]
    if len(mixed):
        report["mean_f_ai_mixed"] = float(mixed.mean())
    return report
