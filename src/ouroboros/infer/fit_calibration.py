"""Fit the global calibrator on a held-out split (report Section 4.3.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.documents import Document
from ..eval.metrics import document_truth
from .calibration import Calibrator, fit_multinomial, grid_search_crf
from .collect import collect
from .postprocess import postprocess
from .predict import Predictor


def fit_calibrator(
    predictor: Predictor,
    docs: list[Document],
    max_tokens: int = 400_000,
    lambdas: list[float] | None = None,
    gammas: list[float] | None = None,
    target_fpr: float | None = 0.005,
    seed: int = 0,
) -> tuple[Calibrator, dict]:
    """Fit unary weights, then sweep the CRF penalties and the human prior."""
    observations = collect(predictor, docs)
    if not observations:
        raise ValueError("no calibration observations")

    features = np.concatenate([o.agg.features for o in observations])
    labels = np.concatenate([o.true_token_labels for o in observations])

    rng = np.random.default_rng(seed)
    if len(labels) > max_tokens:
        keep = rng.choice(len(labels), max_tokens, replace=False)
        features, labels = features[keep], labels[keep]

    intercepts, weights = fit_multinomial(features, labels)
    calibrator = Calibrator(
        intercepts=intercepts,
        weights=weights,
        deltas=[0.0, 0.0, 0.0],
        smoothness_lambda=predictor.cfg.postprocess.smoothness_lambda,
        mixed_gamma=predictor.cfg.postprocess.mixed_gamma,
    )

    grid = [(o.agg.features, o.agg.mixed_logodds, o.true_token_labels) for o in observations]
    lam, gamma, score = grid_search_crf(
        grid,
        calibrator,
        lambdas or [0.0, 1.0, 2.0, 4.0, 6.0, 9.0, 13.0],
        gammas or [0.0, 0.25, 0.5, 1.0],
    )
    calibrator.smoothness_lambda, calibrator.mixed_gamma = lam, gamma

    info = {"crf_token_macro_f1": score, "lambda": lam, "gamma": gamma, "n_tokens": int(len(labels))}

    if target_fpr is not None:
        shift, achieved, curve = _tune_human_prior(
            predictor, observations, calibrator, target_fpr
        )
        calibrator.deltas = [shift, 0.0, 0.0]
        info["human_prior_delta"] = shift
        info["calibration_fpr"] = achieved
        info["operating_curve"] = curve
    return calibrator, info


def _tune_human_prior(predictor, observations, calibrator: Calibrator, target_fpr: float):
    """Choose the human-class prior shift under an explicit three-way trade-off.

    Sweeping delta upward and stopping at the first value that meets an FPR
    target is a trap: the smallest delta that reaches an aggressive target sits
    right at the cliff where every document collapses to `human`, which silently
    destroys mixed-document accuracy and boundary detection. So we score the
    whole sweep on all three document classes and pick the best balanced point
    that still satisfies the constraint.
    """
    cfg = predictor.cfg.postprocess
    groups = {"human": [], "ai": [], "mixed": []}
    for o in observations:
        groups[document_truth(o.doc, cfg)].append(o)
    if not groups["human"]:
        return 0.0, float("nan"), []

    curve = []
    for shift in np.linspace(0.0, 8.0, 33):
        calibrator.deltas = [float(shift), 0.0, 0.0]
        rates = {}
        for name, obs in groups.items():
            if not obs:
                rates[name] = float("nan")
                continue
            hits = sum(
                1 for o in obs if postprocess(o.agg, calibrator, cfg).document_label == name
            )
            rates[name] = hits / len(obs)
        fpr = 1.0 - rates["human"]
        # Balanced objective over the two classes the FPR sweep tends to wreck.
        balance = np.nanmean([rates["ai"], rates["mixed"]])
        curve.append(
            {
                "delta": float(shift),
                "fpr": fpr,
                "ai_recall": rates["ai"],
                "mixed_recall": rates["mixed"],
                "balance": float(balance),
            }
        )

    feasible = [c for c in curve if c["fpr"] <= target_fpr]
    if feasible:
        best = max(feasible, key=lambda c: c["balance"])
    else:
        # No operating point reaches the target: take the lowest FPR instead of
        # pretending, and let the caller see it in the report.
        best = min(curve, key=lambda c: (c["fpr"], -c["balance"]))
    calibrator.deltas = [best["delta"], 0.0, 0.0]
    return best["delta"], best["fpr"], curve
