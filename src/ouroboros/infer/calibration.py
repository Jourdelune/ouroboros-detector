"""Global calibration (report Section 4.3.3).

A multinomial logistic model maps the six-dimensional per-token feature vector

    x_i = [l_t(human), l_t(assisted), l_t(AI), l_s(human), l_s(assisted), l_s(AI)]

to the CRF unary potential  u_i(c) = alpha_c + w_c . x_i + delta_c.

The CRF's smoothness penalty lambda and mixed-relaxation gamma are then chosen
on the same held-out calibration split by grid search, and the class-prior
adjustments delta_c are swept to hit a target false-positive rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Calibrator:
    """Fitted unary-potential model plus the CRF's decoding hyperparameters."""

    intercepts: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    weights: list[list[float]] = field(default_factory=lambda: [[0.0] * 6 for _ in range(3)])
    deltas: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    smoothness_lambda: float = 4.0
    mixed_gamma: float = 0.5
    identity: bool = False

    def unary(self, features: np.ndarray) -> np.ndarray:
        """``(N, 6)`` features -> ``(N, 3)`` unary potentials."""
        if self.identity:
            # Untrained fallback: use the averaged token logits directly.
            return features[:, :3] + np.asarray(self.deltas)[None, :]
        w = np.asarray(self.weights)  # (3, 6)
        return features @ w.T + np.asarray(self.intercepts)[None, :] + np.asarray(self.deltas)[None, :]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls(**json.loads(Path(path).read_text()))


def fit_multinomial(features: np.ndarray, labels: np.ndarray, C: float = 1.0) -> tuple[list, list]:
    """Fit the multinomial logistic calibration model."""
    from sklearn.linear_model import LogisticRegression

    present = np.unique(labels)
    # scikit-learn >= 1.7 fits multinomial by default for multiclass targets.
    model = LogisticRegression(max_iter=2000, C=C)
    model.fit(features, labels)

    weights = np.zeros((3, features.shape[1]))
    intercepts = np.zeros(3)
    if len(present) == 2:
        # sklearn stores a single row for a binary problem.
        weights[present[1]] = model.coef_[0]
        intercepts[present[1]] = model.intercept_[0]
        intercepts[present[0]] = 0.0
    else:
        for row, cls in enumerate(model.classes_):
            weights[int(cls)] = model.coef_[row]
            intercepts[int(cls)] = model.intercept_[row]
    absent = [c for c in range(3) if c not in present]
    for c in absent:
        intercepts[c] = -20.0  # never predict a class we saw no evidence for
    return intercepts.tolist(), weights.tolist()


def grid_search_crf(
    documents: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    calibrator: Calibrator,
    lambdas: list[float],
    gammas: list[float],
) -> tuple[float, float, float]:
    """Pick (lambda, gamma) maximizing token **macro-F1** over calibration documents.

    Raw token accuracy is the wrong objective here: it is dominated by pure-human
    and pure-AI documents, so it happily selects gamma = 0 -- i.e. it concludes
    the mixed head's modulation of the transition penalty is worthless, which is
    exactly the mechanism that lets the decoder place boundaries. Macro-F1 gives
    the rare `ai-assisted` class and the boundary structure a real vote.

    Each document is ``(features, mixed_logodds, token_labels)``.
    """
    import numpy as np

    from .crf import decode

    best = (calibrator.smoothness_lambda, calibrator.mixed_gamma, -1.0)
    for lam in lambdas:
        for gamma in gammas:
            tp = np.zeros(3)
            fp = np.zeros(3)
            fn = np.zeros(3)
            for features, mixed, labels in documents:
                unary = calibrator.unary(features)
                path, _ = decode(unary, mixed, lam, gamma)
                for c in range(3):
                    tp[c] += int(((path == c) & (labels == c)).sum())
                    fp[c] += int(((path == c) & (labels != c)).sum())
                    fn[c] += int(((path != c) & (labels == c)).sum())
            present = (tp + fn) > 0
            f1 = 2 * tp / np.maximum(1e-9, 2 * tp + fp + fn)
            score = float(f1[present].mean()) if present.any() else 0.0
            if score > best[2]:
                best = (lam, gamma, score)
    return best


def tune_deltas(
    documents: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    calibrator: Calibrator,
    postprocess_fn,
    target_fpr: float = 0.005,
    sweep: np.ndarray | None = None,
) -> float:
    """Sweep a shared human-class prior shift to reach a target false-positive rate.

    Returns the chosen shift; the caller writes it into ``calibrator.deltas``.
    The operating point trades recall for precision exactly as the report's
    "production operating point" does (Section 5.2).
    """
    sweep = np.linspace(-3.0, 6.0, 37) if sweep is None else sweep
    human_docs = [d for d in documents if d[3] == "human"]
    if not human_docs:
        return 0.0
    original = list(calibrator.deltas)
    best_shift = 0.0
    for shift in sweep:
        calibrator.deltas = [original[0] + float(shift), original[1], original[2]]
        false_positives = 0
        for features, mixed, _, _ in human_docs:
            if postprocess_fn(features, mixed) != "human":
                false_positives += 1
        fpr = false_positives / len(human_docs)
        best_shift = float(shift)
        if fpr <= target_fpr:
            break
    calibrator.deltas = original
    return best_shift
