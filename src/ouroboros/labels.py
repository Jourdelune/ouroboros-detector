"""Label space and the weighted-AI-fraction bucketing used throughout Ouroboros.

Mirrors Section 3.3 (three-way token provenance) and Section 4.1 (15 ordered
buckets over the weighted AI fraction) of the Pangram 4 technical report.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

N_BUCKETS = 15
IGNORE_INDEX = -100


class Provenance(IntEnum):
    """Per-token authorship classes (report Section 3.3)."""

    HUMAN = 0
    ASSISTED = 1
    AI = 2


class Humanizer(IntEnum):
    """Document-level classes for the auxiliary humanization task (Section 3.4)."""

    HUMAN = 0
    AI_GENERATED = 1
    HUMANIZED_AI = 2
    MIXED_AUTHORSHIP = 3


PROVENANCE_NAMES = ["human", "ai-assisted", "ai-generated"]
HUMANIZER_NAMES = ["human", "ai-generated", "humanized-ai", "mixed-authorship"]

#: Bucket centres f_b, evenly spaced over [0, 1].
BUCKET_CENTERS = np.linspace(0.0, 1.0, N_BUCKETS)


def weighted_ai_fraction(c_human: float, c_assisted: float, c_ai: float) -> float:
    """f_AI = (0.5 * C_AA + C_AG) / (C_H + C_AA + C_AG), Section 3.5 / 4.1."""
    denom = c_human + c_assisted + c_ai
    if denom <= 0:
        return 0.0
    return (0.5 * c_assisted + c_ai) / denom


def fraction_to_bucket(f: float) -> int:
    """Nearest of the 15 ordered buckets for a weighted AI fraction."""
    f = min(1.0, max(0.0, float(f)))
    return int(round(f * (N_BUCKETS - 1)))


def soft_bucket_target(f: float, sharpness: float = 1.0) -> np.ndarray:
    """Linearly interpolated bucket target.

    The report supervises f_AI as a *bucketed regression* target rather than a
    plain one-hot classification, so mass is split between the two buckets that
    bracket f. ``sharpness`` > 1 concentrates the target towards the nearest
    bucket; ``sharpness`` -> inf recovers hard one-hot targets.
    """
    f = min(1.0, max(0.0, float(f)))
    pos = f * (N_BUCKETS - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, N_BUCKETS - 1)
    frac = pos - lo
    target = np.zeros(N_BUCKETS, dtype=np.float32)
    if lo == hi:
        target[lo] = 1.0
        return target
    w_hi = frac**sharpness
    w_lo = (1.0 - frac) ** sharpness
    total = w_lo + w_hi
    target[lo] = w_lo / total
    target[hi] = w_hi / total
    return target


def triangular_basis(f: np.ndarray | float) -> np.ndarray:
    """Equations (1)-(3): project a weighted AI fraction onto three class anchors."""
    f = np.asarray(f, dtype=np.float64)
    phi_human = np.maximum(0.0, 1.0 - 2.0 * f)
    phi_assisted = np.maximum(0.0, 1.0 - 2.0 * np.abs(f - 0.5))
    phi_ai = np.maximum(0.0, 2.0 * f - 1.0)
    return np.stack([phi_human, phi_assisted, phi_ai], axis=-1)


#: Precomputed basis matrix B[b, c] = phi_c(f_b) for the 15 bucket centres.
BUCKET_BASIS = triangular_basis(BUCKET_CENTERS)


def segment_prior(bucket_probs: np.ndarray) -> np.ndarray:
    """Ternary document prior pi_w(c) from a 15-bucket distribution (Section 4.3.3)."""
    weighted = np.asarray(bucket_probs, dtype=np.float64) @ BUCKET_BASIS
    total = weighted.sum(axis=-1, keepdims=True)
    return weighted / np.maximum(total, 1e-12)
