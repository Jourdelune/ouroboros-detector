"""Linear-chain CRF with Potts transitions (report Section 4.3.3).

Transition potential between adjacent labels a and b at boundary i:

    tau_i(a, b) = 0                          if a == b
                = min(0, -lambda + gamma*m_i) otherwise

where m_i is the center-weighted mixed log-odds at the boundary, lambda is the
smoothness penalty and gamma controls how strongly mixed evidence relaxes it.
"""

from __future__ import annotations

import numpy as np

N_STATES = 3


def potts_transitions(mixed_logodds: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """``(N-1, 3, 3)`` transition potentials, one matrix per boundary."""
    off = np.minimum(0.0, -lam + gamma * np.asarray(mixed_logodds, dtype=np.float64))
    n = off.shape[0]
    trans = np.repeat(off[:, None, None], N_STATES, axis=1).repeat(N_STATES, axis=2)
    diag = np.arange(N_STATES)
    trans[:, diag, diag] = 0.0
    return trans


def viterbi(unary: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Multiclass Viterbi decoding. ``unary`` is ``(N, 3)``, ``trans`` ``(N-1, 3, 3)``."""
    n = unary.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    scores = unary[0].copy()
    backpointers = np.zeros((n - 1, N_STATES), dtype=np.int64)
    for i in range(1, n):
        candidates = scores[:, None] + trans[i - 1]
        backpointers[i - 1] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + unary[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(scores.argmax())
    for i in range(n - 2, -1, -1):
        path[i] = backpointers[i, path[i + 1]]
    return path


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(a - peak), axis=axis))


def forward_backward(unary: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Numerically stable marginals ``P(T_i = c | observations)``, shape ``(N, 3)``."""
    n = unary.shape[0]
    if n == 0:
        return np.zeros((0, N_STATES))

    alpha = np.zeros((n, N_STATES))
    alpha[0] = unary[0]
    for i in range(1, n):
        alpha[i] = _logsumexp(alpha[i - 1][:, None] + trans[i - 1], axis=0) + unary[i]

    beta = np.zeros((n, N_STATES))
    for i in range(n - 2, -1, -1):
        beta[i] = _logsumexp(trans[i] + unary[i + 1][None, :] + beta[i + 1][None, :], axis=1)

    log_marginals = alpha + beta
    log_marginals -= _logsumexp(log_marginals, axis=1)[:, None]
    return np.exp(log_marginals)


def decode(
    unary: np.ndarray, mixed_logodds: np.ndarray, lam: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(viterbi_labels, token_marginals)``."""
    n = unary.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, N_STATES))
    if n == 1:
        probs = np.exp(unary - _logsumexp(unary, axis=1)[:, None])
        return unary.argmax(axis=1), probs
    trans = potts_transitions(mixed_logodds[: n - 1], lam, gamma)
    return viterbi(unary, trans), forward_backward(unary, trans)
