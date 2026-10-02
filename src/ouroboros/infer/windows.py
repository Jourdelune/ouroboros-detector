"""Sliding-window inference and observation aggregation (report Section 4.3.2/4.3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import DataConfig
from ..data.dataset import window_bounds
from ..labels import segment_prior
from ..modeling.model import OuroborosModel


@dataclass
class WindowObservation:
    lo: int
    hi: int
    segment_probs: np.ndarray  # (15,)
    token_logits: np.ndarray  # (hi - lo, 3)
    mixed_prob: float


@dataclass
class DocumentObservations:
    text: str
    offsets: list[tuple[int, int]]
    windows: list[WindowObservation]

    @property
    def n_tokens(self) -> int:
        return len(self.offsets)


@dataclass
class AggregatedObservations:
    """Per-token features fed to calibration and the CRF."""

    features: np.ndarray  # (N, 6) = [token logits x3, segment log-prior x3]
    mixed_logodds: np.ndarray  # (N,)
    offsets: list[tuple[int, int]]
    text: str


@torch.no_grad()
def infer_document(
    model: OuroborosModel,
    tokenizer,
    text: str,
    cfg: DataConfig,
    device: str = "cuda",
    batch_size: int = 4,
) -> DocumentObservations:
    """Run Repeat2 inference over every overlapping window of a document."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if len(ids) == 0:
        return DocumentObservations(text, [], [])

    bounds = window_bounds(len(ids), cfg.window_tokens, cfg.stride_tokens)
    observations: list[WindowObservation] = []
    model.eval()

    for start in range(0, len(bounds), batch_size):
        chunk = bounds[start : start + batch_size]
        lengths = [hi - lo for lo, hi in chunk]
        max_len = max(lengths)
        # Repeat2: every supervised token then sees the complete window.
        input_ids = torch.full((len(chunk), 2 * max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), 2 * max_len), dtype=torch.long)
        last_index = torch.zeros(len(chunk), dtype=torch.long)
        supervised_start = torch.zeros(len(chunk), dtype=torch.long)
        for row, ((lo, hi), n) in enumerate(zip(chunk, lengths)):
            window = torch.from_numpy(ids[lo:hi])
            input_ids[row, :n] = window
            input_ids[row, n : 2 * n] = window
            attention[row, : 2 * n] = 1
            last_index[row] = 2 * n - 1
            supervised_start[row] = n

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                input_ids.to(device),
                attention.to(device),
                last_index.to(device),
                supervised_start.to(device),
            )
        seg = torch.softmax(out.segment_logits.float(), -1).cpu().numpy()
        tok = out.token_logits.float().cpu().numpy()
        mixed = torch.softmax(out.mixed_logits.float(), -1)[:, 1].cpu().numpy()

        for row, ((lo, hi), n) in enumerate(zip(chunk, lengths)):
            observations.append(
                WindowObservation(lo, hi, seg[row], tok[row, :n], float(mixed[row]))
            )

    return DocumentObservations(text, offsets, observations)


def _center_weights(lo: int, hi: int) -> np.ndarray:
    """Triangular weights peaking at the window centre, used for mixed evidence."""
    n = hi - lo
    positions = np.arange(n)
    center = (n - 1) / 2.0
    half = max(center, 1.0)
    return np.maximum(0.05, 1.0 - np.abs(positions - center) / half)


def aggregate(obs: DocumentObservations, eps: float = 1e-6) -> AggregatedObservations:
    """Collapse overlapping window observations into one feature row per token.

    Token and segment logits are averaged across every window containing the
    token; the mixed log-odds are averaged with center weighting so a window
    speaks loudest about the tokens at its middle.
    """
    n = obs.n_tokens
    token_sum = np.zeros((n, 3))
    segment_sum = np.zeros((n, 3))
    counts = np.zeros(n)
    mixed_sum = np.zeros(n)
    mixed_weight = np.zeros(n)

    for window in obs.windows:
        lo, hi = window.lo, window.hi
        token_sum[lo:hi] += window.token_logits
        prior = segment_prior(window.segment_probs)
        segment_sum[lo:hi] += np.log(np.clip(prior, eps, None))[None, :]
        counts[lo:hi] += 1.0

        q = min(max(window.mixed_prob, eps), 1 - eps)
        weights = _center_weights(lo, hi)
        mixed_sum[lo:hi] += weights * np.log(q / (1 - q))
        mixed_weight[lo:hi] += weights

    counts = np.maximum(counts, 1.0)
    features = np.concatenate(
        [token_sum / counts[:, None], segment_sum / counts[:, None]], axis=1
    )
    mixed_logodds = mixed_sum / np.maximum(mixed_weight, eps)
    return AggregatedObservations(features, mixed_logodds, obs.offsets, obs.text)
