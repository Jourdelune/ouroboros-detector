"""Windowing, Repeat2 construction and per-window target derivation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..config import DataConfig
from ..labels import (
    IGNORE_INDEX,
    N_BUCKETS,
    Provenance,
    soft_bucket_target,
    weighted_ai_fraction,
)
from .documents import Document


@dataclass
class TokenizedDoc:
    input_ids: np.ndarray  # (T,) int32
    token_labels: np.ndarray  # (T,) int8, provenance per token
    char_lens: np.ndarray  # (T,) int32, characters attributed to each token
    humanizer: int


def char_spans_to_token_labels(
    offsets: list[tuple[int, int]], doc: Document
) -> tuple[np.ndarray, np.ndarray]:
    """Map a document's character-span partition onto its tokens.

    A token takes the label of the span containing its character midpoint, the
    same rule the post-processor uses to assign tokens to sentences.
    """
    starts = np.array([s.start for s in doc.spans], dtype=np.int64)
    labels = np.array([s.label for s in doc.spans], dtype=np.int8)
    mids = np.array([(a + b) / 2.0 for a, b in offsets], dtype=np.float64)
    idx = np.clip(np.searchsorted(starts, mids, side="right") - 1, 0, len(starts) - 1)
    char_lens = np.array([b - a for a, b in offsets], dtype=np.int32)
    return labels[idx], char_lens


def tokenize_documents(
    docs: list[Document], tokenizer, show_progress: bool = True, batch_size: int = 256
) -> list[TokenizedDoc]:
    out: list[TokenizedDoc] = []
    iterator = range(0, len(docs), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="tokenize", unit="batch")
    for start in iterator:
        chunk = docs[start : start + batch_size]
        enc = tokenizer(
            [d.text for d in chunk],
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        for doc, ids, offsets in zip(chunk, enc["input_ids"], enc["offset_mapping"]):
            if not ids:
                continue
            token_labels, char_lens = char_spans_to_token_labels(offsets, doc)
            out.append(
                TokenizedDoc(
                    input_ids=np.asarray(ids, dtype=np.int32),
                    token_labels=token_labels,
                    char_lens=char_lens,
                    humanizer=doc.humanizer,
                )
            )
    return out


def window_bounds(n_tokens: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Overlapping windows of at most ``window`` tokens; the last is end-anchored."""
    if n_tokens <= window:
        return [(0, n_tokens)]
    bounds = []
    start = 0
    while start + window < n_tokens:
        bounds.append((start, start + window))
        start += stride
    bounds.append((n_tokens - window, n_tokens))
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for b in bounds:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def window_targets(
    token_labels: np.ndarray, char_lens: np.ndarray, mixed_threshold: float, sharpness: float
) -> tuple[np.ndarray, int, float]:
    """Segment bucket target, mixed-authorship flag and f_AI for one window."""
    counts = [float(char_lens[token_labels == c].sum()) for c in range(3)]
    f_ai = weighted_ai_fraction(*counts)
    target = soft_bucket_target(f_ai, sharpness)

    bincount = np.bincount(token_labels.astype(np.int64), minlength=3)
    dominant = int(bincount.argmax())
    off_dominant = float(bincount.sum() - bincount[dominant]) / max(1, int(bincount.sum()))
    mixed = int(off_dominant > mixed_threshold)
    return target, mixed, f_ai


class WindowDataset(Dataset):
    """Training windows drawn from tokenized documents.

    Stage 1 feeds single-copy windows. Stage 2 applies the Repeat2 construction
    of report Section 4.1: the window is concatenated with itself and only the
    second copy is supervised, so every supervised token can attend to the whole
    window despite the causal mask.
    """

    def __init__(
        self,
        docs: list[TokenizedDoc],
        cfg: DataConfig,
        stage: int,
        seed: int | None = None,
    ):
        self.docs = docs
        self.cfg = cfg
        self.stage = stage
        rng = random.Random(cfg.seed if seed is None else seed)

        self.index: list[tuple[int, int, int]] = []
        for doc_idx, doc in enumerate(docs):
            bounds = window_bounds(len(doc.input_ids), cfg.window_tokens, cfg.stride_tokens)
            if cfg.max_windows_per_doc is not None and len(bounds) > cfg.max_windows_per_doc:
                bounds = rng.sample(bounds, cfg.max_windows_per_doc)
            for lo, hi in bounds:
                self.index.append((doc_idx, lo, hi))
        rng.shuffle(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        doc_idx, lo, hi = self.index[i]
        doc = self.docs[doc_idx]
        ids = doc.input_ids[lo:hi].astype(np.int64)
        labels = doc.token_labels[lo:hi].astype(np.int64)
        char_lens = doc.char_lens[lo:hi]

        target, mixed, f_ai = window_targets(
            labels, char_lens, self.cfg.mixed_threshold, self.cfg.soft_bucket_sharpness
        )
        n = len(ids)

        if self.stage >= 2:
            input_ids = np.concatenate([ids, ids])
            token_labels = labels
            supervised_start = n
            last_index = 2 * n - 1
        else:
            input_ids = ids
            token_labels = np.full(n, IGNORE_INDEX, dtype=np.int64)
            supervised_start = 0
            last_index = n - 1

        return {
            "input_ids": torch.from_numpy(input_ids),
            "token_labels": torch.from_numpy(token_labels),
            "segment_target": torch.from_numpy(target),
            "segment_mask": torch.tensor(1, dtype=torch.long),
            "mixed_label": torch.tensor(mixed if self.stage >= 2 else IGNORE_INDEX),
            "humanizer_label": torch.tensor(doc.humanizer),
            "last_index": torch.tensor(last_index),
            "supervised_start": torch.tensor(supervised_start),
            "f_ai": torch.tensor(f_ai, dtype=torch.float32),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; ``last_index`` already points at the true final token."""
    max_len = max(len(b["input_ids"]) for b in batch)
    max_sup = max(len(b["token_labels"]) for b in batch)
    size = len(batch)

    input_ids = torch.full((size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((size, max_len), dtype=torch.long)
    token_labels = torch.full((size, max_sup), IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = 1
        m = len(item["token_labels"])
        token_labels[i, :m] = item["token_labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_labels": token_labels,
        "segment_target": torch.stack([b["segment_target"] for b in batch]),
        "segment_mask": torch.stack([b["segment_mask"] for b in batch]),
        "mixed_label": torch.stack([b["mixed_label"] for b in batch]),
        "humanizer_label": torch.stack([b["humanizer_label"] for b in batch]),
        "last_index": torch.stack([b["last_index"] for b in batch]),
        "supervised_start": torch.stack([b["supervised_start"] for b in batch]),
        "f_ai": torch.stack([b["f_ai"] for b in batch]),
    }
