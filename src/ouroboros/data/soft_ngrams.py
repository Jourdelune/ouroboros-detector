"""Soft N-Grams labeling -- Algorithm 1 of the Pangram 4 technical report.

Given a human source document ``S`` and its AI-edited target ``T``, assign every
clause of ``T`` one of {human, ai-assisted, ai-generated} by asking, in order:

1. is there an exact or near-exact *lexical* match in S?  -> human
2. is there a *semantic* match in S?                      -> ai-assisted
3. otherwise the clause is open generation                -> ai-generated

The labeler is deliberately invariant to deletions from S and to clause
rearrangement between S and T (report Section 3.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from rapidfuzz import fuzz

from ..labels import Provenance, weighted_ai_fraction
from .clauses import Clause, ClauseSplitter, heuristic_clauses


@dataclass
class SoftNGramConfig:
    #: L(x, y) above this means the clause survived the edit verbatim.
    lexical_human: float = 0.88
    #: L(x, y) above this (but below ``lexical_human``) is a substantial rewrite.
    lexical_assisted: float = 0.45
    #: E(x, y) above this counts as a semantic match to an idea present in S.
    semantic_assisted: float = 0.62
    #: Merge up to this many adjacent source clauses when searching for a match.
    max_merge: int = 2
    char_ngram: int = 3


def _char_ngrams(text: str, n: int) -> set[str]:
    text = " ".join(text.lower().split())
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_similarity(x: str, y: str, n: int = 3) -> float:
    """L(x, y): token n-gram overlap combined with character n-gram overlap."""
    token_score = fuzz.token_set_ratio(x, y) / 100.0
    char_score = _jaccard(_char_ngrams(x, n), _char_ngrams(y, n))
    seq_score = fuzz.ratio(x, y) / 100.0
    return max(seq_score, 0.5 * token_score + 0.5 * char_score)


class Embedder:
    """Lazy sentence-transformers wrapper used for E(x, y)."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.name = name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=256,
        )
        return np.asarray(vecs, dtype=np.float32)


class NullEmbedder:
    """Drop-in replacement that disables the semantic branch of Algorithm 1."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 1), dtype=np.float32)


@dataclass
class LabeledClause:
    start: int
    end: int
    label: int
    lexical: float
    semantic: float


def label_edit(
    source: str,
    target: str,
    embedder: Embedder | NullEmbedder | None = None,
    cfg: SoftNGramConfig | None = None,
    splitter: ClauseSplitter | Callable[[str], list[Clause]] = heuristic_clauses,
) -> tuple[list[LabeledClause], float]:
    """Run Algorithm 1 on one (source, edited-target) pair.

    Returns the clause labels for ``target`` and the weighted AI fraction f_AI.
    """
    cfg = cfg or SoftNGramConfig()
    embedder = embedder if embedder is not None else NullEmbedder()

    src_clauses = splitter(source)
    tgt_clauses = splitter(target)
    if not tgt_clauses:
        return [], 0.0

    # Candidate spans in S: single clauses plus merged runs, which lets a target
    # clause match a source idea that the editor split or joined.
    candidates: list[str] = []
    for width in range(1, cfg.max_merge + 1):
        for i in range(len(src_clauses) - width + 1):
            candidates.append("".join(c.text for c in src_clauses[i : i + width]))
    if not candidates:
        candidates = [source]

    src_vecs = embedder.encode(candidates)
    tgt_vecs = embedder.encode([c.text for c in tgt_clauses])
    use_semantic = src_vecs.shape[1] > 1 and tgt_vecs.shape[1] > 1
    sem_matrix = tgt_vecs @ src_vecs.T if use_semantic else None

    labeled: list[LabeledClause] = []
    for idx, clause in enumerate(tgt_clauses):
        text = clause.text.strip()
        if not text:
            # Pure whitespace inherits the previous clause's label.
            label = labeled[-1].label if labeled else int(Provenance.HUMAN)
            labeled.append(LabeledClause(clause.start, clause.end, label, 1.0, 1.0))
            continue

        lex = max(lexical_similarity(cand, clause.text, cfg.char_ngram) for cand in candidates)
        sem = float(sem_matrix[idx].max()) if sem_matrix is not None else 0.0

        if lex >= cfg.lexical_human:
            label = int(Provenance.HUMAN)
        elif lex >= cfg.lexical_assisted or sem >= cfg.semantic_assisted:
            label = int(Provenance.ASSISTED)
        else:
            label = int(Provenance.AI)
        labeled.append(LabeledClause(clause.start, clause.end, label, lex, sem))

    counts = [0.0, 0.0, 0.0]
    for lc in labeled:
        counts[lc.label] += lc.end - lc.start
    f_ai = weighted_ai_fraction(*counts)
    return labeled, f_ai


def _prepare(source: str, target: str, cfg: SoftNGramConfig, splitter) -> tuple[list[str], list[Clause]]:
    """Clause-split both sides and build the candidate spans searched in S."""
    src_clauses = splitter(source)
    tgt_clauses = splitter(target)
    candidates: list[str] = []
    for width in range(1, cfg.max_merge + 1):
        for i in range(len(src_clauses) - width + 1):
            candidates.append("".join(c.text for c in src_clauses[i : i + width]))
    if not candidates:
        candidates = [source]
    return candidates, tgt_clauses


def lexical_max(
    candidates: list[str], targets: list[str], n: int = 3
) -> np.ndarray:
    """Per-target best lexical similarity against every candidate.

    The pairwise ratios go through ``rapidfuzz.process.cdist``, which runs in C
    across threads, and the character n-gram sets are built once per string
    instead of once per (candidate, target) pair. The naive double loop was
    rebuilding the same sets thousands of times per document.
    """
    from rapidfuzz import process

    if not candidates or not targets:
        return np.zeros(len(targets))

    seq = process.cdist(targets, candidates, scorer=fuzz.ratio, workers=-1) / 100.0
    token = process.cdist(targets, candidates, scorer=fuzz.token_set_ratio, workers=-1) / 100.0

    cand_grams = [_char_ngrams(c, n) for c in candidates]
    char = np.zeros_like(seq)
    for i, target in enumerate(targets):
        grams = _char_ngrams(target, n)
        if not grams:
            continue
        for j, other in enumerate(cand_grams):
            if other:
                char[i, j] = len(grams & other) / len(grams | other)

    combined = np.maximum(seq, 0.5 * token + 0.5 * char)
    return combined.max(axis=1)


def _decide(
    candidates: list[str],
    tgt_clauses: list[Clause],
    sem_row_max: np.ndarray | None,
    cfg: SoftNGramConfig,
) -> tuple[list[LabeledClause], float]:
    lex_max = lexical_max(candidates, [c.text for c in tgt_clauses], cfg.char_ngram)
    labeled: list[LabeledClause] = []
    for idx, clause in enumerate(tgt_clauses):
        if not clause.text.strip():
            label = labeled[-1].label if labeled else int(Provenance.HUMAN)
            labeled.append(LabeledClause(clause.start, clause.end, label, 1.0, 1.0))
            continue
        lex = float(lex_max[idx])
        sem = float(sem_row_max[idx]) if sem_row_max is not None else 0.0
        if lex >= cfg.lexical_human:
            label = int(Provenance.HUMAN)
        elif lex >= cfg.lexical_assisted or sem >= cfg.semantic_assisted:
            label = int(Provenance.ASSISTED)
        else:
            label = int(Provenance.AI)
        labeled.append(LabeledClause(clause.start, clause.end, label, lex, sem))

    counts = [0.0, 0.0, 0.0]
    for lc in labeled:
        counts[lc.label] += lc.end - lc.start
    return labeled, weighted_ai_fraction(*counts)


def label_edits(
    pairs: Sequence[tuple[str, str]],
    embedder: Embedder | NullEmbedder | None = None,
    cfg: SoftNGramConfig | None = None,
    splitter: ClauseSplitter | Callable[[str], list[Clause]] = heuristic_clauses,
    batch_pairs: int = 96,
    progress: bool = True,
) -> list[tuple[list[LabeledClause], float]]:
    """Batched Algorithm 1 over many (source, target) pairs.

    Identical results to calling :func:`label_edit` per pair, but every pair in a
    batch shares one embedding call. Encoding one pair at a time leaves the GPU
    almost idle -- it was 73% of the runtime -- because each call is a handful of
    short strings.
    """
    from tqdm.auto import tqdm

    cfg = cfg or SoftNGramConfig()
    embedder = embedder if embedder is not None else NullEmbedder()
    results: list[tuple[list[LabeledClause], float]] = []

    batches = range(0, len(pairs), batch_pairs)
    for start in tqdm(batches, desc="soft-ngrams", disable=not progress, unit="batch"):
        chunk = pairs[start : start + batch_pairs]
        prepared = [_prepare(src, tgt, cfg, splitter) for src, tgt in chunk]

        # One encode call for every candidate and every target clause in the batch.
        flat: list[str] = []
        spans: list[tuple[int, int, int, int]] = []
        for candidates, tgt_clauses in prepared:
            c0 = len(flat)
            flat.extend(candidates)
            t0 = len(flat)
            flat.extend(c.text for c in tgt_clauses)
            spans.append((c0, t0, t0, len(flat)))

        vectors = embedder.encode(flat)
        use_semantic = vectors.ndim == 2 and vectors.shape[1] > 1

        for (candidates, tgt_clauses), (c0, c1, t0, t1) in zip(prepared, spans):
            if not tgt_clauses:
                results.append(([], 0.0))
                continue
            sem_row_max = None
            if use_semantic and c1 > c0 and t1 > t0:
                sem_row_max = (vectors[t0:t1] @ vectors[c0:c1].T).max(axis=1)
            results.append(_decide(candidates, tgt_clauses, sem_row_max, cfg))
    return results
