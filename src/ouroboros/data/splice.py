"""Heterogeneous mixed-text construction (report Section 3.2, Figure 3).

In heterogeneous mixed text every token has a well-defined author, so splicing
real human and real AI passages at sentence boundaries yields exact ground-truth
provenance spans at no annotation cost. This complements the homogeneous
(AI-edited) examples labeled by :mod:`ouroboros.data.soft_ngrams`.
"""

from __future__ import annotations

import random

from ..labels import Humanizer, Provenance
from .clauses import split_sentences
from .documents import Document, Span

PATTERNS = ("human_then_ai", "ai_then_human", "ai_insert", "alternating")


def _take_sentences(text: str, lo: float, hi: float, rng: random.Random) -> str:
    """Return a contiguous run of whole sentences covering a random fraction."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text
    frac = rng.uniform(lo, hi)
    keep = max(1, int(round(len(sentences) * frac)))
    start = rng.randrange(0, max(1, len(sentences) - keep + 1))
    chosen = sentences[start : start + keep]
    return text[chosen[0].start : chosen[-1].end]


def _join(pieces: list[tuple[str, int]]) -> Document | None:
    """Concatenate ``(text, label)`` pieces into a span-annotated document."""
    parts: list[str] = []
    spans: list[Span] = []
    cursor = 0
    for text, label in pieces:
        text = text.strip()
        if not text:
            continue
        chunk = text if not parts else " " + text
        parts.append(chunk)
        spans.append(Span(cursor, cursor + len(chunk), label))
        cursor += len(chunk)
    if len(spans) < 2:
        return None
    return Document(id="", text="".join(parts), spans=spans, humanizer=int(Humanizer.MIXED_AUTHORSHIP))


def splice(
    human_text: str,
    ai_text: str,
    rng: random.Random,
    pattern: str | None = None,
    doc_id: str = "splice",
    source: str = "splice",
) -> Document | None:
    """Build one heterogeneous mixed document from a human and an AI passage."""
    pattern = pattern or rng.choice(PATTERNS)
    H, A = int(Provenance.HUMAN), int(Provenance.AI)

    if pattern == "human_then_ai":
        pieces = [
            (_take_sentences(human_text, 0.3, 0.8, rng), H),
            (_take_sentences(ai_text, 0.2, 0.6, rng), A),
        ]
    elif pattern == "ai_then_human":
        pieces = [
            (_take_sentences(ai_text, 0.3, 0.8, rng), A),
            (_take_sentences(human_text, 0.2, 0.6, rng), H),
        ]
    elif pattern == "ai_insert":
        sentences = split_sentences(human_text)
        if len(sentences) < 3:
            return None
        cut = rng.randrange(1, len(sentences) - 1)
        pieces = [
            (human_text[: sentences[cut].start], H),
            (_take_sentences(ai_text, 0.2, 0.5, rng), A),
            (human_text[sentences[cut].start :], H),
        ]
    else:  # alternating
        human_parts = split_sentences(human_text)
        ai_parts = split_sentences(ai_text)
        if len(human_parts) < 2 or len(ai_parts) < 2:
            return None
        pieces = []
        hi = ai_i = 0
        turn_human = rng.random() < 0.5
        while (hi < len(human_parts) or ai_i < len(ai_parts)) and len(pieces) < 6:
            step = rng.randint(1, 2)
            if turn_human and hi < len(human_parts):
                run = human_parts[hi : hi + step]
                pieces.append((human_text[run[0].start : run[-1].end], H))
                hi += step
            elif ai_i < len(ai_parts):
                run = ai_parts[ai_i : ai_i + step]
                pieces.append((ai_text[run[0].start : run[-1].end], A))
                ai_i += step
            turn_human = not turn_human

    doc = _join(pieces)
    if doc is None:
        return None
    doc.id = doc_id
    doc.source = source
    doc.meta = {"pattern": pattern}
    doc.validate()
    return doc
