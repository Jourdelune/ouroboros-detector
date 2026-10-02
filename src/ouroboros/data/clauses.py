"""Clause segmentation.

Report Section 3.5 defines a clause as "a group of words that contains a subject
and a verb" and splits with Claude Haiku 4.5. We provide a dependency-free
heuristic splitter as the default and an optional LLM-backed splitter that talks
to any OpenAI-compatible endpoint (including a local server).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

# Sentence terminators, allowing a trailing quote or bracket (Section 4.3.3).
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’\)\]]*\s+|\n\s*\n")

# Clause-initial cues: coordinators and subordinators that typically introduce a
# new subject+verb group.
_CLAUSE_CUE = re.compile(
    r"(?:,\s+|;\s+|:\s+|\s+—\s+|\s+-\s+)"
    r"(?=(?:and|but|or|so|yet|because|although|though|while|whereas|which|who|that|"
    r"when|where|after|before|since|unless|if|as|then)\b)",
    re.IGNORECASE,
)
_SEMICOLON = re.compile(r";\s+")


@dataclass(frozen=True)
class Clause:
    start: int
    end: int
    text: str


class ClauseSplitter(Protocol):
    def __call__(self, text: str) -> list[Clause]: ...


def split_sentences(text: str) -> list[Clause]:
    """Split into sentences, keeping character offsets aligned with ``text``."""
    spans: list[Clause] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if end > cursor and text[cursor:end].strip():
            spans.append(Clause(cursor, end, text[cursor:end]))
            cursor = end
    if cursor < len(text) and text[cursor:].strip():
        spans.append(Clause(cursor, len(text), text[cursor:]))
    if not spans and text:
        spans = [Clause(0, len(text), text)]
    return spans


def heuristic_clauses(text: str, min_chars: int = 18) -> list[Clause]:
    """Split a document into clause-sized units without external dependencies."""
    out: list[Clause] = []
    for sentence in split_sentences(text):
        cuts = [0]
        for pattern in (_CLAUSE_CUE, _SEMICOLON):
            for match in pattern.finditer(sentence.text):
                cuts.append(match.end())
        cuts = sorted(set(cuts + [len(sentence.text)]))
        pieces: list[tuple[int, int]] = []
        for lo, hi in zip(cuts, cuts[1:]):
            if hi - lo < min_chars and pieces:
                pieces[-1] = (pieces[-1][0], hi)  # merge slivers leftwards
            else:
                pieces.append((lo, hi))
        for lo, hi in pieces:
            if sentence.text[lo:hi].strip():
                out.append(
                    Clause(sentence.start + lo, sentence.start + hi, sentence.text[lo:hi])
                )
    if not out and text:
        out = [Clause(0, len(text), text)]
    return out


class LLMClauseSplitter:
    """Clause splitter backed by an OpenAI-compatible chat endpoint.

    Falls back to the heuristic splitter whenever the model's output cannot be
    realigned with the source text, which keeps labeling deterministic.
    """

    PROMPT = (
        "Split the text into clauses. A clause is a group of words containing a "
        "subject and a verb. Return a JSON array of the clause strings, in order, "
        "whose concatenation reproduces the text exactly (keep all whitespace and "
        "punctuation). Return only the JSON array.\n\nText:\n"
    )

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def __call__(self, text: str) -> list[Clause]:
        try:
            raw = self.client.complete(self.PROMPT + text, max_tokens=4096)
            pieces = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
        except Exception:
            return heuristic_clauses(text)
        return realign(text, [str(p) for p in pieces]) or heuristic_clauses(text)


def realign(text: str, pieces: list[str]) -> list[Clause]:
    """Map model-returned clause strings back onto offsets in ``text``."""
    clauses: list[Clause] = []
    cursor = 0
    for piece in pieces:
        stripped = piece.strip()
        if not stripped:
            continue
        idx = text.find(stripped, cursor)
        if idx < 0:
            return []
        clauses.append(Clause(cursor, idx + len(stripped), text[cursor : idx + len(stripped)]))
        cursor = idx + len(stripped)
    if not clauses:
        return []
    if cursor < len(text):  # trailing whitespace belongs to the last clause
        last = clauses[-1]
        clauses[-1] = Clause(last.start, len(text), text[last.start :])
    return clauses
