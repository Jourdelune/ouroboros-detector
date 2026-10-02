"""Humanizer attacks and incidental corruption.

Report Section 3.4: the humanizer head is a four-way document-level classifier.
It must fire on *deliberate* evasion (typo injection, casing changes, synonym
substitution, homoglyph attacks, commercial "humanizer" rewrites) and must not
fire on incidental artifacts (PDF extraction errors, OCR noise, copy-paste
corruption, Unicode-normalization problems).

Every transform here is character-local, so a transformed document keeps its
provenance spans aligned by construction: we re-derive the span boundaries from
the running offset map rather than assuming lengths are preserved.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from .documents import Document, Span

HOMOGLYPHS = {
    "a": "а",  # Cyrillic a
    "c": "с",
    "e": "е",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
    "A": "А",
    "B": "В",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
}

ZERO_WIDTH = ["​", "‌", "‍", "⁠"]

# Small, dependency-free synonym table aimed at the register LLMs over-use.
SYNONYMS = {
    "utilize": "use", "utilizes": "uses", "leverage": "use", "leverages": "uses",
    "furthermore": "also", "moreover": "besides", "however": "but", "therefore": "so",
    "additionally": "also", "crucial": "key", "essential": "vital", "significant": "big",
    "numerous": "many", "various": "several", "demonstrate": "show", "demonstrates": "shows",
    "facilitate": "help", "individuals": "people", "obtain": "get", "require": "need",
    "requires": "needs", "provide": "give", "provides": "gives", "ensure": "make sure",
    "comprehensive": "complete", "robust": "strong", "delve": "dig", "intricate": "complex",
    "realm": "area", "pivotal": "central", "myriad": "many", "underscore": "stress",
    "showcase": "show", "foster": "build", "navigate": "handle", "landscape": "field",
}

_WORD = re.compile(r"\b[A-Za-z]{3,}\b")


@dataclass
class OffsetText:
    """A rewritten string together with a map from new -> old character offsets."""

    text: str
    origin: list[int]

    @classmethod
    def identity(cls, text: str) -> "OffsetText":
        return cls(text, list(range(len(text))))


def _rewrite(ot: OffsetText, edits: list[tuple[int, int, str]]) -> OffsetText:
    """Apply non-overlapping ``(start, end, replacement)`` edits, tracking offsets."""
    edits = sorted(edits, key=lambda e: e[0])
    out_chars: list[str] = []
    out_origin: list[int] = []
    cursor = 0
    for start, end, replacement in edits:
        if start < cursor:
            continue
        out_chars.append(ot.text[cursor:start])
        out_origin.extend(ot.origin[cursor:start])
        out_chars.append(replacement)
        # Replacement characters inherit the origin of the span they replace so
        # that provenance spans stay aligned.
        anchor = ot.origin[start] if start < len(ot.origin) else (ot.origin[-1] if ot.origin else 0)
        out_origin.extend([anchor] * len(replacement))
        cursor = end
    out_chars.append(ot.text[cursor:])
    out_origin.extend(ot.origin[cursor:])
    return OffsetText("".join(out_chars), out_origin)


def _sample_words(text: str, rate: float, rng: random.Random) -> list[re.Match]:
    matches = list(_WORD.finditer(text))
    if not matches:
        return []
    k = max(1, int(len(matches) * rate))
    return rng.sample(matches, min(k, len(matches)))


# ----------------------------------------------------------------------
# deliberate evasion
# ----------------------------------------------------------------------
def typo_injection(ot: OffsetText, rng: random.Random, rate: float = 0.02) -> OffsetText:
    edits = []
    for m in _sample_words(ot.text, rate, rng):
        word = m.group()
        i = rng.randrange(len(word) - 1)
        mode = rng.choice(("swap", "drop", "double"))
        if mode == "swap":
            new = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        elif mode == "drop":
            new = word[:i] + word[i + 1 :]
        else:
            new = word[: i + 1] + word[i] + word[i + 1 :]
        edits.append((m.start(), m.end(), new))
    return _rewrite(ot, edits)


def casing_attack(ot: OffsetText, rng: random.Random, rate: float = 0.08) -> OffsetText:
    edits = []
    for m in _sample_words(ot.text, rate, rng):
        word = m.group()
        new = word.lower() if word[0].isupper() else word.capitalize()
        if new != word:
            edits.append((m.start(), m.end(), new))
    return _rewrite(ot, edits)


def synonym_substitution(ot: OffsetText, rng: random.Random, rate: float = 0.6) -> OffsetText:
    edits = []
    for m in _WORD.finditer(ot.text):
        repl = SYNONYMS.get(m.group().lower())
        if repl and rng.random() < rate:
            if m.group()[0].isupper():
                repl = repl.capitalize()
            edits.append((m.start(), m.end(), repl))
    return _rewrite(ot, edits)


def homoglyph_attack(ot: OffsetText, rng: random.Random, rate: float = 0.01) -> OffsetText:
    edits = []
    for i, ch in enumerate(ot.text):
        if ch in HOMOGLYPHS and rng.random() < rate:
            edits.append((i, i + 1, HOMOGLYPHS[ch]))
    return _rewrite(ot, edits)


def zero_width_injection(ot: OffsetText, rng: random.Random, rate: float = 0.01) -> OffsetText:
    edits = []
    for m in _sample_words(ot.text, rate, rng):
        edits.append((m.end(), m.end(), rng.choice(ZERO_WIDTH)))
    return _rewrite(ot, edits)


def punctuation_swap(ot: OffsetText, rng: random.Random, rate: float = 0.5) -> OffsetText:
    """Replace the em-dashes and curly quotes that betray LLM output."""
    table = {"—": " - ", "–": "-", "’": "'", "“": '"', "”": '"'}
    edits = [
        (i, i + 1, table[ch])
        for i, ch in enumerate(ot.text)
        if ch in table and rng.random() < rate
    ]
    return _rewrite(ot, edits)


EVASION_ATTACKS: dict[str, Callable[[OffsetText, random.Random], OffsetText]] = {
    "typo": typo_injection,
    "casing": casing_attack,
    "synonym": synonym_substitution,
    "homoglyph": homoglyph_attack,
    "zero_width": zero_width_injection,
    "punctuation": punctuation_swap,
}


# ----------------------------------------------------------------------
# incidental artifacts -- explicitly NOT humanization (Section 3.4)
# ----------------------------------------------------------------------
def pdf_extraction_noise(ot: OffsetText, rng: random.Random) -> OffsetText:
    """Hyphenated line breaks, stray newlines, ligature loss."""
    edits = []
    for m in _sample_words(ot.text, 0.05, rng):
        word = m.group()
        if len(word) > 6:
            i = len(word) // 2
            edits.append((m.start(), m.end(), word[:i] + "-\n" + word[i:]))
    for i, ch in enumerate(ot.text):
        if ch == " " and rng.random() < 0.01:
            edits.append((i, i + 1, "\n"))
    return _rewrite(ot, edits)


def ocr_noise(ot: OffsetText, rng: random.Random) -> OffsetText:
    table = {"l": "1", "I": "l", "O": "0", "0": "O", "S": "5", "rn": "m"}
    edits = []
    for i, ch in enumerate(ot.text):
        if ch in table and rng.random() < 0.01:
            edits.append((i, i + 1, table[ch]))
    return _rewrite(ot, edits)


def mojibake(ot: OffsetText, rng: random.Random) -> OffsetText:
    """Unicode-normalization damage of the kind copy-paste produces."""
    text = unicodedata.normalize("NFKD", ot.text)
    edits = []
    for i, ch in enumerate(ot.text):
        if ord(ch) > 127 and rng.random() < 0.7:
            broken = ch.encode("utf-8").decode("latin-1", errors="replace")
            edits.append((i, i + 1, broken))
    del text
    return _rewrite(ot, edits)


INCIDENTAL_ARTIFACTS: dict[str, Callable[[OffsetText, random.Random], OffsetText]] = {
    "pdf": pdf_extraction_noise,
    "ocr": ocr_noise,
    "mojibake": mojibake,
}


def apply_transforms(
    doc: Document,
    names: list[str],
    rng: random.Random,
    new_humanizer: int | None = None,
    suffix: str = "",
) -> Document:
    """Apply named transforms to a document, keeping provenance spans aligned."""
    registry = {**EVASION_ATTACKS, **INCIDENTAL_ARTIFACTS}
    ot = OffsetText.identity(doc.text)
    for name in names:
        ot = registry[name](ot, rng)

    # Re-derive spans: a new character belongs to the span that owned its origin.
    boundaries = [s.end for s in doc.spans]
    spans: list[Span] = []
    cursor = 0
    for span_idx, span in enumerate(doc.spans):
        limit = boundaries[span_idx]
        end = cursor
        while end < len(ot.text) and ot.origin[end] < limit:
            end += 1
        if span_idx == len(doc.spans) - 1:
            end = len(ot.text)
        if end > cursor or span_idx == len(doc.spans) - 1:
            spans.append(Span(cursor, end, span.label))
        cursor = end
    spans = [s for s in spans if len(s) > 0] or [Span(0, len(ot.text), doc.spans[0].label)]
    spans[-1] = Span(spans[-1].start, len(ot.text), spans[-1].label)

    return Document(
        id=f"{doc.id}{suffix}",
        text=ot.text,
        spans=spans,
        humanizer=doc.humanizer if new_humanizer is None else new_humanizer,
        source=doc.source,
        meta={**doc.meta, "transforms": names},
    )
