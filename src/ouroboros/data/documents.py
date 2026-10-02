"""Document record: text plus a character-aligned provenance partition."""

from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..labels import Humanizer, Provenance


@dataclass
class Span:
    start: int
    end: int
    label: int  # Provenance

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class Document:
    """A training example.

    ``spans`` must form a complete, non-overlapping, ordered partition of
    ``text`` by character offset -- the report computes every document fraction
    by character length (Section 4.3.4).
    """

    id: str
    text: str
    spans: list[Span]
    humanizer: int = int(Humanizer.HUMAN)
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        cursor = 0
        for span in self.spans:
            if span.start != cursor:
                raise ValueError(f"{self.id}: span gap/overlap at {span.start} (expected {cursor})")
            if span.end < span.start:
                raise ValueError(f"{self.id}: inverted span {span}")
            cursor = span.end
        if cursor != len(self.text):
            raise ValueError(f"{self.id}: spans cover {cursor} of {len(self.text)} chars")

    def char_counts(self) -> tuple[float, float, float]:
        counts = [0.0, 0.0, 0.0]
        for span in self.spans:
            counts[span.label] += len(span)
        return counts[0], counts[1], counts[2]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spans"] = [[s.start, s.end, s.label] for s in self.spans]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Document":
        spans = [Span(int(a), int(b), int(c)) for a, b, c in payload["spans"]]
        return cls(
            id=payload["id"],
            text=payload["text"],
            spans=spans,
            humanizer=int(payload.get("humanizer", Humanizer.HUMAN)),
            source=payload.get("source", "unknown"),
            meta=payload.get("meta", {}),
        )

    @classmethod
    def uniform(
        cls,
        id: str,
        text: str,
        label: Provenance | int,
        humanizer: int,
        source: str,
        **meta: Any,
    ) -> "Document":
        return cls(
            id=id,
            text=text,
            spans=[Span(0, len(text), int(label))],
            humanizer=int(humanizer),
            source=source,
            meta=meta,
        )


def iter_json_lines(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping malformed lines.

    Generation jobs append to these files while they run, so the last line can
    be a partial write; a crashed job leaves the same trace. Skipping is the
    right behaviour -- refusing to read 60k good records because of one torn
    line is not.
    """
    bad = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[data] skipped {bad} malformed line(s) in {path}")


def write_jsonl(path: str | Path, docs: Iterable[Document]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            doc.validate()
            fh.write(json.dumps(doc.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[Document]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Document.from_json(json.loads(line))


def read_shards(paths: Iterable[str | Path]) -> list[Document]:
    """Read every shard, expanding glob patterns (absolute paths included)."""
    docs: list[Document] = []
    for path in paths:
        matches = sorted(glob.glob(str(path))) or [str(path)]
        for candidate in matches:
            docs.extend(read_jsonl(candidate))
    return docs
