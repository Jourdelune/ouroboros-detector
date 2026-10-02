"""AI-assisted text generation (report Section 2.3, following EditLens).

We start from confirmed human documents and apply AI edits of varying intensity.
The resulting (source, target) pairs are labeled by the Soft N-Grams labeler in
:mod:`ouroboros.data.soft_ngrams`, which recovers clause-level provenance and
the weighted AI fraction without any human annotation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm.auto import tqdm

from .generation import LLMClient

#: Edit instructions ordered from light touch-ups (mostly human output) to heavy
#: rewrites (mostly AI output), so the 15-bucket head sees the whole spectrum.
EDIT_INSTRUCTIONS: list[tuple[str, str]] = [
    ("light", "Fix any spelling and grammar mistakes in the text below. Change nothing else."),
    ("light", "Lightly copyedit the text below for clarity. Keep the author's wording wherever possible."),
    ("light", "Improve the punctuation and sentence flow of the text below without rewriting it."),
    ("medium", "Polish the text below to make it read more smoothly. Keep the same ideas and structure."),
    ("medium", "Rewrite the text below to be more concise while preserving every point it makes."),
    ("medium", "Make the text below more descriptive and vivid, keeping all of the original details."),
    ("medium", "Rewrite the text below in a more professional tone."),
    ("heavy", "Rewrite the text below completely in your own words, keeping only the core ideas."),
    ("heavy", "Expand the text below into a longer piece, adding new supporting detail and examples."),
    ("heavy", "Continue and finish the text below, adding at least one new paragraph of your own."),
]

PROMPT = "{instruction}\n\nReturn only the resulting text.\n\nText:\n{document}"

#: Same instructions, but pinned to the source language. Soft N-Grams compares
#: the edit against its source, so an edit that silently translates would be
#: labeled as pure generation and poison the `ai-assisted` class.
PROMPT_KEEP_LANGUAGE = (
    "{instruction}\n\nWrite your answer in the SAME LANGUAGE as the text below. "
    "Return only the resulting text.\n\nText:\n{document}"
)


@dataclass
class EditConfig:
    max_tokens: int = 768
    temperature: float = 0.9
    min_chars: int = 300
    #: Reject a "rewrite" that came back byte-identical to the source.
    require_change: bool = True


def generate_edits(
    client: LLMClient,
    documents: Sequence[str],
    ids: Sequence[str] | None = None,
    cfg: EditConfig | None = None,
    seed: int = 0,
    progress: bool = True,
    out_path: str | Path | None = None,
    chunk_size: int = 240,
) -> list[dict]:
    """Produce one AI-edited variant per input document.

    Generation runs in chunks and, when ``out_path`` is given, each chunk is
    appended to disk as soon as it completes. A long local run therefore shows
    progress and survives interruption instead of holding every result in memory
    until the very end.
    """
    cfg = cfg or EditConfig()
    rng = random.Random(seed)
    ids = list(ids) if ids is not None else [str(i) for i in range(len(documents))]

    handle = None
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")

    pairs: list[dict] = []
    bar = tqdm(total=len(documents), desc="edits", unit="doc", disable=not progress)
    try:
        for start in range(0, len(documents), chunk_size):
            chunk_docs = list(documents[start : start + chunk_size])
            chunk_ids = ids[start : start + chunk_size]
            chosen = [rng.choice(EDIT_INSTRUCTIONS) for _ in chunk_docs]
            prompts = [
                PROMPT.format(instruction=instruction, document=doc)
                for (_, instruction), doc in zip(chosen, chunk_docs)
            ]
            outputs = client.complete_batch(
                prompts, max_tokens=cfg.max_tokens, temperature=cfg.temperature
            )
            for doc_id, source, (intensity, instruction), target in zip(
                chunk_ids, chunk_docs, chosen, outputs
            ):
                target = target.strip()
                if len(target) < cfg.min_chars:
                    continue
                if cfg.require_change and target == source.strip():
                    continue
                record = {
                    "id": doc_id,
                    "source_id": doc_id,
                    "source": source,
                    "target": target,
                    "instruction": instruction,
                    "intensity": intensity,
                    "generator": getattr(client, "model_name", "local"),
                    "tier": "local",
                    "source_name": "edited",
                }
                pairs.append(record)
                if handle is not None:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if handle is not None:
                handle.flush()
            bar.update(len(chunk_docs))
            bar.set_postfix(kept=len(pairs))
    finally:
        bar.close()
        if handle is not None:
            handle.close()
    return pairs


def write_pairs(path: str | Path, pairs: list[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)
