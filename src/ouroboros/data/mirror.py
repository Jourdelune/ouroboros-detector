"""Synthetic mirroring (report Section 2.2).

Two steps, following the original Pangram technical report:

1. ask the LLM for the *topic* of a human document;
2. ask it to write a document about that topic.

The result is AI text matched in topic to a real human document, which teaches
the detector "how was this written?" rather than "what is this about?". A final
verbatim-overlap check discards mirrors that regurgitate the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from tqdm.auto import tqdm

from .generation import LLMClient

TOPIC_PROMPT = "What is the topic of this article? Answer with the topic only.\n\n{document}"
WRITE_PROMPT = "Write an article about {topic}"
WRITE_PROMPT_GROUNDED = "Write an article titled \"{title}\" about {topic}"
QA_PROMPT = "{question}"


@dataclass
class MirrorConfig:
    topic_tokens: int = 48
    body_tokens: int = 640
    temperature: float = 0.9
    #: Discard a mirror whose longest verbatim overlap with the source exceeds
    #: this fraction of the mirror's length.
    max_verbatim_overlap: float = 0.25
    min_chars: int = 400


def longest_common_substring_ratio(source: str, candidate: str) -> float:
    """Fraction of the candidate covered by its longest verbatim word run from source.

    The report's final check "compares the generated text against its original
    source and discards examples where a significant portion of the original
    document is repeated verbatim" (Section 2.2).
    """
    src_words = source.lower().split()
    cand_words = candidate.lower().split()
    if not cand_words:
        return 0.0
    matcher = SequenceMatcher(None, src_words, cand_words, autojunk=False)
    longest = matcher.find_longest_match(0, len(src_words), 0, len(cand_words))
    return longest.size / len(cand_words)


def synthetic_mirror(
    client: LLMClient,
    documents: Sequence[str],
    titles: Sequence[str] | None = None,
    cfg: MirrorConfig | None = None,
    progress: bool = True,
) -> list[dict]:
    """Generate one topic-matched synthetic mirror per input document."""
    cfg = cfg or MirrorConfig()
    titles = list(titles) if titles is not None else [""] * len(documents)

    topics = client.complete_batch(
        [TOPIC_PROMPT.format(document=d) for d in documents],
        max_tokens=cfg.topic_tokens,
        temperature=0.3,
    )
    prompts = [
        WRITE_PROMPT_GROUNDED.format(title=t, topic=topic) if t else WRITE_PROMPT.format(topic=topic)
        for t, topic in zip(titles, topics)
    ]
    bodies = client.complete_batch(prompts, max_tokens=cfg.body_tokens, temperature=cfg.temperature)

    out: list[dict] = []
    iterator = zip(documents, topics, bodies)
    if progress:
        iterator = tqdm(iterator, total=len(documents), desc="mirror")
    for source, topic, body in iterator:
        body = body.strip()
        if len(body) < cfg.min_chars:
            continue
        if longest_common_substring_ratio(source, body) > cfg.max_verbatim_overlap:
            continue  # the mirror repeats the human source; discard it
        out.append({"topic": topic.strip(), "text": body})
    return out


def qa_mirror(client: LLMClient, questions: Sequence[str], cfg: MirrorConfig | None = None) -> list[dict]:
    """For Q&A corpora the question alone is a sufficient mirror prompt."""
    cfg = cfg or MirrorConfig()
    answers = client.complete_batch(
        [QA_PROMPT.format(question=q) for q in questions],
        max_tokens=cfg.body_tokens,
        temperature=cfg.temperature,
    )
    return [
        {"topic": q, "text": a.strip()} for q, a in zip(questions, answers) if len(a.strip()) >= cfg.min_chars
    ]
