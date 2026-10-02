"""Explanations: which words pushed the model towards "AI" or "human".

Three complementary views, all expressed against the same scalar *AI score*:

* **Gradient x Input** -- one backward pass per window; the derivative of the
  score with respect to each input embedding, dotted with that embedding. Fast,
  local (first-order) and somewhat noisy.
* **Integrated Gradients** -- the gradient averaged along a straight path from
  a zero-embedding baseline to the real input, times the input. Satisfies
  completeness (attributions sum to ``score(x) - score(baseline)``) and is far
  less noisy, at ``steps`` times the cost.
* **Word / sentence occlusion** -- remove each word (or sentence) in turn and
  re-score the document. No gradients: a direct, model-agnostic "what if it
  were not there", usually the most faithful and readable of the three.

Scores (the ``target``):

* ``"token"``: mean over the window's tokens of the token head's AI log-odds,
  ``log(p_assisted + p_ai) - log(p_human)``. An optional per-token mask restricts
  the mean, e.g. to the tokens the decoder labelled AI ("why is *this* AI?").
* ``"document"``: segment head log-odds that the weighted AI fraction is above
  one half, ``logsumexp(buckets f > .5) - logsumexp(buckets f < .5)``.

Log-odds rather than probabilities keep gradients from vanishing once the
softmax saturates on a confidently classified document. Under Repeat2 every
token appears twice in the input; its attribution is the sum over both copies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from ..data.clauses import split_sentences
from ..data.dataset import window_bounds
from ..labels import BUCKET_CENTERS
from ..modeling.model import OuroborosModel, OuroborosOutput

TARGETS = ("token", "document")
METHODS = ("gradxinput", "integrated_gradients", "word_occlusion")

_WORD = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)
_ABOVE = torch.from_numpy(BUCKET_CENTERS > 0.5)
_BELOW = torch.from_numpy(BUCKET_CENTERS < 0.5)


def ai_score(
    out: OuroborosOutput,
    target: str,
    lengths: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B,)`` AI score for each window of a Repeat2 batch.

    ``token_mask`` (``(B, S)``, token target only) selects the tokens averaged.
    """
    if target == "token":
        logits = out.token_logits.float()  # (B, S, 3)
        per_token = torch.logsumexp(logits[..., 1:], -1) - logits[..., 0]
        positions = torch.arange(per_token.shape[1], device=per_token.device)
        mask = (positions[None, :] < lengths[:, None]).float()
        if token_mask is not None:
            mask = mask * token_mask[:, : mask.shape[1]].float()
        return (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)
    if target == "document":
        seg = out.segment_logits.float()
        above = torch.logsumexp(seg[:, _ABOVE.to(seg.device)], -1)
        below = torch.logsumexp(seg[:, _BELOW.to(seg.device)], -1)
        return above - below
    raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")


def _repeat2_batch(ids: np.ndarray, chunk: list[tuple[int, int]], pad_id: int):
    lengths = [hi - lo for lo, hi in chunk]
    max_len = max(lengths)
    input_ids = torch.full((len(chunk), 2 * max_len), pad_id, dtype=torch.long)
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
    return input_ids, attention, last_index, supervised_start, torch.tensor(lengths)


def _window_attribution(
    model: OuroborosModel,
    window_ids: np.ndarray,
    pad_id: int,
    target: str,
    method: str,
    steps: int,
    device: str,
    token_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Per-token attribution ``(n,)`` for one window, and the window's score."""
    n = len(window_ids)
    input_ids, attention, last, start, lengths = _repeat2_batch(
        window_ids, [(0, n)], pad_id
    )
    input_ids, attention = input_ids.to(device), attention.to(device)
    last, start, lengths = last.to(device), start.to(device), lengths.to(device)
    mask = None if token_mask is None else torch.from_numpy(token_mask)[None].to(device)
    embed = model.backbone.get_input_embeddings()
    with torch.no_grad():
        full = embed(input_ids).detach()

    def score_and_grad(embeds: torch.Tensor) -> tuple[float, torch.Tensor]:
        embeds = embeds.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, attention, last, start, inputs_embeds=embeds)
        score = ai_score(out, target, lengths, mask).sum()
        (grad,) = torch.autograd.grad(score, embeds)
        return float(score.item()), grad.float()

    if method == "gradxinput":
        score, grad = score_and_grad(full)
        attribution = (grad * full.float()).sum(-1)[0]
    elif method == "integrated_gradients":
        total = torch.zeros_like(full, dtype=torch.float32)
        # Right Riemann sum over alpha in (0, 1]; the last step is the input itself.
        for alpha in torch.linspace(1.0 / steps, 1.0, steps):
            score, grad = score_and_grad(full * alpha.item())
            total += grad
        attribution = (total / steps * full.float()).sum(-1)[0]
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    per_token = attribution[:n] + attribution[n : 2 * n]  # both Repeat2 copies
    return per_token.cpu().numpy(), score


def token_attributions(
    model: OuroborosModel,
    tokenizer,
    text: str,
    window_tokens: int,
    stride_tokens: int,
    target: str = "token",
    method: str = "gradxinput",
    steps: int = 16,
    device: str = "cuda",
    token_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """Per-token attributions over a whole document (averaged across windows).

    Returns ``(attributions (N,), token char offsets, mean window score)``.
    Positive attribution pushes the score towards AI, negative towards human.
    ``token_mask`` (``(N,)`` bool, token target only) restricts the explained
    score to those tokens; windows containing none of them are skipped.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if len(ids) == 0:
        return np.zeros(0), [], 0.0

    total = np.zeros(len(ids))
    counts = np.zeros(len(ids))
    scores = []
    model.eval()
    for lo, hi in window_bounds(len(ids), window_tokens, stride_tokens):
        window_mask = None if token_mask is None else token_mask[lo:hi]
        if window_mask is not None and not window_mask.any():
            continue
        attribution, score = _window_attribution(
            model, ids[lo:hi], tokenizer.pad_token_id, target, method, steps, device, window_mask
        )
        total[lo:hi] += attribution
        counts[lo:hi] += 1
        scores.append(score)
    return total / np.maximum(counts, 1), offsets, float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def document_scores(
    model: OuroborosModel,
    tokenizer,
    texts: list[str],
    window_tokens: int,
    stride_tokens: int,
    target: str = "token",
    device: str = "cuda",
    batch_size: int = 4,
) -> list[float]:
    """AI score of each text: window scores averaged, weighted by window length."""
    jobs = []  # (text index, ids, lo, hi)
    for index, text in enumerate(texts):
        ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
        for lo, hi in window_bounds(len(ids), window_tokens, stride_tokens) if len(ids) else []:
            jobs.append((index, ids, lo, hi))

    weighted = np.zeros(len(texts))
    weights = np.zeros(len(texts))
    model.eval()
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        rows = [ids[lo:hi] for _, ids, lo, hi in batch]
        input_ids, attention, last, sup, lengths = _repeat2_batch(
            np.concatenate(rows), _concat_bounds(rows), tokenizer.pad_token_id
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids.to(device), attention.to(device), last.to(device), sup.to(device))
        values = ai_score(out, target, lengths.to(device)).cpu().numpy()
        for (index, _, lo, hi), value in zip(batch, values):
            weighted[index] += value * (hi - lo)
            weights[index] += hi - lo
    return [float(w / c) if c else 0.0 for w, c in zip(weighted, weights)]


def _concat_bounds(rows: list[np.ndarray]) -> list[tuple[int, int]]:
    bounds, cursor = [], 0
    for row in rows:
        bounds.append((cursor, cursor + len(row)))
        cursor += len(row)
    return bounds


# --------------------------------------------------------------------------
# word grouping and sentence occlusion
# --------------------------------------------------------------------------
@dataclass
class WordScore:
    start: int
    end: int
    text: str
    score: float


def words_from_tokens(
    text: str, offsets: list[tuple[int, int]], values: np.ndarray, reduce: str = "sum"
) -> list[WordScore]:
    """Aggregate per-token values onto words (and standalone punctuation).

    A token belongs to the word containing its first non-space character, so
    BPE pieces such as ``" lever" + "aging"`` collapse onto ``leveraging``.
    Tokens made only of whitespace are dropped. ``reduce`` is ``"sum"`` for
    attributions (they are additive) and ``"mean"`` for probabilities.
    """
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    owner = np.full(len(text) + 1, -1, dtype=np.int64)
    for index, (a, b) in enumerate(words):
        owner[a:b] = index
    total = np.zeros(len(words))
    count = np.zeros(len(words))
    for (a, b), value in zip(offsets, values):
        piece = text[a:b]
        stripped = len(piece) - len(piece.lstrip())
        if a + stripped >= b:
            continue
        index = owner[a + stripped]
        if index < 0:
            continue
        total[index] += value
        count[index] += 1
    if reduce == "mean":
        total = total / np.maximum(count, 1)
    return [
        WordScore(a, b, text[a:b], float(total[i]))
        for i, (a, b) in enumerate(words)
        if count[i] > 0
    ]


def word_occlusion(
    model: OuroborosModel,
    tokenizer,
    text: str,
    window_tokens: int,
    stride_tokens: int,
    target: str = "token",
    device: str = "cuda",
    max_words: int = 400,
) -> tuple[float, list[WordScore]]:
    """Re-score the document with each word removed; ``score`` = full - without.

    Only the first ``max_words`` words are occluded (cost grows with
    words x windows).
    """
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)][:max_words]
    variants = [text] + [text[:a] + text[b:] for a, b in words]
    scores = document_scores(
        model, tokenizer, variants, window_tokens, stride_tokens, target, device, batch_size=8
    )
    base = scores[0]
    return base, [
        WordScore(a, b, text[a:b], base - without)
        for (a, b), without in zip(words, scores[1:])
    ]


@dataclass
class SentenceScore:
    start: int
    end: int
    text: str
    delta: float  # score(full) - score(without this sentence); > 0 => pushed towards AI


def sentence_occlusion(
    model: OuroborosModel,
    tokenizer,
    text: str,
    window_tokens: int,
    stride_tokens: int,
    target: str = "token",
    device: str = "cuda",
    max_sentences: int = 80,
) -> tuple[float, list[SentenceScore]]:
    """Re-score the document with each sentence removed."""
    sentences = split_sentences(text)[:max_sentences]
    if len(sentences) < 2:
        return 0.0, []
    variants = [text] + [text[: s.start] + text[s.end :] for s in sentences]
    scores = document_scores(
        model, tokenizer, variants, window_tokens, stride_tokens, target, device
    )
    base = scores[0]
    return base, [
        SentenceScore(s.start, s.end, s.text, base - without)
        for s, without in zip(sentences, scores[1:])
    ]
