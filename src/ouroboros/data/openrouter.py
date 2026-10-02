"""Diverse AI-rewrite generation through OpenRouter, under a hard budget cap.

Report Section 2.2/2.3: the AI half of the corpus must span many generators so
the detector learns "how was this written?" rather than the fingerprint of one
model. RAID's generations come from 2024-era open models, so we add rewrites of
real human documents from a portfolio spanning free open-weight models up to
current frontier models.

Cost control is explicit: every tier has its own cap, spend is read back from
OpenRouter's reported per-request cost, and generation stops the moment a cap is
reached. Results stream to disk as they arrive, so a crash never loses work.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import requests
from tqdm.auto import tqdm

API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: str
    weight: float


#: Portfolio spanning five price tiers. Weights are relative sampling shares
#: *within* a tier; the tier budgets below decide how much each tier produces.
PORTFOLIO: list[ModelSpec] = [
    # --- free open-weight models (no spend, rate limited) ------------------
    ModelSpec("nvidia/nemotron-3-ultra-550b-a55b:free", "free", 1.0),
    ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", "free", 1.0),
    ModelSpec("nvidia/nemotron-3.5-lightning:free", "free", 1.0),
    ModelSpec("google/gemma-4-31b-it:free", "free", 1.0),
    ModelSpec("google/gemma-4-26b-a4b-it:free", "free", 1.0),
    ModelSpec("qwen/qwen3.8-27b:free", "free", 1.0),
    ModelSpec("z-ai/glm-5.2:free", "free", 1.0),
    ModelSpec("thinkingmachines/inkling:free", "free", 1.0),
    ModelSpec("nex-agi/nex-n2.5-pro:free", "free", 1.0),
    ModelSpec("poolside/laguna-s-2.1:free", "free", 1.0),
    # --- cheap -------------------------------------------------------------
    ModelSpec("mistralai/mistral-nemo", "cheap", 1.0),
    ModelSpec("meta-llama/llama-3.1-8b-instruct", "cheap", 1.0),
    ModelSpec("mistralai/mistral-small-24b-instruct-2501", "cheap", 1.0),
    ModelSpec("openai/gpt-oss-20b", "cheap", 1.0),
    ModelSpec("qwen/qwen3.7-flash", "cheap", 1.0),
    ModelSpec("google/gemma-3-12b-it", "cheap", 1.0),
    ModelSpec("deepseek/deepseek-v4-flash", "cheap", 1.0),
    ModelSpec("qwen/qwen3-30b-a3b-instruct-2507", "cheap", 1.0),
    ModelSpec("microsoft/phi-4", "cheap", 1.0),
    ModelSpec("amazon/nova-lite-v1", "cheap", 1.0),
    ModelSpec("openai/gpt-5-nano", "cheap", 1.0),
    # --- mid ---------------------------------------------------------------
    ModelSpec("openai/gpt-6-luna", "mid", 1.0),
    ModelSpec("deepseek/deepseek-chat", "mid", 1.0),
    ModelSpec("meta-llama/llama-4-maverick", "mid", 1.0),
    ModelSpec("z-ai/glm-4.5-air", "mid", 1.0),
    ModelSpec("minimax/minimax-m2", "mid", 1.0),
    ModelSpec("openai/gpt-5.6-luna", "mid", 1.0),
    ModelSpec("google/gemini-3.1-flash-lite", "mid", 1.0),
    ModelSpec("mistralai/mistral-medium-3.1", "mid", 1.0),
    ModelSpec("moonshotai/kimi-k2.5", "mid", 1.0),
    ModelSpec("x-ai/grok-4.3", "mid", 1.0),
    # --- frontier ----------------------------------------------------------
    ModelSpec("anthropic/claude-haiku-4.5", "frontier", 1.0),
    ModelSpec("google/gemini-3.8-flash", "frontier", 1.0),
    ModelSpec("qwen/qwen3.7-max", "frontier", 1.0),
    ModelSpec("x-ai/grok-4.7", "frontier", 1.0),
    ModelSpec("openai/gpt-5.4-mini", "frontier", 1.0),
    # --- ultra (a small but real taste of the top of the market) -----------
    ModelSpec("anthropic/claude-sonnet-5", "ultra", 1.0),
    ModelSpec("anthropic/claude-opus-5", "ultra", 0.5),
    ModelSpec("openai/gpt-5.6-sol", "ultra", 1.0),
    ModelSpec("openai/gpt-5", "ultra", 1.0),
    ModelSpec("google/gemini-3.1-pro-preview", "ultra", 1.0),
]

#: Spend caps per tier. "free" gets a real (small) cap rather than zero: some
#: ":free" endpoints still report a residual cost, and a zero cap would lock the
#: whole tier out after a single call.
DEFAULT_TIER_BUDGETS = {
    "free": 0.06,
    "cheap": 0.28,
    "mid": 0.88,
    "frontier": 0.58,
    "ultra": 0.60,
}

#: Target share of *samples* per tier. Volume is driven by these weights while
#: the budgets above act as hard stops, so the cheap tiers supply most of the
#: data and the expensive tiers supply diversity at the top of the market.
DEFAULT_TIER_VOLUME = {
    "free": 0.34,
    "cheap": 0.46,
    "mid": 0.155,
    "frontier": 0.035,
    "ultra": 0.010,
}


@dataclass
class Budget:
    """Thread-safe per-tier spend tracker with hard caps."""

    caps: dict[str, float]
    spent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def remaining(self, tier: str) -> float:
        with self._lock:
            return self.caps.get(tier, 0.0) - self.spent[tier]

    def exhausted(self, tier: str) -> bool:
        return self.remaining(tier) <= 0.0

    def all_exhausted(self) -> bool:
        return all(self.exhausted(t) for t in self.caps)

    def charge(self, tier: str, cost: float) -> None:
        with self._lock:
            self.spent[tier] += cost
            self.calls[tier] += 1

    @property
    def total(self) -> float:
        with self._lock:
            return sum(self.spent.values())

    def report(self) -> dict:
        with self._lock:
            return {
                "total_usd": round(sum(self.spent.values()), 4),
                "by_tier": {
                    t: {"usd": round(self.spent[t], 4), "calls": self.calls[t], "cap": self.caps[t]}
                    for t in self.caps
                },
            }


class OpenRouterClient:
    """Minimal, budget-aware OpenRouter chat client."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        referer: str = "https://github.com/ouroboros-detector",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": referer,
            "X-Title": "ouroboros",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()

    def chat(self, model: str, prompt: str, max_tokens: int, temperature: float) -> tuple[str, float]:
        """Return ``(text, cost_usd)``; raises on unrecoverable failure."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},
            # Keep reasoning models from burning the budget on hidden tokens.
            "reasoning": {"exclude": True, "effort": "low"},
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    API_URL, headers=self.headers, json=payload, timeout=self.timeout
                )
                if response.status_code == 429:
                    time.sleep(2.0 * (attempt + 1) + random.random())
                    continue
                if response.status_code >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                body = response.json()
                if "error" in body and not body.get("choices"):
                    raise RuntimeError(str(body["error"])[:200])
                text = (body["choices"][0]["message"].get("content") or "").strip()
                cost = float(body.get("usage", {}).get("cost", 0.0) or 0.0)
                return text, cost
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last_error = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"{model}: {last_error}")


class ModelSampler:
    """Samples a tier by target volume share, then a model inside that tier.

    Sampling the tier first is what keeps the sample mix under control: a flat
    draw over the portfolio would spend the budget on whichever tier happens to
    have the most entries, not on the mix we actually want.
    """

    def __init__(
        self,
        portfolio: list[ModelSpec],
        budget: Budget,
        volume: dict[str, float] | None = None,
        failure_limit: int = 8,
    ):
        self.portfolio = portfolio
        self.budget = budget
        self.volume = dict(volume or DEFAULT_TIER_VOLUME)
        self.failure_limit = failure_limit
        self.failures: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _alive_by_tier(self) -> dict[str, list[ModelSpec]]:
        with self._lock:
            failures = dict(self.failures)
        alive: dict[str, list[ModelSpec]] = defaultdict(list)
        for m in self.portfolio:
            if failures.get(m.id, 0) >= self.failure_limit:
                continue
            if self.budget.exhausted(m.tier):
                continue
            alive[m.tier].append(m)
        return alive

    def pick(self, rng: random.Random) -> ModelSpec | None:
        alive = self._alive_by_tier()
        tiers = [t for t in alive if alive[t]]
        if not tiers:
            return None
        weights = [max(self.volume.get(t, 0.0), 1e-6) for t in tiers]
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        candidates = alive[tier]
        return rng.choices(candidates, weights=[m.weight for m in candidates], k=1)[0]

    def record_failure(self, model_id: str) -> None:
        with self._lock:
            self.failures[model_id] += 1


@dataclass
class JobResult:
    ok: bool
    record: dict | None = None
    cost: float = 0.0
    model: str = ""
    error: str = ""


def _wait_first(pending: set) -> tuple[set, set]:
    from concurrent.futures import FIRST_COMPLETED, wait

    done, still = wait(pending, return_when=FIRST_COMPLETED)
    return done, set(still)


def run_portfolio(
    tasks: list[dict],
    prompt_fn: Callable[[dict, ModelSpec], str],
    record_fn: Callable[[dict, ModelSpec, str], dict | None],
    out_path: str | Path,
    tier_budgets: dict[str, float] | None = None,
    tier_volume: dict[str, float] | None = None,
    portfolio: list[ModelSpec] | None = None,
    max_tokens: int = 900,
    temperature: float = 0.9,
    workers: int = 12,
    seed: int = 0,
    api_key: str | None = None,
) -> dict:
    """Fan `tasks` out across the portfolio, streaming accepted records to disk.

    ``prompt_fn`` builds the prompt for a task, ``record_fn`` turns the model's
    reply into a record (returning ``None`` rejects it, e.g. too short).
    """
    from concurrent.futures import ThreadPoolExecutor

    budget = Budget(caps=dict(tier_budgets or DEFAULT_TIER_BUDGETS))
    sampler = ModelSampler(portfolio or PORTFOLIO, budget, tier_volume)
    client = OpenRouterClient(api_key=api_key)
    rng = random.Random(seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    pick_lock = threading.Lock()
    by_model: dict[str, int] = defaultdict(int)
    rejected = errors = 0

    def worker(task: dict) -> JobResult:
        with pick_lock:
            model = sampler.pick(rng)
        if model is None:
            return JobResult(ok=False, error="budget exhausted")
        try:
            text, cost = client.chat(
                model.id, prompt_fn(task, model), max_tokens, temperature
            )
        except Exception as exc:  # noqa: BLE001
            sampler.record_failure(model.id)
            return JobResult(ok=False, model=model.id, error=str(exc)[:160])
        budget.charge(model.tier, cost)
        record = record_fn(task, model, text)
        if record is None:
            return JobResult(ok=False, cost=cost, model=model.id, error="rejected")
        return JobResult(ok=True, record=record, cost=cost, model=model.id)

    handle = out_path.open("a", encoding="utf-8")
    pending: set = set()
    queue = iter(tasks)
    depth = max(workers * 2, 8)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            bar = tqdm(total=len(tasks), desc="openrouter", unit="doc")

            def refill() -> None:
                # Only ever keep `depth` requests in flight, so hitting a budget
                # cap stops spending almost immediately instead of draining a
                # fully pre-submitted queue.
                while len(pending) < depth and not budget.all_exhausted():
                    try:
                        pending.add(pool.submit(worker, next(queue)))
                    except StopIteration:
                        return

            refill()
            while pending:
                done, pending = _wait_first(pending)
                for future in done:
                    result = future.result()
                    if result.ok and result.record is not None:
                        with write_lock:
                            handle.write(json.dumps(result.record, ensure_ascii=False) + "\n")
                            handle.flush()
                        by_model[result.model] += 1
                    elif result.error == "rejected":
                        rejected += 1
                    else:
                        errors += 1
                    bar.update(1)
                bar.set_postfix(
                    usd=f"{budget.total:.3f}", ok=sum(by_model.values()), err=errors
                )
                if not budget.all_exhausted():
                    refill()
            bar.close()
    finally:
        handle.close()

    report = budget.report()
    report["accepted"] = sum(by_model.values())
    report["rejected"] = rejected
    report["errors"] = errors
    report["by_model"] = dict(sorted(by_model.items(), key=lambda kv: -kv[1]))
    return report
