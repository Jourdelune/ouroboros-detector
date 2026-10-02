"""LLM clients used to build AI and AI-assisted training text.

Two backends: a local transformers model (the default -- it keeps the whole
pipeline runnable on one GPU) and any OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from tqdm.auto import tqdm


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.9) -> str: ...

    def complete_batch(
        self, prompts: Sequence[str], max_tokens: int = 512, temperature: float = 0.9
    ) -> list[str]: ...


@dataclass
class LocalHFClient:
    """Batched chat generation with a local instruct model."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 16
    max_input_tokens: int = 1536

    def __post_init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=self.dtype, device_map=self.device
        )
        self.model.eval()

    def _render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.inference_mode()
    def complete_batch(
        self, prompts: Sequence[str], max_tokens: int = 512, temperature: float = 0.9
    ) -> list[str]:
        out: list[str] = []
        starts = range(0, len(prompts), self.batch_size)
        for start in tqdm(starts, desc="generate", unit="batch", leave=False):
            chunk = [self._render(p) for p in prompts[start : start + self.batch_size]]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=False,
            ).to(self.device)
            generated = self.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            for row, ids in zip(enc["input_ids"], generated):
                out.append(
                    self.tokenizer.decode(ids[len(row) :], skip_special_tokens=True).strip()
                )
        return out

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.9) -> str:
        return self.complete_batch([prompt], max_tokens, temperature)[0]


@dataclass
class OpenAICompatClient:
    """Any OpenAI-compatible chat endpoint (vLLM, llama.cpp server, a provider)."""

    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_workers: int = 8

    def __post_init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.base_url, api_key=os.environ.get(self.api_key_env, "not-needed")
        )

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.9) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_batch(
        self, prompts: Sequence[str], max_tokens: int = 512, temperature: float = 0.9
    ) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self.complete(p, max_tokens, temperature), prompts))


def build_client(spec: str, **kwargs) -> LLMClient:
    """``local:<hf-model>`` or ``openai:<model>[@<base_url>]``."""
    kind, _, rest = spec.partition(":")
    if kind == "local":
        return LocalHFClient(model_name=rest or "Qwen/Qwen2.5-1.5B-Instruct", **kwargs)
    if kind == "openai":
        model, _, base_url = rest.partition("@")
        return OpenAICompatClient(model=model, base_url=base_url or None)
    raise ValueError(f"unknown client spec {spec!r}")
