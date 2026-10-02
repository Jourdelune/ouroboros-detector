"""The Ouroboros model: one shared causal backbone with four task heads.

Report Section 4.1. Three window-level heads read the hidden state at the final
supervised sequence position ``h_S`` -- under causal attention that position has
seen the whole window -- while the tokenwise provenance head projects every
position ``h_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..config import BackboneConfig, Config, LoraConfig
from ..labels import N_BUCKETS

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class OuroborosOutput:
    segment_logits: torch.Tensor  # (B, 15)
    token_logits: torch.Tensor  # (B, S, 3)
    mixed_logits: torch.Tensor  # (B, 2)
    humanizer_logits: torch.Tensor  # (B, 4)


def _expert_module_names(model: nn.Module) -> list[str]:
    """Leaf linear names that live inside a routed-expert block, if any."""
    names = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "expert" in full_name.lower() and "shared" not in full_name.lower():
            names.add(full_name.split(".")[-1])
    return sorted(names)


class OuroborosModel(nn.Module):
    """Shared backbone + segment / tokenwise / mixed / humanizer heads."""

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        # A single dense layer per task, as described in Section 4.1.
        self.segment_head = nn.Linear(hidden_size, N_BUCKETS)
        self.token_head = nn.Linear(hidden_size, 3)
        self.mixed_head = nn.Linear(hidden_size, 2)
        self.humanizer_head = nn.Linear(hidden_size, 4)
        for head in (self.segment_head, self.token_head, self.mixed_head, self.humanizer_head):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)
        self.heads = nn.ModuleDict(
            {
                "segment": self.segment_head,
                "token": self.token_head,
                "mixed": self.mixed_head,
                "humanizer": self.humanizer_head,
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_index: torch.Tensor,
        supervised_start: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> OuroborosOutput:
        """Run the backbone once and read all four heads off its hidden states.

        Args:
            input_ids: ``(B, L)``. In stage 2 ``L`` covers the Repeat2 sequence.
            attention_mask: ``(B, L)`` padding mask.
            last_index: ``(B,)`` index of the final supervised position ``h_S``.
            supervised_start: ``(B,)`` first position of the supervised copy.
                With Repeat2 this is the start of the *second* copy; the token
                head is only read out from there onwards.
            inputs_embeds: ``(B, L, H)`` optional precomputed input embeddings,
                used instead of ``input_ids`` (gradient attribution needs them).
        """
        hidden = self.backbone(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        hidden = hidden.to(self.segment_head.weight.dtype)

        batch = hidden.shape[0]
        arange = torch.arange(batch, device=hidden.device)
        pooled = hidden[arange, last_index]  # h_S

        segment_logits = self.segment_head(pooled)
        mixed_logits = self.mixed_head(pooled)
        # Stop-gradient probe: the humanizer loss must not update the backbone
        # (Section 4.1 / Table 2).
        humanizer_logits = self.humanizer_head(pooled.detach())

        if supervised_start is None:
            token_logits = self.token_head(hidden)
        else:
            window = int((last_index - supervised_start).max().item()) + 1
            offsets = supervised_start.unsqueeze(1) + torch.arange(
                window, device=hidden.device
            ).unsqueeze(0)
            offsets = offsets.clamp(max=hidden.shape[1] - 1)
            gathered = hidden.gather(
                1, offsets.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )
            token_logits = self.token_head(gathered)

        return OuroborosOutput(
            segment_logits=segment_logits,
            token_logits=token_logits,
            mixed_logits=mixed_logits,
            humanizer_logits=humanizer_logits,
        )

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone")
        torch.save(self.heads.state_dict(), path / "heads.pt")

    def load_heads(self, path: str | Path, device: str | torch.device = "cpu") -> None:
        state = torch.load(Path(path) / "heads.pt", map_location=device)
        self.heads.load_state_dict(state)


def build_backbone(cfg: BackboneConfig, lora_cfg: LoraConfig | None, device: str = "cuda"):
    """Load the causal backbone (no LM head) and attach a fresh LoRA adapter."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    hf_config = AutoConfig.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModel.from_pretrained(
        cfg.name,
        dtype=DTYPES[cfg.dtype],
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    if lora_cfg is not None:
        targets = list(lora_cfg.target_modules)
        if lora_cfg.target_routed_experts:
            targets = sorted(set(targets) | set(_expert_module_names(model)))
        present = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
        targets = [t for t in targets if t in present]
        if not targets:
            raise ValueError(f"no LoRA target modules of {lora_cfg.target_modules} found in {cfg.name}")
        peft_cfg = PeftLoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias="none",
            target_modules=targets,
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, peft_cfg)

    hidden_size = getattr(hf_config, "hidden_size", None) or hf_config.text_config.hidden_size
    return model, hidden_size


def build_model(cfg: Config, device: str = "cuda", with_lora: bool = True) -> OuroborosModel:
    backbone, hidden = build_backbone(cfg.backbone, cfg.lora if with_lora else None, device)
    model = OuroborosModel(backbone, hidden)
    model.heads.to(DTYPES[cfg.backbone.dtype])
    return model.to(device)


def load_tokenizer(cfg: BackboneConfig):
    tok = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
