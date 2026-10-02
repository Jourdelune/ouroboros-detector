"""Typed configuration objects, loaded from YAML with dotted-key overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackboneConfig:
    #: Any causal-LM checkpoint. The report uses "a popular open-weight MoE
    #: model"; the backbone is swappable here so the recipe fits a 12 GB GPU.
    name: str = "Qwen/Qwen3-0.6B-Base"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False


@dataclass
class LoraConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    #: "Attention + dense" was the best iso-compute setting in report Table 1.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    #: Extra module name patterns to target when the backbone is an MoE and
    #: routed experts should be adapted too (report Table 1, candidates B/C).
    target_routed_experts: bool = False


@dataclass
class DataConfig:
    train_shards: list[str] = field(default_factory=list)
    eval_shards: list[str] = field(default_factory=list)
    calibration_shards: list[str] = field(default_factory=list)
    test_shards: list[str] = field(default_factory=list)
    window_tokens: int = 512
    stride_tokens: int = 256
    #: Report Section 3.1: documents shorter than this are out of scope.
    min_words: int = 50
    #: Windows per document drawn per epoch during training (None = all).
    max_windows_per_doc: int | None = 2
    #: Report Section 4.1: the mixed-authorship target fires when more than this
    #: fraction of supervised tokens fall outside the window's dominant class.
    mixed_threshold: float = 0.15
    soft_bucket_sharpness: float = 1.0
    seed: int = 0


@dataclass
class TrainConfig:
    output_dir: str = "runs/ouroboros"
    stage: int = 1
    steps: int = 2000
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 500
    seed: int = 0
    #: Loss weights. The humanizer head is trained as a stop-gradient probe with
    #: weight 0.25 in both stages (report Table 2).
    w_segment: float = 1.0
    w_token: float = 1.0
    w_mixed: float = 0.5
    w_humanizer: float = 0.25
    #: Stage 2 resumes from the stage-1 adapter merged into the backbone.
    init_from: str | None = None


@dataclass
class PostprocessConfig:
    #: Potts smoothness penalty and mixed-evidence relaxation (Section 4.3.3).
    smoothness_lambda: float = 4.0
    mixed_gamma: float = 0.5
    #: Optional class-prior adjustments delta_c added to the unary potentials.
    class_prior_delta: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    min_segment_tokens: int = 32
    #: Document decision rule from Section 5.1.
    human_fraction_threshold: float = 0.90
    ai_fraction_threshold: float = 0.80


@dataclass
class Config:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls, payload: dict[str, Any]):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, optionally applying ``a.b=value`` CLI overrides."""
    payload: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        parent = raw.pop("extends", None)
        if parent:
            parent_path = (Path(path).parent / parent).resolve()
            base = dataclasses.asdict(load_config(parent_path))
            payload = _merge(base, raw)
        else:
            payload = raw
    for override in overrides or []:
        key, _, value = override.partition("=")
        if not _:
            raise ValueError(f"malformed override {override!r}, expected key=value")
        node = payload
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)

    cfg = Config()
    for section in ("backbone", "lora", "data", "train", "postprocess"):
        if section in payload:
            setattr(cfg, section, _build(type(getattr(cfg, section)), payload[section]))
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
