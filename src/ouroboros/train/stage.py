"""Two-stage orchestration (report Section 4.2, Table 2).

Stage 1  base backbone + fresh LoRA, single-copy 512-token inputs, trains the
         segment head and the stop-gradient humanizer probe.
Stage 2  stage-1 adapter merged into the backbone, fresh LoRA on top, Repeat2
         inputs, adds the tokenwise provenance head and the mixed head.

Stage 1 exists purely as an efficiency optimization: Repeat2 doubles the tokens
processed per window, so bootstrapping representations cheaply in stage 1 lets
stage 2 run for fewer steps.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import torch

from ..config import Config
from ..modeling.model import DTYPES, load_tokenizer
from .loop import train


def merge_adapter(cfg: Config, checkpoint: Path, destination: Path) -> Path:
    """Merge a stage's LoRA adapter into the backbone weights and save it."""
    from peft import PeftModel
    from transformers import AutoModel

    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)

    base = AutoModel.from_pretrained(
        cfg.backbone.name,
        dtype=DTYPES[cfg.backbone.dtype],
        trust_remote_code=cfg.backbone.trust_remote_code,
    )
    merged = PeftModel.from_pretrained(base, Path(checkpoint) / "backbone").merge_and_unload()
    merged.save_pretrained(destination)
    load_tokenizer(cfg.backbone).save_pretrained(destination)
    del base, merged
    torch.cuda.empty_cache()
    return destination


def run_stages(
    cfg1: Config,
    cfg2: Config | None,
    device: str = "cuda",
    skip_stage1: bool = False,
) -> Path:
    """Run stage 1, merge, then run stage 2 initialized from the merged backbone."""
    stage1_ckpt = Path(cfg1.train.output_dir) / "checkpoint"
    if not skip_stage1:
        stage1_ckpt = train(cfg1, stage=1, device=device)
    if cfg2 is None:
        return stage1_ckpt

    merged_dir = Path(cfg1.train.output_dir) / "merged"
    print(f"[stage] merging stage-1 adapter into backbone -> {merged_dir}")
    merge_adapter(cfg1, stage1_ckpt, merged_dir)

    cfg2 = copy.deepcopy(cfg2)
    cfg2.backbone.name = str(merged_dir)
    # Carry the stage-1 segment and humanizer heads forward so they do not
    # restart from scratch on top of the merged backbone.
    if cfg2.train.init_from is None:
        cfg2.train.init_from = str(stage1_ckpt)
    return train(cfg2, stage=2, device=device)
