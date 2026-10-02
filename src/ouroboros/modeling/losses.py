"""Multi-task loss for the Ouroboros heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import TrainConfig
from ..labels import IGNORE_INDEX
from .model import OuroborosOutput


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy against a soft target distribution, averaged over the batch."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(targets * logp).sum(dim=-1).mean()


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy that returns 0 when every label in the batch is masked."""
    valid = labels != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), labels[valid])


def compute_losses(
    out: OuroborosOutput,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
    stage: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of the active task losses for the given training stage."""
    logs: dict[str, float] = {}
    total = out.segment_logits.sum() * 0.0

    seg_mask = batch["segment_mask"].bool()
    if bool(seg_mask.any()):
        seg_loss = soft_cross_entropy(
            out.segment_logits[seg_mask], batch["segment_target"][seg_mask]
        )
        total = total + cfg.w_segment * seg_loss
        logs["loss/segment"] = float(seg_loss.detach())

    hum_loss = _masked_ce(out.humanizer_logits, batch["humanizer_label"])
    total = total + cfg.w_humanizer * hum_loss
    logs["loss/humanizer"] = float(hum_loss.detach())

    if stage >= 2:
        token_logits = out.token_logits.reshape(-1, out.token_logits.shape[-1])
        token_labels = batch["token_labels"].reshape(-1)
        tok_loss = _masked_ce(token_logits, token_labels)
        total = total + cfg.w_token * tok_loss
        logs["loss/token"] = float(tok_loss.detach())

        mixed_loss = _masked_ce(out.mixed_logits, batch["mixed_label"])
        total = total + cfg.w_mixed * mixed_loss
        logs["loss/mixed"] = float(mixed_loss.detach())

    logs["loss/total"] = float(total.detach())
    return total, logs
