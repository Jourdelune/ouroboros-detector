"""Training loop shared by both stages."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..config import Config, save_config
from ..data.dataset import WindowDataset, collate, tokenize_documents
from ..data.documents import read_shards
from ..labels import BUCKET_CENTERS, IGNORE_INDEX
from ..modeling.losses import compute_losses
from ..modeling.model import DTYPES, OuroborosModel, build_model, load_tokenizer


@dataclass
class EvalReport:
    steps: int
    metrics: dict[str, float]


def _param_groups(model: OuroborosModel, cfg: Config):
    head_params, lora_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.startswith("heads.") or "_head." in name else lora_params).append(param)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.train.lr})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.train.head_lr})
    return groups


def _prepare_trainable(model: OuroborosModel) -> None:
    """Keep adapter and head parameters in fp32 for stable optimizer states."""
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def make_loader(
    docs, tokenizer, cfg: Config, stage: int, shuffle: bool, batch_size: int | None = None
) -> DataLoader:
    tokenized = tokenize_documents(docs, tokenizer)
    dataset = WindowDataset(tokenized, cfg.data, stage)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )


@torch.no_grad()
def evaluate(model: OuroborosModel, loader: DataLoader, cfg: Config, stage: int, max_batches: int = 60) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    f_true, f_pred = [], []
    tok_correct = tok_total = 0
    hum_correct = hum_total = 0
    losses: list[float] = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["last_index"],
                batch["supervised_start"] if stage >= 2 else None,
            )
            loss, _ = compute_losses(out, batch, cfg.train, stage)
        losses.append(float(loss))

        probs = torch.softmax(out.segment_logits.float(), dim=-1).cpu().numpy()
        f_pred.extend(probs @ BUCKET_CENTERS)
        f_true.extend(batch["f_ai"].cpu().numpy())

        hum_mask = batch["humanizer_label"] != IGNORE_INDEX
        if bool(hum_mask.any()):
            pred = out.humanizer_logits.argmax(-1)
            hum_correct += int((pred[hum_mask] == batch["humanizer_label"][hum_mask]).sum())
            hum_total += int(hum_mask.sum())

        if stage >= 2:
            mask = batch["token_labels"] != IGNORE_INDEX
            if bool(mask.any()):
                pred = out.token_logits.argmax(-1)
                tok_correct += int((pred[mask] == batch["token_labels"][mask]).sum())
                tok_total += int(mask.sum())

    model.train()
    f_true_a, f_pred_a = np.array(f_true), np.array(f_pred)
    metrics = {
        "eval/loss": float(np.mean(losses)) if losses else float("nan"),
        "eval/f_ai_mae": float(np.abs(f_true_a - f_pred_a).mean()) if len(f_true_a) else float("nan"),
        "eval/humanizer_acc": hum_correct / hum_total if hum_total else float("nan"),
    }
    # Binary separation of the two poles, the headline quantity of Section 5.
    human_mask, ai_mask = f_true_a <= 0.05, f_true_a >= 0.95
    if human_mask.any() and ai_mask.any():
        metrics["eval/pole_gap"] = float(f_pred_a[ai_mask].mean() - f_pred_a[human_mask].mean())
    if tok_total:
        metrics["eval/token_acc"] = tok_correct / tok_total
    return metrics


def train(cfg: Config, stage: int, device: str = "cuda") -> Path:
    torch.manual_seed(cfg.train.seed)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    tokenizer = load_tokenizer(cfg.backbone)
    model = build_model(cfg, device=device)
    if cfg.train.init_from:
        init_path = Path(cfg.train.init_from)
        state = torch.load(init_path / "heads.pt", map_location="cpu")
        current = model.heads.state_dict()
        carried = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        model.heads.load_state_dict(carried, strict=False)
        print(f"[train] warm-started {len(carried)} head tensors from {init_path}")
    _prepare_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] stage {stage}: {trainable/1e6:.2f}M trainable / {total/1e6:.1f}M params")

    train_docs = read_shards(cfg.data.train_shards)
    eval_docs = read_shards(cfg.data.eval_shards)
    print(f"[train] {len(train_docs)} train docs, {len(eval_docs)} eval docs")
    train_loader = make_loader(train_docs, tokenizer, cfg, stage, shuffle=True)
    eval_loader = make_loader(eval_docs, tokenizer, cfg, stage, shuffle=False) if eval_docs else None
    print(f"[train] {len(train_loader.dataset)} train windows")

    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg), weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95)
    )
    warmup = max(1, int(cfg.train.steps * cfg.train.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, cfg.train.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard: `tensorboard --logdir runs` shows both stages side by side.
    tb_dir = out_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text(
        "config",
        f"```yaml\n{yaml.safe_dump(cfg.to_dict(), sort_keys=False)}\n```",
        0,
    )
    print(f"[train] tensorboard --logdir {out_dir.parent}   (this run: {tb_dir})")

    history: list[dict] = []
    # Track the best eval checkpoint separately from the last one. With a cosine
    # schedule the two usually coincide, but they do not have to -- and when the
    # run is cut before convergence the last checkpoint is not necessarily best.
    best_metric = float("inf")
    best_step = -1
    model.train()
    step = 0
    tokens_seen = 0
    running: dict[str, float] = {}
    start = time.time()
    bar = tqdm(total=cfg.train.steps, desc=f"stage {stage}", unit="step")

    while step < cfg.train.steps:
        for batch in train_loader:
            if step >= cfg.train.steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["last_index"],
                    batch["supervised_start"] if stage >= 2 else None,
                )
                loss, logs = compute_losses(out, batch, cfg.train, stage)
            (loss / cfg.train.grad_accum).backward()

            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + value / cfg.train.log_every

            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            bar.update(1)

            if step % cfg.train.log_every == 0:
                running["lr"] = scheduler.get_last_lr()[0]
                running["step"] = step
                bar.set_postfix({k.split("/")[-1]: f"{v:.3f}" for k, v in running.items() if k.startswith("loss")})
                for key, value in running.items():
                    if key != "step":
                        writer.add_scalar(f"stage{stage}/{key}", value, step)
                writer.add_scalar(
                    f"stage{stage}/throughput/tokens_per_s",
                    tokens_seen / max(1e-6, time.time() - start),
                    step,
                )
                writer.add_scalar(
                    "gpu/mem_allocated_gb",
                    torch.cuda.max_memory_allocated() / 1e9,
                    step,
                )
                history.append(dict(running))
                running = {}

            if eval_loader is not None and step % cfg.train.eval_every == 0:
                metrics = evaluate(model, eval_loader, cfg, stage)
                metrics["step"] = step
                if metrics["eval/loss"] < best_metric:
                    best_metric = metrics["eval/loss"]
                    best_step = step
                    model.save(out_dir / "best")
                    (out_dir / "best" / "best.json").write_text(
                        json.dumps({"step": step, **metrics}, indent=2)
                    )
                metrics["eval/best_loss"] = best_metric
                for key, value in metrics.items():
                    if key != "step":
                        writer.add_scalar(f"stage{stage}/{key}", value, step)
                writer.flush()
                history.append(metrics)
                tqdm.write(f"[eval @ {step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "step"))

            if step % cfg.train.save_every == 0 or step == cfg.train.steps:
                model.save(out_dir / "checkpoint")
    bar.close()

    if eval_loader is not None:
        final = evaluate(model, eval_loader, cfg, stage, max_batches=200)
        final["step"] = step
        for key, value in final.items():
            if key != "step":
                writer.add_scalar(f"stage{stage}/final/{key}", value, step)
        history.append(final)
        print("[final] " + " ".join(f"{k}={v:.4f}" for k, v in final.items() if k != "step"))

    writer.close()
    model.save(out_dir / "checkpoint")
    if best_step >= 0:
        print(
            f"[train] best eval/loss {best_metric:.4f} at step {best_step} "
            f"(saved in {out_dir / 'best'}); last checkpoint is step {step}"
        )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] stage {stage} finished in {(time.time()-start)/60:.1f} min -> {out_dir/'checkpoint'}")
    return out_dir / "checkpoint"
