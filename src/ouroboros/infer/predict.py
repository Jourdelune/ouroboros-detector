"""Public prediction API: text in, segments and a document verdict out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import Config, load_config
from ..labels import HUMANIZER_NAMES
from ..modeling.model import DTYPES, OuroborosModel, load_tokenizer
from .calibration import Calibrator
from .postprocess import DocumentPrediction, postprocess
from .windows import aggregate, infer_document


def load_checkpoint(run_dir: str | Path, device: str = "cuda") -> tuple[OuroborosModel, object, Config]:
    """Rebuild a trained model from a run directory."""
    from peft import PeftModel
    from transformers import AutoModel

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    ckpt = run_dir / "checkpoint"

    base = AutoModel.from_pretrained(
        cfg.backbone.name,
        dtype=DTYPES[cfg.backbone.dtype],
        attn_implementation=cfg.backbone.attn_implementation,
        trust_remote_code=cfg.backbone.trust_remote_code,
    )
    backbone = PeftModel.from_pretrained(base, ckpt / "backbone")
    hidden = base.config.hidden_size
    model = OuroborosModel(backbone, hidden)
    model.load_heads(ckpt, device="cpu")
    model.heads.to(DTYPES[cfg.backbone.dtype])
    model = model.to(device).eval()

    tokenizer = load_tokenizer(cfg.backbone)
    return model, tokenizer, cfg


def load_pretrained(path_or_repo: str | Path, device: str = "cuda") -> tuple[OuroborosModel, object, Config, Path]:
    """Load the released weights: a Hugging Face repo id or a local directory holding

    ``model.safetensors`` + ``config.json`` (the merged backbone), ``heads.pt``, ``ouroboros_config.yaml``,
    the tokenizer files and ``calibrator.json``.
    """
    from transformers import AutoModel

    path = Path(path_or_repo)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(path_or_repo)))
    cfg = load_config(path / "ouroboros_config.yaml")
    cfg.backbone.name = str(path)
    backbone = AutoModel.from_pretrained(
        path, dtype=DTYPES[cfg.backbone.dtype], attn_implementation=cfg.backbone.attn_implementation
    )
    model = OuroborosModel(backbone, backbone.config.hidden_size)
    model.load_heads(path, device="cpu")
    model.heads.to(DTYPES[cfg.backbone.dtype])
    model = model.to(device).eval()
    return model, load_tokenizer(cfg.backbone), cfg, path


@dataclass
class Predictor:
    model: OuroborosModel
    tokenizer: object
    cfg: Config
    calibrator: Calibrator
    device: str = "cuda"

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, device: str = "cuda") -> "Predictor":
        """``Predictor.from_pretrained("<user>/<repo>")`` or a local directory with the released files."""
        model, tokenizer, cfg, path = load_pretrained(path_or_repo, device)
        calibrator = Calibrator.load(path / "calibrator.json")
        return cls(model, tokenizer, cfg, calibrator, device)

    @classmethod
    def from_run(cls, run_dir: str | Path, device: str = "cuda") -> "Predictor":
        if (Path(run_dir) / "ouroboros_config.yaml").exists() or not Path(run_dir).exists():
            return cls.from_pretrained(run_dir, device)  # a released model directory or a Hugging Face repo id
        model, tokenizer, cfg = load_checkpoint(run_dir, device)
        calib_path = Path(run_dir) / "calibrator.json"
        if calib_path.exists():
            calibrator = Calibrator.load(calib_path)
        else:
            calibrator = Calibrator(
                identity=True,
                smoothness_lambda=cfg.postprocess.smoothness_lambda,
                mixed_gamma=cfg.postprocess.mixed_gamma,
                deltas=list(cfg.postprocess.class_prior_delta),
            )
        return cls(model, tokenizer, cfg, calibrator, device)

    def observe(self, text: str):
        obs = infer_document(self.model, self.tokenizer, text, self.cfg.data, self.device)
        return aggregate(obs), obs

    def predict(self, text: str) -> DocumentPrediction:
        agg, _ = self.observe(text)
        return postprocess(agg, self.calibrator, self.cfg.postprocess)

    @torch.no_grad()
    def humanizer(self, text: str) -> dict[str, float]:
        """Document-level humanization probabilities (report Section 3.4)."""
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.cfg.data.window_tokens,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        doubled = torch.cat([ids, ids]).unsqueeze(0).to(self.device)
        attention = torch.ones_like(doubled)
        last = torch.tensor([doubled.shape[1] - 1], device=self.device)
        start = torch.tensor([len(ids)], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(doubled, attention, last, start)
        probs = torch.softmax(out.humanizer_logits.float(), -1)[0].cpu().numpy()
        return dict(zip(HUMANIZER_NAMES, probs.tolist()))
