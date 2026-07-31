"""
Unified training entry point for the pruning pipeline.

Modes:
  standard  — FP32 baseline from COCO pretrained weights
  sparsity  — BN-gamma sparsity regularization before pruning
  finetune  — Recover accuracy on a pruned checkpoint (requires maskbndict)

Examples:
  python scripts/train.py --mode standard
  python scripts/train.py --mode sparsity --config configs/pipeline.yaml
  python scripts/train.py --mode finetune --weights weights/pruned_r0.30.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics import YOLO

from _config import default_data_yaml, load_pipeline_config, resolve_path
from _paths import REPO_ROOT


MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "standard": {
        "weights": "weights/yolov8s.pt",
        "epochs": 200,
        "batch": 48,
        "sr": 0.0,
        "finetune": False,
        "project": ".",
        "name": "runs/train-normal",
        "device": 0,
    },
    "sparsity": {
        "weights": "weights/original.pt",
        "epochs": 80,
        "batch": 96,
        "sr": 0.015,
        "lr0": 0.01,
        "patience": 50,
        "finetune": False,
        "project": ".",
        "name": "runs/train-sparsity",
        "device": 0,
    },
    "finetune": {
        "weights": "weights/pruned_r0.30.pt",
        "epochs": 100,
        "batch": 96,
        "sr": 0.0,
        "lr0": 0.001,
        "optimizer": "AdamW",
        "patience": 20,
        "finetune": True,
        "project": ".",
        "name": "runs/train-finetune",
        "device": 0,
    },
}


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _apply_pipeline_config(mode: str, pipeline: dict[str, Any]) -> dict[str, Any]:
    cfg = MODE_DEFAULTS[mode].copy()
    section = pipeline.get(mode, {})
    paths = pipeline.get("paths", {})

    if mode == "standard" and "pretrained" in paths:
        cfg["weights"] = paths["pretrained"]
    elif mode == "sparsity":
        if "original" in paths:
            cfg["weights"] = paths["original"]
    elif mode == "finetune" and "pruned" in paths:
        cfg["weights"] = paths["pruned"]

    for key in ("epochs", "batch", "sr", "lr0", "patience", "optimizer", "device", "project", "name"):
        if key in section:
            cfg[key] = section[key]
    return cfg


def train(mode: str, data: str | Path, overrides: dict[str, Any], pipeline_path: Path | None) -> None:
    cfg = MODE_DEFAULTS[mode].copy()
    if pipeline_path and pipeline_path.exists():
        cfg = _apply_pipeline_config(mode, load_pipeline_config(pipeline_path))
    cfg = _merge(cfg, overrides)

    weights = str(resolve_path(cfg["weights"]))
    train_kwargs: dict[str, Any] = {
        "data": str(resolve_path(data)),
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "sr": cfg["sr"],
        "project": cfg["project"],
        "name": cfg["name"],
        "device": cfg["device"],
        "resume": cfg.get("resume", False),
    }

    for optional in ("lr0", "patience", "optimizer"):
        if optional in cfg:
            train_kwargs[optional] = cfg[optional]
    if cfg.get("finetune"):
        train_kwargs["finetune"] = True

    model = YOLO(weights)
    model.train(**train_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for the pruning + QAT pipeline")
    parser.add_argument("--mode", choices=MODE_DEFAULTS.keys(), default="finetune")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    parser.add_argument("--data", type=str, default=None, help="Dataset yaml (default: configs/data.yaml or data.yaml)")

    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--sr", type=float, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data or str(default_data_yaml())
    overrides = {
        "weights": args.weights,
        "epochs": args.epochs,
        "batch": args.batch,
        "sr": args.sr,
        "lr0": args.lr0,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "resume": args.resume,
    }
    train(args.mode, data, overrides, Path(args.config))


if __name__ == "__main__":
    main()
