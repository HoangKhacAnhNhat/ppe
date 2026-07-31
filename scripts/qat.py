"""
QAT (Quantization-Aware Training) for pruned YOLOv8.

Pipeline:
  1. scripts/train.py --mode sparsity
  2. scripts/prune.py
  3. scripts/train.py --mode finetune
  4. scripts/qat.py          (this file)
  5. scripts/export_onnx.py

Example:
  python scripts/qat.py \\
      --pruned-checkpoint weights/pruned_r0.30.pt \\
      --pretrained-weight runs/train-finetune/weights/best.pt \\
      --data-config configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics.qat.nvidia_tensorrt.qat_pruned_trainer import QuantizationPrunedTrainer
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER

from _config import default_data_yaml, load_pipeline_config, resolve_path
from _paths import REPO_ROOT


def train(args: argparse.Namespace):
    LOGGER.info(f"[QAT-Pruned] Pruned checkpoint: {args.pruned_checkpoint}")
    if args.pretrained_weight:
        LOGGER.info(f"[QAT-Pruned] Finetuned weight: {args.pretrained_weight}")
    else:
        LOGGER.info("[QAT-Pruned] Using pruned checkpoint weights (no separate finetuned weight)")

    overrides = {
        "model": args.model_config,
        "data": args.data_config,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": args.cache,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "freeze": args.freeze,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "augment": args.augment,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "patience": args.patience,
        "deterministic": args.deterministic,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": args.plots,
        "recalib_every": args.recalib_every,
        "pruned_checkpoint": args.pruned_checkpoint,
    }

    trainer = QuantizationPrunedTrainer(cfg=DEFAULT_CFG_DICT.copy(), overrides=overrides)
    weight_to_load = args.pretrained_weight or args.pruned_checkpoint
    trainer.model = trainer.get_model(weights=weight_to_load)
    return trainer.train()


def parse_args() -> argparse.Namespace:
    pipeline = load_pipeline_config()
    qat_cfg = pipeline.get("qat", {})
    paths = pipeline.get("paths", {})

    parser = argparse.ArgumentParser(description="QAT training for pruned YOLOv8")
    parser.add_argument(
        "--pruned-checkpoint",
        type=str,
        default=paths.get("pruned"),
        help="Pruned checkpoint from scripts/prune.py (contains maskbndict)",
    )
    parser.add_argument(
        "--pretrained-weight",
        type=str,
        default=None,
        help="Finetuned weight (recommended). Falls back to pruned checkpoint if omitted.",
    )
    parser.add_argument("--model-config", type=str, default="yolov8s.yaml")
    parser.add_argument("--data-config", type=str, default=str(default_data_yaml()))
    parser.add_argument("--epochs", type=int, default=qat_cfg.get("epochs", 10))
    parser.add_argument("--imgsz", type=int, default=qat_cfg.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=qat_cfg.get("batch", 16))
    parser.add_argument("--device", default=qat_cfg.get("device", 0))
    parser.add_argument("--workers", type=int, default=qat_cfg.get("workers", 8))
    parser.add_argument("--cache", type=str, default=qat_cfg.get("cache", "ram"))
    parser.add_argument("--optimizer", type=str, default=qat_cfg.get("optimizer", "AdamW"))
    parser.add_argument("--lr0", type=float, default=qat_cfg.get("lr0", 1e-3))
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.90)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--freeze", type=int, default=qat_cfg.get("freeze", 10))
    parser.add_argument("--box", type=float, default=0.04)
    parser.add_argument("--cls", type=float, default=2.5)
    parser.add_argument("--dfl", type=float, default=1.2)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=qat_cfg.get("patience", 5))
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--project", type=str, default=qat_cfg.get("project", "runs/qat-pruned"))
    parser.add_argument("--name", type=str, default=qat_cfg.get("name", "train"))
    parser.add_argument("--exist-ok", action="store_true", default=True)
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--recalib-every", type=int, default=qat_cfg.get("recalib_every", 1))
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    args = parser.parse_args()

    if not args.pruned_checkpoint:
        parser.error("--pruned-checkpoint is required (or set paths.pruned in pipeline.yaml)")

    if args.pretrained_weight is None and paths.get("finetuned"):
        candidate = resolve_path(paths["finetuned"])
        if candidate.exists():
            args.pretrained_weight = str(candidate)

    args.pruned_checkpoint = str(resolve_path(args.pruned_checkpoint))
    if args.pretrained_weight:
        args.pretrained_weight = str(resolve_path(args.pretrained_weight))
    args.data_config = str(resolve_path(args.data_config))
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
