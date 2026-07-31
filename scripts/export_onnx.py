"""
Export pruned + QAT model to TensorRT-compatible ONNX.

Example:
  python scripts/export_onnx.py \\
      --pruned-checkpoint weights/pruned_r0.30.pt \\
      --weight runs/qat-pruned/train/weights/best.pt
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics.qat.nvidia_tensorrt.qat_pruned_exporter import export_onnx_pruned

from _config import load_pipeline_config, resolve_path
from _paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    pipeline = load_pipeline_config()
    paths = pipeline.get("paths", {})

    parser = argparse.ArgumentParser(description="Export pruned + QAT YOLOv8 to TensorRT ONNX")
    parser.add_argument(
        "--pruned-checkpoint",
        type=str,
        default=paths.get("pruned"),
        help="Pruned checkpoint from scripts/prune.py",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=paths.get("qat"),
        help="QAT-trained weight (best.pt)",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    args = parser.parse_args()

    if not args.pruned_checkpoint or not args.weight:
        parser.error("--pruned-checkpoint and --weight are required")

    args.pruned_checkpoint = str(resolve_path(args.pruned_checkpoint))
    args.weight = str(resolve_path(args.weight))
    return args


def main() -> None:
    args = parse_args()
    export_onnx_pruned(pruned_checkpoint=args.pruned_checkpoint, qat_weight=args.weight)


if __name__ == "__main__":
    main()
