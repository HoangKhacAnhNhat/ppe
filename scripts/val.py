"""
Evaluate model accuracy and benchmark inference latency.

Example:
  python scripts/val.py --weight weights/pruned_r0.30.pt --data configs/data.yaml
  python scripts/val.py --weight runs/qat-pruned/train/weights/best.engine --benchmark-runs 100
"""

from __future__ import annotations

import argparse
import time

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from ultralytics import YOLO

from _config import default_data_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO model and benchmark latency")
    parser.add_argument("--weight", type=str, required=True, help="Path to .pt or .engine checkpoint")
    parser.add_argument("--data", type=str, default=str(default_data_yaml()))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    parser.add_argument(
        "--skip-fuse",
        action="store_true",
        help="Disable BN+Conv fuse for QAT .pt checkpoints (preserves fake-quant pattern)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight = str(resolve_path(args.weight))
    data = str(resolve_path(args.data))

    model = YOLO(weight)
    if args.skip_fuse and weight.endswith(".pt"):
        model.model.fuse = lambda verbose=True: model.model

    device = args.device
    if device != "cpu" and torch.cuda.is_available():
        model.model.to(device)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, device=device if device != "cpu" else "cpu")
    with torch.no_grad():
        _ = model.model(dummy)

    latencies = []
    for _ in range(args.benchmark_runs):
        img = torch.randn(1, 3, args.imgsz, args.imgsz, device=dummy.device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.model(img)
        latencies.append(time.perf_counter() - start)

    avg_latency = float(np.mean(latencies))
    print(f"Average Latency per image: {avg_latency * 1000:.2f} ms")
    print(f"Approx FPS: {1 / avg_latency:.2f}")

    metrics = model.val(data=data, batch=args.batch, device=device, split=args.split, imgsz=args.imgsz)
    print("=== Evaluation Metrics ===")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
