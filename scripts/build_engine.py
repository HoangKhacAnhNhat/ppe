"""
Build TensorRT engine from QAT ONNX (no trtexec binary required).

Example:
  python scripts/build_engine.py \\
      --onnx runs/qat-pruned/train/weights/best.tensorrt.onnx \\
      --engine runs/qat-pruned/train/weights/best.engine \\
      --data-yaml configs/data.yaml
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tensorrt as trt
import yaml

from _config import default_data_yaml, load_pipeline_config, resolve_path
from _paths import REPO_ROOT


def build_engine(
    onnx_path,
    engine_path,
    workspace_gb=2,
    fp16=True,
    int8=True,
    imgsz=640,
    batch=1,
    names=None,
    task="detect",
    stride=32,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ONNX parse error] {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("[build] INT8 enabled (scales from QAT QDQ nodes)")

    print(f"[build] TensorRT {trt.__version__} — building engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    data = bytes(serialized)
    meta = {
        "description": "YOLOv8 pruned + QAT INT8",
        "author": "ultralytics",
        "task": task,
        "batch": batch,
        "imgsz": [imgsz, imgsz],
        "names": names if names is not None else {i: str(i) for i in range(80)},
        "stride": stride,
        "args": {"batch": batch, "imgsz": imgsz},
    }
    meta_bytes = json.dumps(meta).encode()

    with open(engine_path, "wb") as f:
        f.write(len(meta_bytes).to_bytes(4, byteorder="little", signed=True))
        f.write(meta_bytes)
        f.write(data)
    print(
        f"[build] Saved engine → {engine_path} "
        f"({(len(data) + len(meta_bytes) + 4) / 1e6:.1f} MB, {len(meta_bytes)} B metadata)"
    )


def parse_args() -> argparse.Namespace:
    pipeline = load_pipeline_config()
    paths = pipeline.get("paths", {})
    engine_cfg = pipeline.get("engine", {})

    parser = argparse.ArgumentParser(description="Build TensorRT engine from QAT ONNX")
    parser.add_argument("--onnx", type=str, default=paths.get("qat_onnx"))
    parser.add_argument("--engine", type=str, default=paths.get("engine"))
    parser.add_argument("--workspace-gb", type=int, default=engine_cfg.get("workspace_gb", 2))
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--imgsz", type=int, default=engine_cfg.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=engine_cfg.get("batch", 1))
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--names", type=str, default=None)
    parser.add_argument("--data-yaml", type=str, default=str(default_data_yaml()))
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    args = parser.parse_args()

    if not args.onnx or not args.engine:
        parser.error("--onnx and --engine are required")

    args.onnx = str(resolve_path(args.onnx))
    args.engine = str(resolve_path(args.engine))
    if args.data_yaml:
        args.data_yaml = str(resolve_path(args.data_yaml))
    return args


def main() -> None:
    args = parse_args()

    if args.data_yaml:
        with open(args.data_yaml, encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        raw_names = data_cfg["names"]
        if isinstance(raw_names, dict):
            names = {int(k): v for k, v in raw_names.items()}
        else:
            names = {i: n for i, n in enumerate(raw_names)}
        print(f"[build] Loaded {len(names)} class names from {args.data_yaml}: {names}")
    elif args.names:
        names = {int(k): v for k, v in json.loads(args.names).items()}
    else:
        names = {i: str(i) for i in range(args.nc)}

    build_engine(
        args.onnx,
        args.engine,
        args.workspace_gb,
        fp16=not args.no_fp16,
        int8=not args.no_int8,
        imgsz=args.imgsz,
        batch=args.batch,
        names=names,
        task=args.task,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
