"""
BN-gamma structured pruning (Network Slimming) for YOLOv8.

Output: weights/pruned_r{ratio}.pt containing {'model', 'maskbndict'}.

Example:
  python scripts/prune.py \\
      --weights runs/train-sparsity/weights/last.pt \\
      --prune-ratio 0.30
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import yaml

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Concat, Conv
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.block_pruned import C2fPruned, SPPFPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import colorstr

from _config import default_data_yaml, load_pipeline_config, pruned_checkpoint_name, resolve_path
from _paths import REPO_ROOT

warnings.filterwarnings("ignore")


def main(opt: argparse.Namespace) -> None:
    weights = opt.weights[0] if isinstance(opt.weights, list) else opt.weights
    prune_ratio = opt.prune_ratio
    cfg = opt.cfg
    model_size = opt.model_size
    save_dir = Path(opt.save_dir)

    model = AutoBackend(weights, fuse=False)
    model.eval()

    bn_dict: dict[str, nn.BatchNorm2d] = {}
    ignore_bn_list: list[str] = []
    chunk_bn_list: list[str] = []
    for name, module in model.model.named_modules():
        if isinstance(module, Bottleneck):
            if module.add:
                ignore_bn_list.append(f"{name[:-4]}.cv1.bn")
                ignore_bn_list.append(f"{name}.cv2.bn")
            else:
                chunk_bn_list.append(f"{name[:-4]}.cv1.bn")
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module

    for ignore_bn_name in ignore_bn_list:
        assert ignore_bn_name in bn_dict

    bn_dict = {k: v for k, v in bn_dict.items() if k not in ignore_bn_list}
    bn_weights: list[float] = []
    for module in bn_dict.values():
        bn_weights.extend(module.weight.data.abs().clone().cpu().tolist())

    sorted_bn = torch.sort(torch.tensor(bn_weights))[0]
    highest_thre = min(module.weight.data.abs().clone().cpu().max() for module in bn_dict.values())
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(sorted_bn)
    thre = sorted_bn[int(len(sorted_bn) * prune_ratio)]
    print(
        f"Pruning gamma values should be less than {colorstr(f'{highest_thre:.4f}')}, "
        f"yours is {colorstr(f'{thre:.4f}')}"
    )
    print(
        f"The corresponding pruning ratio should be less than {colorstr(f'{percent_limit:.3f}')}, "
        f"yours is {colorstr(f'{prune_ratio:.3f}')}"
    )
    if prune_ratio > percent_limit:
        prune_ratio = percent_limit
        print(f"Pruning ratio falling down to {colorstr(f'{prune_ratio:.3f}')}")

    with open(cfg, encoding="ascii", errors="ignore") as f:
        model_yamls = yaml.safe_load(f)

    pruned_yaml = {
        "nc": model.model.nc,
        "scales": model_yamls["scales"],
        "scale": model_size,
        "backbone": [
            [-1, 1, Conv, [64, 3, 2]],
            [-1, 1, Conv, [128, 3, 2]],
            [-1, 3, C2fPruned, [128, True]],
            [-1, 1, Conv, [256, 3, 2]],
            [-1, 6, C2fPruned, [256, True]],
            [-1, 1, Conv, [512, 3, 2]],
            [-1, 6, C2fPruned, [512, True]],
            [-1, 1, Conv, [1024, 3, 2]],
            [-1, 3, C2fPruned, [1024, True]],
            [-1, 1, SPPFPruned, [1024, 5]],
        ],
        "head": [
            [-1, 1, nn.Upsample, [None, 2, "nearest"]],
            [[-1, 6], 1, Concat, [1]],
            [-1, 3, C2fPruned, [512]],
            [-1, 1, nn.Upsample, [None, 2, "nearest"]],
            [[-1, 4], 1, Concat, [1]],
            [-1, 3, C2fPruned, [256]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, 12], 1, Concat, [1]],
            [-1, 3, C2fPruned, [512]],
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, 9], 1, Concat, [1]],
            [-1, 3, C2fPruned, [1024]],
            [[15, 18, 21], 1, DetectPruned, [model.model.nc]],
        ],
    }

    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    maskbndict: dict[str, torch.Tensor] = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            origin_channels = module.weight.data.size()[0]
            remaining_channels = origin_channels
            mask = torch.ones(origin_channels)
            if name not in ignore_bn_list:
                mask = module.weight.data.abs().gt(thre).float()
                if name in chunk_bn_list and mask.sum() % 2 == 1:
                    flattened_sorted_weight = torch.sort(module.weight.data.abs().view(-1))[0]
                    idx = torch.min(torch.nonzero(flattened_sorted_weight.gt(thre))).item()
                    thre_ = flattened_sorted_weight[idx - 1] - 1e-6
                    mask = module.weight.data.abs().gt(thre_).float()
                assert mask.sum() > 0, f"bn {name} has no active elements"
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                remaining_channels = mask.sum().int()
            maskbndict[name] = mask
            print(
                f"|\t{name:<25}{'|':<10}{origin_channels:<20}{'|':<10}{remaining_channels:<20}|"
            )
    print("=" * 94)

    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model.eval()

    current_to_prev = pruned_model.current_to_prev
    changed: list[str] = []
    pattern_c2f = re.compile(r"model.\d+.m.0.cv1.bn")
    pattern_detect = re.compile(r"model.\d+.cv\d.\d.2")

    for (name_org, module_org), (name_pruned, module_pruned) in zip(
        model.model.named_modules(), pruned_model.named_modules()
    ):
        assert name_org == name_pruned
        if "dfl" in name_org:
            break

        if pattern_detect.fullmatch(name_org) is not None:
            prev_bn_layer_name = current_to_prev[name_org]
            in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[:, in_channels_mask, :, :]
            if module_org.bias is not None:
                module_pruned.bias.data = module_org.bias.data
            continue

        if isinstance(module_org, nn.Conv2d):
            currnet_bn_layer_name = name_org[:-4] + "bn"
            out_channels_mask = maskbndict[currnet_bn_layer_name].to(torch.bool)
            prev_bn_layer_name = current_to_prev.get(currnet_bn_layer_name, None)
            if isinstance(prev_bn_layer_name, list):
                in_channels_mask = torch.cat([maskbndict[ni] for ni in prev_bn_layer_name], dim=0).to(torch.bool)
            elif prev_bn_layer_name is not None:
                in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
                if pattern_c2f.fullmatch(currnet_bn_layer_name) is not None:
                    in_channels_mask = in_channels_mask.chunk(2, 0)[1]
                if name_org == "model.9.cv2.conv":
                    in_channels_mask = torch.cat([in_channels_mask for _ in range(4)], dim=0)
            else:
                in_channels_mask = torch.ones(module_org.weight.data.shape[1], dtype=torch.bool)

            state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
            state_dict_org = state_dict_org[:, in_channels_mask, :, :]
            module_pruned.weight.data = state_dict_org
            if module_org.bias is not None:
                module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            changed.append(currnet_bn_layer_name)

        if isinstance(module_org, nn.BatchNorm2d):
            out_channels_mask = maskbndict[name_org].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[out_channels_mask]
            module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            module_pruned.running_mean = module_org.running_mean[out_channels_mask]
            module_pruned.running_var = module_org.running_var[out_channels_mask]

    missing = [name for name in maskbndict if name not in changed]
    assert not missing, f"missing: {missing}"

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / pruned_checkpoint_name(prune_ratio)
    torch.save({"model": pruned_model, "maskbndict": maskbndict}, save_path)
    print(f"Saved pruned checkpoint → {save_path}")

    loaded = torch.load(save_path, map_location="cpu", weights_only=False)["model"].cuda()
    loaded(torch.randn(1, 3, 640, 640, device="cuda"))


def parse_opt() -> argparse.Namespace:
    pipeline = load_pipeline_config()
    prune_cfg = pipeline.get("prune", {})
    paths = pipeline.get("paths", {})

    parser = argparse.ArgumentParser(description="BN-gamma structured pruning for YOLOv8")
    parser.add_argument("--data", type=str, default=str(default_data_yaml()))
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=[paths.get("sparsity_ckpt", "runs/train-sparsity/weights/last.pt")],
    )
    parser.add_argument("--cfg", type=str, default=prune_cfg.get("cfg", "ultralytics/cfg/models/v8/yolov8.yaml"))
    parser.add_argument("--model-size", type=str, default=prune_cfg.get("model_size", "s"))
    parser.add_argument("--prune-ratio", type=float, default=prune_cfg.get("ratio", 0.30))
    parser.add_argument("--save-dir", type=str, default=prune_cfg.get("save_dir", "weights"))
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    opt = parser.parse_args()
    opt.weights = [str(resolve_path(w)) for w in opt.weights]
    opt.cfg = str(resolve_path(opt.cfg))
    opt.save_dir = str(resolve_path(opt.save_dir))
    return opt


if __name__ == "__main__":
    main(parse_opt())
