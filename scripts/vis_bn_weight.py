"""
Visualize BN gamma distribution shift after sparsity training.

Example:
  python scripts/vis_bn_weight.py \\
      --normal weights/original.pt \\
      --sparse runs/train-sparsity/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from _config import load_pipeline_config, resolve_path
from _paths import ASSETS_DIR, REPO_ROOT


def extract_bn_gamma(model) -> np.ndarray:
    gamma_weights = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            gamma_weights.append(module.weight.detach().cpu().numpy())
    return np.concatenate(gamma_weights)


def plot_bn_gamma_hist(gamma_normal_vals, gamma_sparse_vals, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(gamma_normal_vals, bins=200, density=True, alpha=0.6, color="black", label="Normal Training")
    plt.hist(gamma_sparse_vals, bins=200, density=True, alpha=0.6, color="crimson", label="Sparsity Training")
    plt.xlim(0, 6)
    plt.xlabel("BN Gamma")
    plt.ylabel("Density")
    plt.title("BN Gamma Distribution Shift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hist_and_heatmap(gamma_normal_vals, gamma_sparse_vals, save_path: Path) -> None:
    bins = 200
    x_min, x_max = -6, 6
    hist_normal, _ = np.histogram(gamma_normal_vals, bins=bins, range=(x_min, x_max), density=True)
    hist_sparse, _ = np.histogram(gamma_sparse_vals, bins=bins, range=(x_min, x_max), density=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    axes[0].imshow(hist_normal[np.newaxis, :], aspect="auto", cmap="hot", extent=[x_min, x_max, 0, 1])
    axes[0].set_title("Normal Training")
    axes[0].set_yticks([])
    axes[0].set_xlabel("BN Gamma")

    im2 = axes[1].imshow(hist_sparse[np.newaxis, :], aspect="auto", cmap="hot", extent=[x_min, x_max, 0, 1])
    axes[1].set_title("Sparsity Training")
    axes[1].set_yticks([])
    axes[1].set_xlabel("BN Gamma")
    fig.colorbar(im2, ax=axes, label="Density")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    pipeline = load_pipeline_config()
    paths = pipeline.get("paths", {})

    parser = argparse.ArgumentParser(description="Visualize BN gamma distribution")
    parser.add_argument("--normal", type=str, default=paths.get("original", "weights/original.pt"))
    parser.add_argument("--sparse", type=str, default=paths.get("sparsity_ckpt", "runs/train-sparsity/weights/last.pt"))
    parser.add_argument("--output-dir", type=str, default=str(ASSETS_DIR))
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    args = parser.parse_args()
    args.normal = str(resolve_path(args.normal))
    args.sparse = str(resolve_path(args.sparse))
    args.output_dir = str(resolve_path(args.output_dir))
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    gamma_normal = extract_bn_gamma(YOLO(args.normal).model)
    gamma_sparse = extract_bn_gamma(YOLO(args.sparse).model)

    plot_bn_gamma_hist(gamma_normal, gamma_sparse, output_dir / "bn_gamma_hist.jpg")
    plot_hist_and_heatmap(gamma_normal, gamma_sparse, output_dir / "bn_gamma_hist_heatmap.jpg")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
