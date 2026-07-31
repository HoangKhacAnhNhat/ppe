"""Load and resolve paths from configs/pipeline.yaml."""

from pathlib import Path
from typing import Any

import yaml

from _paths import CONFIGS_DIR, REPO_ROOT


def resolve_path(value: str | Path) -> Path:
    """Resolve a repo-relative path string to an absolute Path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_pipeline_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load pipeline.yaml and resolve all paths under the 'paths' key."""
    config_path = config_path or CONFIGS_DIR / "pipeline.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "paths" in cfg:
        cfg["paths"] = {k: str(resolve_path(v)) for k, v in cfg["paths"].items()}
    return cfg


def default_data_yaml() -> Path:
    """Prefer configs/data.yaml, fall back to repo-root data.yaml."""
    for candidate in (CONFIGS_DIR / "data.yaml", REPO_ROOT / "data.yaml"):
        if candidate.exists():
            return candidate
    return CONFIGS_DIR / "data.yaml"


def pruned_checkpoint_name(ratio: float) -> str:
    """Standard pruned checkpoint filename, e.g. pruned_r0.30.pt."""
    return f"pruned_r{ratio:.2f}.pt"
