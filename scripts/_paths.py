"""Repository path helpers for pipeline scripts."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ASSETS_DIR = REPO_ROOT / "assets"
WEIGHTS_DIR = REPO_ROOT / "weights"
RUNS_DIR = REPO_ROOT / "runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
