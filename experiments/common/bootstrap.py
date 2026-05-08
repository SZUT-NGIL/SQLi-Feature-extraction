from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OURS_DIR = ROOT_DIR / "ours"


def bootstrap_repo_paths() -> Path:
    """补齐导入路径，兼容仓库里旧式的 `from utils...` 导入。"""
    for path in (ROOT_DIR, OURS_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return ROOT_DIR


bootstrap_repo_paths()
