"""Canonical filesystem locations. Override REPO via env CRAVE_REPO if the tree moves."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("CRAVE_REPO", "/vePFS/tim/workspace/deepdive_kai0"))
TEMP = REPO / "temp"
DOCS = REPO / "docs"
# Docs that live inside the crave package (committed alongside the code).
CRAVE_DOCS = REPO / "crave/docs"

# Frozen HF feature-encoder weights (shared cache).
HF_HUB = Path("/vePFS/xiezhicong/.cache/huggingface/hub")

# Default output root for generated artifacts (plots / galleries / videos / json).
def out_dir(name: str) -> Path:
    d = TEMP / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# Durable visualization outputs live under crave/docs/ (committed), not temp/.
def viz_dir(name: str = "") -> Path:
    d = CRAVE_DOCS / "visualization"
    if name:
        d = d / name
    d.mkdir(parents=True, exist_ok=True)
    return d
