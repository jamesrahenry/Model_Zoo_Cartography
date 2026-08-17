"""Shared pytest setup.

Puts census/, null_baseline/, and train/ on sys.path so tests can import the
repo's modules the same way the scripts themselves do (see e.g.
census/eigenspace_overlap.py's own sys.path.insert calls) — there is no
package structure (no __init__.py, no pyproject.toml) to import through
otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for sub in ("census", "null_baseline", "train"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
