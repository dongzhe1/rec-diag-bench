"""Bootstraps the Python path to include the src/ directory.

Every script in this project imports this module first so that
``from coldtail import ...`` works regardless of the working directory.

CPU-only — pure path manipulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
