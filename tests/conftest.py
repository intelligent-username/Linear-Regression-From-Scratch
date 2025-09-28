"""Test configuration.

Adds project root to sys.path so tests can import `naivelinear` without requiring an explicit editable install.
If the package is installed (e.g. `pip install -e .`), this has no negative effect.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
