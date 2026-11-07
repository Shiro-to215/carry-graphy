"""Shared filesystem locations."""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
ENTRY_SCRIPT = APP_DIR / "integrated_app.py"
DEFAULT_RESULTS_DIR = APP_DIR / "統合結果"
