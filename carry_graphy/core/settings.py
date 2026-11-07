"""Persistent user settings (last used values, paths and window geometry)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

SETTINGS_VERSION = 1
SETTINGS_ENV = "CARRY_GRAPHY_SETTINGS"


def default_settings_path() -> Path:
    override = os.environ.get(SETTINGS_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "carry-graphy" / "settings.json"


class SettingsStore:
    """Small JSON-backed store. Failures never raise; they only disable saving."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_settings_path()
        self.data: dict[str, Any] = {}
        self.last_error: str | None = None

    def load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        except (OSError, ValueError) as exc:
            self.last_error = f"設定ファイルを読み込めませんでした: {exc}"
            raw = {}
        if not isinstance(raw, dict) or raw.get("version") not in (None, SETTINGS_VERSION):
            raw = {}
        self.data = raw
        return self.data

    def save(self, data: dict[str, Any]) -> bool:
        payload = dict(data)
        payload["version"] = SETTINGS_VERSION
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp_path, self.path)
        except OSError as exc:
            self.last_error = f"設定ファイルを保存できませんでした: {exc}"
            return False
        self.data = payload
        return True


def get_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}
