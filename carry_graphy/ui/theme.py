"""Design tokens: colours, typography and spacing.

Design goals (from user feedback):
* White-based surfaces with a single blue accent.
* Readable text: dark, high-contrast colours and larger font sizes.
* Big click targets.
* Clear hierarchy: exactly one filled (primary) button per screen.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass

import customtkinter as ctk

# ---- Colours -------------------------------------------------------------
BG = "#F3F5F9"              # window background (slightly grey so white cards stand out)
SURFACE = "#FFFFFF"         # cards, sidebar, action bar
SURFACE_ALT = "#F8FAFC"     # inset areas inside cards
BORDER = "#E2E7EF"
BORDER_STRONG = "#C9D1DD"

TEXT = "#111827"            # primary text
TEXT_SECONDARY = "#374151"  # body copy / labels
TEXT_MUTED = "#5B6472"      # hints (still >= 5:1 contrast on white)
TEXT_ON_ACCENT = "#FFFFFF"

ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
ACCENT_SOFT = "#EEF4FF"
ACCENT_SOFT_HOVER = "#DCE7FE"
ACCENT_TEXT = "#1E40AF"

SUCCESS = "#15803D"
SUCCESS_SOFT = "#EAF7EF"
SUCCESS_BORDER = "#A7DDB8"
WARNING = "#A15C07"
WARNING_SOFT = "#FFF6E5"
WARNING_BORDER = "#F5CF8A"
DANGER = "#B91C1C"
DANGER_HOVER = "#991B1B"
DANGER_SOFT = "#FDEEEE"
DANGER_BORDER = "#F4B4B4"

DISABLED_BG = "#E5E8EE"
DISABLED_TEXT = "#9AA3B2"
HOVER_NEUTRAL = "#F1F4F8"

CANVAS_BG = "#E7EBF1"

TONES = {
    "info": (ACCENT_SOFT, "#BFD3FB", ACCENT_TEXT),
    "success": (SUCCESS_SOFT, SUCCESS_BORDER, SUCCESS),
    "warning": (WARNING_SOFT, WARNING_BORDER, WARNING),
    "danger": (DANGER_SOFT, DANGER_BORDER, DANGER),
    "neutral": (SURFACE_ALT, BORDER, TEXT_SECONDARY),
}

# ---- Geometry ------------------------------------------------------------
RADIUS_CARD = 14
RADIUS_CONTROL = 10
PAD_PAGE_X = 36
PAD_CARD = 22
GAP = 16
CONTENT_MAX_WIDTH = 980
SIDEBAR_WIDTH = 252

BUTTON_HEIGHT = 46
BUTTON_HEIGHT_SM = 38
ENTRY_HEIGHT = 42

# ---- Typography ----------------------------------------------------------
_UI_FAMILIES = ("Yu Gothic UI", "Meiryo UI", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP")
_MONO_FAMILIES = ("Cascadia Mono", "Consolas", "BIZ UDGothic", "Menlo", "DejaVu Sans Mono")


@dataclass
class Fonts:
    family: str
    mono_family: str
    display: ctk.CTkFont      # page title
    title: ctk.CTkFont        # card title
    subtitle: ctk.CTkFont     # emphasised body
    body: ctk.CTkFont
    body_bold: ctk.CTkFont
    label: ctk.CTkFont        # form labels
    caption: ctk.CTkFont
    caption_bold: ctk.CTkFont
    button: ctk.CTkFont
    button_sm: ctk.CTkFont
    mono: ctk.CTkFont
    step_number: ctk.CTkFont
    hero: ctk.CTkFont

    def canvas(self, size: int, weight: str = "normal") -> tuple[str, int, str]:
        """Font tuple for plain tk widgets (size in points)."""
        return (self.family, size, weight)


_fonts: Fonts | None = None


def _pick_family(candidates: tuple[str, ...], root: tk.Misc) -> str:
    available = set(tkfont.families(root))
    for name in candidates:
        if name in available:
            return name
    return tkfont.nametofont("TkDefaultFont", root).actual("family")


def init_fonts(root: tk.Misc) -> Fonts:
    global _fonts
    family = _pick_family(_UI_FAMILIES, root)
    mono = _pick_family(_MONO_FAMILIES, root)
    _fonts = Fonts(
        family=family,
        mono_family=mono,
        display=ctk.CTkFont(family, 26, "bold"),
        title=ctk.CTkFont(family, 18, "bold"),
        subtitle=ctk.CTkFont(family, 16, "bold"),
        body=ctk.CTkFont(family, 15),
        body_bold=ctk.CTkFont(family, 15, "bold"),
        label=ctk.CTkFont(family, 14, "bold"),
        caption=ctk.CTkFont(family, 13),
        caption_bold=ctk.CTkFont(family, 13, "bold"),
        button=ctk.CTkFont(family, 16, "bold"),
        button_sm=ctk.CTkFont(family, 14, "bold"),
        mono=ctk.CTkFont(mono, 14),
        step_number=ctk.CTkFont(family, 14, "bold"),
        hero=ctk.CTkFont(family, 30, "bold"),
    )
    return _fonts


def fonts() -> Fonts:
    if _fonts is None:
        raise RuntimeError("init_fonts() must be called after the root window is created")
    return _fonts


def apply_global_theme() -> None:
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
