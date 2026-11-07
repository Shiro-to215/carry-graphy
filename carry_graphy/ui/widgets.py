"""Reusable themed widgets built on CustomTkinter."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Iterable, Literal

import customtkinter as ctk

from . import theme as t

ButtonVariant = Literal["primary", "secondary", "ghost", "danger", "soft"]


def bind_tree(widget: tk.Misc, sequence: str, callback: Callable[[tk.Event], object]) -> None:
    """Bind an event on a widget and all of its descendants."""
    try:
        tk.Misc.bind(widget, sequence, callback, add="+")
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        bind_tree(child, sequence, callback)


def set_tree_cursor(widget: tk.Misc, cursor: str) -> None:
    try:
        tk.Misc.configure(widget, cursor=cursor)  # type: ignore[call-arg]
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        set_tree_cursor(child, cursor)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

class WrapLabel(ctk.CTkLabel):
    """A left-aligned label whose wrap length follows its allocated width."""

    def __init__(self, master: tk.Misc, text: str = "", *, font: ctk.CTkFont | None = None,
                 text_color: str = t.TEXT_SECONDARY, wrap: bool = True, **kwargs) -> None:
        kwargs.setdefault("anchor", "w")
        kwargs.setdefault("justify", "left")
        kwargs.setdefault("height", 0)
        super().__init__(
            master, text=text, font=font or t.fonts().body, text_color=text_color,
            fg_color="transparent", wraplength=560 if wrap else 0, **kwargs,
        )
        self._wrap = wrap
        self._last_wrap = 0
        if wrap:
            tk.Misc.bind(self, "<Configure>", self._on_configure, add="+")

    def _on_configure(self, event: tk.Event) -> None:
        scaling = self._get_widget_scaling()
        width = max(80, int(event.width / scaling) - 4)
        if abs(width - self._last_wrap) > 3:
            self._last_wrap = width
            self.configure(wraplength=width)


def label(master: tk.Misc, text: str = "", kind: str = "body", color: str | None = None,
          wrap: bool = True, **kwargs) -> WrapLabel:
    f = t.fonts()
    font = {
        "display": f.display, "title": f.title, "subtitle": f.subtitle, "body": f.body,
        "body_bold": f.body_bold, "label": f.label, "caption": f.caption,
        "caption_bold": f.caption_bold, "mono": f.mono,
    }[kind]
    default_color = {
        "display": t.TEXT, "title": t.TEXT, "subtitle": t.TEXT, "label": t.TEXT,
        "body_bold": t.TEXT, "caption": t.TEXT_MUTED, "caption_bold": t.TEXT_SECONDARY,
    }.get(kind, t.TEXT_SECONDARY)
    return WrapLabel(master, text, font=font, text_color=color or default_color, wrap=wrap, **kwargs)


def tag(master: tk.Misc, text: str, tone: str = "info") -> ctk.CTkLabel:
    bg, _, fg = t.TONES[tone]
    return ctk.CTkLabel(
        master, text=f" {text} ", font=t.fonts().caption_bold, text_color=fg,
        fg_color=bg, corner_radius=6, height=24,
    )


# --------------------------------------------------------------------------
# Buttons
# --------------------------------------------------------------------------

def button(master: tk.Misc, text: str, command: Callable[[], object] | None = None, *,
           variant: ButtonVariant = "secondary", size: Literal["lg", "md", "sm"] = "md",
           width: int | None = None) -> ctk.CTkButton:
    f = t.fonts()
    height = {"lg": t.BUTTON_HEIGHT + 4, "md": t.BUTTON_HEIGHT, "sm": t.BUTTON_HEIGHT_SM}[size]
    font = f.button if size != "sm" else f.button_sm
    styles = {
        "primary": dict(fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER, text_color=t.TEXT_ON_ACCENT,
                        border_width=0, text_color_disabled=t.DISABLED_TEXT),
        "secondary": dict(fg_color=t.SURFACE, hover_color=t.HOVER_NEUTRAL, text_color=t.TEXT,
                          border_width=1, border_color=t.BORDER_STRONG, text_color_disabled=t.DISABLED_TEXT),
        "soft": dict(fg_color=t.ACCENT_SOFT, hover_color=t.ACCENT_SOFT_HOVER, text_color=t.ACCENT_TEXT,
                     border_width=0, text_color_disabled=t.DISABLED_TEXT),
        "ghost": dict(fg_color="transparent", hover_color=t.HOVER_NEUTRAL, text_color=t.ACCENT_TEXT,
                      border_width=0, text_color_disabled=t.DISABLED_TEXT),
        "danger": dict(fg_color=t.SURFACE, hover_color=t.DANGER_SOFT, text_color=t.DANGER,
                       border_width=1, border_color=t.DANGER_BORDER, text_color_disabled=t.DISABLED_TEXT),
    }[variant]
    min_width = {"lg": 200, "md": 120, "sm": 88}[size]
    widget = ctk.CTkButton(
        master, text=text, command=command, height=height, width=width or min_width,
        corner_radius=t.RADIUS_CONTROL, font=font, **styles,
    )
    widget._carry_variant = variant  # type: ignore[attr-defined]
    return widget


def set_button_enabled(widget: ctk.CTkButton, enabled: bool) -> None:
    variant = getattr(widget, "_carry_variant", "secondary")
    widget.configure(state="normal" if enabled else "disabled")
    if variant == "primary":
        widget.configure(fg_color=t.ACCENT if enabled else t.DISABLED_BG)


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------

class Card(ctk.CTkFrame):
    """White rounded surface with an optional numbered header."""

    def __init__(self, master: tk.Misc, title: str | None = None, description: str | None = None, *,
                 number: int | None = None, badge: tuple[str, str] | None = None,
                 padding: int = t.PAD_CARD, **kwargs) -> None:
        super().__init__(
            master, fg_color=t.SURFACE, corner_radius=t.RADIUS_CARD,
            border_width=1, border_color=t.BORDER, **kwargs,
        )
        self.columnconfigure(0, weight=1)
        row = 0
        if title:
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.grid(row=row, column=0, sticky="ew", padx=padding, pady=(padding - 2, 0))
            header.columnconfigure(2, weight=1)
            column = 0
            if number is not None:
                ctk.CTkLabel(
                    header, text=str(number), width=28, height=28, corner_radius=14,
                    fg_color=t.ACCENT_SOFT, text_color=t.ACCENT_TEXT, font=t.fonts().step_number,
                ).grid(row=0, column=column, padx=(0, 10))
                column += 1
            ctk.CTkLabel(
                header, text=title, font=t.fonts().title, text_color=t.TEXT, anchor="w",
            ).grid(row=0, column=column, sticky="w")
            if badge:
                tag(header, badge[0], badge[1]).grid(row=0, column=column + 1, sticky="w", padx=(10, 0))
            row += 1
            if description:
                label(self, description, "body", color=t.TEXT_MUTED).grid(
                    row=row, column=0, sticky="ew", padx=padding, pady=(4, 0)
                )
                row += 1
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=row, column=0, sticky="nsew", padx=padding,
                       pady=(14 if title else padding, padding))
        self.rowconfigure(row, weight=1)


class Callout(ctk.CTkFrame):
    """Coloured message box for important notes."""

    ICONS = {"info": "i", "success": "✓", "warning": "!", "danger": "!", "neutral": "i"}

    def __init__(self, master: tk.Misc, text: str, *, title: str | None = None,
                 tone: str = "info") -> None:
        bg, border, fg = t.TONES[tone]
        super().__init__(master, fg_color=bg, corner_radius=t.RADIUS_CONTROL,
                         border_width=1, border_color=border)
        self.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            self, text=self.ICONS[tone], width=24, height=24, corner_radius=12,
            fg_color=fg, text_color=bg, font=t.fonts().caption_bold,
        ).grid(row=0, column=0, rowspan=2, sticky="n", padx=(14, 10), pady=14)
        row = 0
        if title:
            label(self, title, "body_bold", color=fg).grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=(12, 0))
            row = 1
        self.text_label = label(self, text, "body", color=t.TEXT if tone != "neutral" else t.TEXT_SECONDARY)
        self.text_label.grid(row=row, column=1, sticky="ew", padx=(0, 14),
                             pady=(2 if title else 13, 12))


class Collapsible(ctk.CTkFrame):
    """Card whose body can be expanded and collapsed."""

    def __init__(self, master: tk.Misc, title: str, subtitle: str | None = None, *,
                 opened: bool = False, on_toggle: Callable[[bool], None] | None = None) -> None:
        super().__init__(master, fg_color=t.SURFACE, corner_radius=t.RADIUS_CARD,
                         border_width=1, border_color=t.BORDER)
        self.columnconfigure(0, weight=1)
        self._opened = opened
        self._on_toggle = on_toggle
        self.header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        self.header.grid(row=0, column=0, sticky="ew", padx=t.PAD_CARD, pady=16)
        self.header.columnconfigure(1, weight=1)
        self.chevron = ctk.CTkLabel(self.header, text="", width=22, font=t.fonts().subtitle,
                                    text_color=t.TEXT_SECONDARY)
        self.chevron.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 8))
        ctk.CTkLabel(self.header, text=title, font=t.fonts().subtitle, text_color=t.TEXT,
                     anchor="w").grid(row=0, column=1, sticky="w")
        if subtitle:
            ctk.CTkLabel(self.header, text=subtitle, font=t.fonts().caption, text_color=t.TEXT_MUTED,
                         anchor="w").grid(row=1, column=1, sticky="w")
        bind_tree(self.header, "<Button-1>", lambda _e: self.toggle())
        set_tree_cursor(self.header, "hand2")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self._apply()

    @property
    def opened(self) -> bool:
        return self._opened

    def toggle(self) -> None:
        self._opened = not self._opened
        self._apply()
        if self._on_toggle:
            self._on_toggle(self._opened)

    def _apply(self) -> None:
        self.chevron.configure(text="▾" if self._opened else "▸")
        if self._opened:
            self.body.grid(row=1, column=0, sticky="nsew", padx=t.PAD_CARD, pady=(0, t.PAD_CARD))
        else:
            self.body.grid_forget()


class OptionCard(ctk.CTkFrame):
    """Large selectable card acting as a radio button."""

    def __init__(self, master: tk.Misc, *, title: str, description: str, value: str,
                 variable: tk.StringVar, command: Callable[[str], None] | None = None,
                 badge: tuple[str, str] | None = None, details: str | None = None) -> None:
        super().__init__(master, fg_color=t.SURFACE, corner_radius=t.RADIUS_CARD,
                         border_width=2, border_color=t.BORDER, cursor="hand2")
        self.value = value
        self.variable = variable
        self.command = command
        self._hover = False
        self.columnconfigure(1, weight=1)
        self.indicator = ctk.CTkLabel(self, text="", width=22, height=22, corner_radius=11,
                                      font=t.fonts().caption_bold)
        self.indicator.grid(row=0, column=0, sticky="n", padx=(18, 12), pady=(20, 0))
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=1, sticky="ew", padx=(0, 18), pady=(16, 0))
        head.columnconfigure(0, weight=1)
        self.title_label = ctk.CTkLabel(head, text=title, font=t.fonts().subtitle, text_color=t.TEXT, anchor="w")
        self.title_label.grid(row=0, column=0, sticky="w")
        if badge:
            tag(head, badge[0], badge[1]).grid(row=0, column=1, sticky="e", padx=(8, 0))
        label(self, description, "body", color=t.TEXT_SECONDARY).grid(
            row=1, column=1, sticky="ew", padx=(0, 18), pady=(4, 16 if not details else 6)
        )
        if details:
            label(self, details, "caption", color=t.TEXT_MUTED).grid(
                row=2, column=1, sticky="ew", padx=(0, 18), pady=(0, 16)
            )
        bind_tree(self, "<Button-1>", lambda _e: self.select())
        bind_tree(self, "<Enter>", lambda _e: self._set_hover(True))
        bind_tree(self, "<Leave>", lambda _e: self._set_hover(False))
        set_tree_cursor(self, "hand2")
        variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def select(self) -> None:
        if self.variable.get() != self.value:
            self.variable.set(self.value)
        if self.command:
            self.command(self.value)

    def _set_hover(self, hover: bool) -> None:
        self._hover = hover
        self.refresh()

    def refresh(self) -> None:
        if not self.winfo_exists():
            return
        selected = self.variable.get() == self.value
        if selected:
            self.configure(border_color=t.ACCENT, fg_color=t.ACCENT_SOFT)
            self.indicator.configure(text="✓", fg_color=t.ACCENT, text_color=t.TEXT_ON_ACCENT)
        else:
            self.configure(border_color=t.BORDER_STRONG if self._hover else t.BORDER, fg_color=t.SURFACE)
            self.indicator.configure(text="", fg_color=t.DISABLED_BG, text_color=t.TEXT_MUTED)


class StatTile(ctk.CTkFrame):
    def __init__(self, master: tk.Misc, caption: str, value: str, *, tone: str | None = None,
                 note: str | None = None) -> None:
        bg, border, fg = t.TONES[tone] if tone else (t.SURFACE_ALT, t.BORDER, t.TEXT)
        super().__init__(master, fg_color=bg, corner_radius=t.RADIUS_CONTROL, border_width=1, border_color=border)
        self.columnconfigure(0, weight=1)
        label(self, caption, "caption_bold", color=t.TEXT_MUTED if not tone else fg).grid(
            row=0, column=0, sticky="ew", padx=16, pady=(12, 0))
        self.value_label = label(self, value, "subtitle", color=t.TEXT if not tone else fg)
        self.value_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(2, 12 if not note else 0))
        if note:
            label(self, note, "caption", color=t.TEXT_MUTED).grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 12))


# --------------------------------------------------------------------------
# Form inputs
# --------------------------------------------------------------------------

def entry(master: tk.Misc, variable: tk.StringVar, *, width: int = 160, placeholder: str | None = None,
          justify: str = "left") -> ctk.CTkEntry:
    return ctk.CTkEntry(
        master, textvariable=variable, width=width, height=t.ENTRY_HEIGHT,
        corner_radius=t.RADIUS_CONTROL, border_width=1, border_color=t.BORDER_STRONG,
        fg_color=t.SURFACE, text_color=t.TEXT, font=t.fonts().body,
        placeholder_text=placeholder, justify=justify,
    )


class FormField(ctk.CTkFrame):
    """Label stacked above an entry, with optional unit and hint."""

    def __init__(self, master: tk.Misc, label_text: str, variable: tk.StringVar, *, unit: str | None = None,
                 hint: str | None = None, width: int = 160) -> None:
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(0, weight=1)
        label(self, label_text, "label", wrap=False).grid(row=0, column=0, columnspan=2, sticky="w")
        self.entry = entry(self, variable, width=width)
        self.entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        if unit:
            ctk.CTkLabel(self, text=unit, font=t.fonts().body, text_color=t.TEXT_MUTED).grid(
                row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        if hint:
            label(self, hint, "caption").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))


class ChoiceField(ctk.CTkFrame):
    def __init__(self, master: tk.Misc, label_text: str, variable: tk.StringVar, values: Iterable[str], *,
                 hint: str | None = None, width: int = 220) -> None:
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(0, weight=1)
        label(self, label_text, "label", wrap=False).grid(row=0, column=0, sticky="w")
        self.menu = ctk.CTkOptionMenu(
            self, variable=variable, values=list(values), width=width, height=t.ENTRY_HEIGHT,
            corner_radius=t.RADIUS_CONTROL, fg_color=t.SURFACE_ALT, button_color=t.ACCENT_SOFT,
            button_hover_color=t.ACCENT_SOFT_HOVER, text_color=t.TEXT, font=t.fonts().body,
            dropdown_font=t.fonts().body, dropdown_fg_color=t.SURFACE, dropdown_hover_color=t.ACCENT_SOFT,
            dropdown_text_color=t.TEXT, dynamic_resizing=False,
        )
        self.menu.grid(row=1, column=0, sticky="w", pady=(6, 0))
        if hint:
            label(self, hint, "caption").grid(row=2, column=0, sticky="ew", pady=(4, 0))


class DirectoryMemory:
    """Remembers the last folder used for each kind of file dialog."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.dirs: dict[str, str] = dict(initial or {})

    def get(self, key: str, fallback: str | Path | None = None) -> str | None:
        value = self.dirs.get(key)
        if value and Path(value).is_dir():
            return value
        if fallback:
            path = Path(fallback).expanduser()
            candidate = path if path.is_dir() else path.parent
            if candidate.is_dir():
                return str(candidate)
        return None

    def remember(self, key: str, path: str | Path) -> None:
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        self.dirs[key] = str(folder)


class PathField(ctk.CTkFrame):
    """File/folder path input with browse button and drag-and-drop support."""

    def __init__(
        self,
        master: tk.Misc,
        label_text: str,
        variable: tk.StringVar,
        *,
        mode: Literal["open", "save", "dir"] = "open",
        dialog_title: str = "ファイルを選択",
        filetypes: tuple[tuple[str, str], ...] = (("すべてのファイル", "*.*"),),
        extensions: tuple[str, ...] = (),
        default_extension: str | None = None,
        memory: DirectoryMemory | None = None,
        memory_key: str = "default",
        dnd_enabled: bool = False,
        required: bool = False,
        hint: str | None = None,
        placeholder: str | None = None,
        on_error: Callable[[str], None] | None = None,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self.variable = variable
        self.mode = mode
        self.dialog_title = dialog_title
        self.filetypes = filetypes
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.default_extension = default_extension
        self.memory = memory
        self.memory_key = memory_key
        self.on_error = on_error
        self.on_change = on_change
        self.columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        label(head, label_text, "label", wrap=False).pack(side="left")
        if required:
            tag(head, "必須", "info").pack(side="left", padx=(8, 0))

        self.box = ctk.CTkFrame(self, fg_color=t.SURFACE_ALT, corner_radius=t.RADIUS_CONTROL + 2,
                                border_width=1, border_color=t.BORDER_STRONG)
        self.box.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.box.columnconfigure(0, weight=1)
        if dnd_enabled and not placeholder:
            placeholder = "ここにドラッグ＆ドロップ、または［参照］で選択"
        self.entry = ctk.CTkEntry(
            self.box, textvariable=variable, height=t.ENTRY_HEIGHT - 2, corner_radius=t.RADIUS_CONTROL,
            border_width=0, fg_color=t.SURFACE_ALT, text_color=t.TEXT, font=t.fonts().body,
            placeholder_text=placeholder,
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(6, 6), pady=5)
        browse_text = {"open": "参照…", "save": "保存先…", "dir": "フォルダ…"}[mode]
        self.browse_button = button(self.box, browse_text, self.browse, variant="secondary", size="sm")
        self.browse_button.grid(row=0, column=1, padx=(0, 6), pady=6)
        if hint:
            label(self, hint, "caption").grid(row=2, column=0, sticky="ew", pady=(5, 0))
        if dnd_enabled:
            self._register_drop()

    # ---- Browse ----------------------------------------------------------
    def browse(self) -> None:
        current = self.variable.get().strip()
        initialdir = self.memory.get(self.memory_key, current or None) if self.memory else None
        options: dict[str, object] = {"title": self.dialog_title, "parent": self.winfo_toplevel()}
        if initialdir:
            options["initialdir"] = initialdir
        if self.mode == "open":
            path = filedialog.askopenfilename(filetypes=self.filetypes, **options)
        elif self.mode == "save":
            if current:
                options["initialfile"] = Path(current).name
            path = filedialog.asksaveasfilename(
                filetypes=self.filetypes, defaultextension=self.default_extension or "", **options
            )
        else:
            path = filedialog.askdirectory(**options)
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        self.variable.set(path)
        if self.memory:
            self.memory.remember(self.memory_key, path)
        if self.on_change:
            self.on_change(path)

    # ---- Drag and drop ---------------------------------------------------
    def _register_drop(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES
        except ImportError:
            return
        targets = [self.box, self.entry, getattr(self.entry, "_entry", None), self.box._canvas]
        for target in targets:
            if target is None:
                continue
            try:
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                target.dnd_bind("<<DropLeave>>", self._on_drop_leave)
                target.dnd_bind("<<Drop>>", self._on_drop)
            except (tk.TclError, AttributeError):
                continue

    def _on_drop_enter(self, event: tk.Event) -> str:
        self.box.configure(border_color=t.ACCENT, fg_color=t.ACCENT_SOFT, border_width=2)
        self.entry.configure(fg_color=t.ACCENT_SOFT)
        return getattr(event, "action", "copy")

    def _on_drop_leave(self, event: tk.Event) -> str:
        self._reset_box()
        return getattr(event, "action", "copy")

    def _reset_box(self) -> None:
        self.box.configure(border_color=t.BORDER_STRONG, fg_color=t.SURFACE_ALT, border_width=1)
        self.entry.configure(fg_color=t.SURFACE_ALT)

    def _on_drop(self, event: tk.Event) -> str:
        self._reset_box()
        try:
            items = list(self.tk.splitlist(event.data))
        except tk.TclError:
            items = [str(event.data)]
        if not items:
            return "refuse_drop"
        if len(items) > 1:
            self._error("ファイルは1つだけドロップしてください。")
            return "refuse_drop"
        candidate = Path(items[0])
        error = self.validate_drop(candidate)
        if error:
            self._error(error)
            return "refuse_drop"
        self.set_path(str(candidate))
        return getattr(event, "action", "copy")

    def validate_drop(self, path: Path) -> str | None:
        if self.mode == "dir":
            return None if path.is_dir() else "フォルダをドロップしてください。"
        if path.is_dir():
            return "フォルダではなくファイルをドロップしてください。"
        if self.extensions and path.suffix.lower() not in self.extensions:
            allowed = " / ".join(self.extensions)
            return f"対応していない形式です（{path.suffix or '拡張子なし'}）。{allowed} を指定してください。"
        return None

    def _error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)


# --------------------------------------------------------------------------
# Toast notifications
# --------------------------------------------------------------------------

class ToastManager:
    """Shows transient notifications in the bottom-right corner of the window."""

    def __init__(self, root: tk.Misc, bottom_offset: int = 96) -> None:
        self.root = root
        self.bottom_offset = bottom_offset
        self._frame: ctk.CTkFrame | None = None
        self._after_id: str | None = None

    def show(self, message: str, *, title: str | None = None, tone: str = "success",
             duration_ms: int = 4200) -> None:
        self.hide()
        bg, border, fg = t.TONES[tone]
        frame = ctk.CTkFrame(self.root, fg_color=t.SURFACE, corner_radius=12, border_width=1,
                             border_color=border)
        frame.columnconfigure(1, weight=1)
        icon = {"success": "✓", "danger": "!", "warning": "!", "info": "i", "neutral": "i"}[tone]
        ctk.CTkLabel(frame, text=icon, width=30, height=30, corner_radius=15, fg_color=fg,
                     text_color=t.SURFACE, font=t.fonts().body_bold).grid(
            row=0, column=0, rowspan=2, padx=(16, 12), pady=16, sticky="n")
        row = 0
        if title:
            ctk.CTkLabel(frame, text=title, font=t.fonts().body_bold, text_color=t.TEXT, anchor="w").grid(
                row=0, column=1, sticky="w", padx=(0, 8), pady=(14, 0))
            row = 1
        ctk.CTkLabel(frame, text=message, font=t.fonts().body, text_color=t.TEXT_SECONDARY, anchor="w",
                     justify="left", wraplength=360).grid(
            row=row, column=1, sticky="w", padx=(0, 8), pady=(2 if title else 18, 14))
        close = ctk.CTkButton(frame, text="✕", width=30, height=30, corner_radius=8, fg_color="transparent",
                              hover_color=t.HOVER_NEUTRAL, text_color=t.TEXT_MUTED, font=t.fonts().body,
                              command=self.hide)
        close.grid(row=0, column=2, padx=(0, 10), pady=10, sticky="n")
        frame.place(relx=1.0, rely=1.0, x=-24, y=-self.bottom_offset, anchor="se")
        frame.lift()
        self._frame = frame
        self._after_id = self.root.after(duration_ms, self.hide)

    def hide(self) -> None:
        if self._after_id:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._frame is not None:
            try:
                self._frame.destroy()
            except tk.TclError:
                pass
            self._frame = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

class StepSidebar(ctk.CTkFrame):
    """Vertical step list showing progress through the workflow."""

    def __init__(self, master: tk.Misc, steps: list[tuple[str, str]], on_select: Callable[[int], None],
                 app_title: str, app_subtitle: str) -> None:
        super().__init__(master, fg_color=t.SURFACE, corner_radius=0, width=t.SIDEBAR_WIDTH)
        self.grid_propagate(False)
        self.pack_propagate(False)
        self.on_select = on_select
        self.columnconfigure(0, weight=1)

        brand = ctk.CTkFrame(self, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=22, pady=(26, 22))
        brand.columnconfigure(1, weight=1)
        ctk.CTkLabel(brand, text="◎", width=40, height=40, corner_radius=10, fg_color=t.ACCENT,
                     text_color=t.TEXT_ON_ACCENT, font=t.fonts().title).grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ctk.CTkLabel(brand, text=app_title, font=t.fonts().subtitle, text_color=t.TEXT, anchor="w").grid(
            row=0, column=1, sticky="w")
        ctk.CTkLabel(brand, text=app_subtitle, font=t.fonts().caption, text_color=t.TEXT_MUTED, anchor="w").grid(
            row=1, column=1, sticky="w")

        ctk.CTkFrame(self, fg_color=t.BORDER, height=1, corner_radius=0).grid(row=1, column=0, sticky="ew", padx=18)
        ctk.CTkLabel(self, text="作業の流れ", font=t.fonts().caption_bold, text_color=t.TEXT_MUTED,
                     anchor="w").grid(row=2, column=0, sticky="ew", padx=24, pady=(18, 6))

        self.items: list[dict[str, object]] = []
        for index, (title, subtitle) in enumerate(steps):
            item = ctk.CTkFrame(self, fg_color="transparent", corner_radius=10, height=60)
            item.grid(row=3 + index, column=0, sticky="ew", padx=12, pady=2)
            item.columnconfigure(1, weight=1)
            badge = ctk.CTkLabel(item, text=str(index + 1), width=32, height=32, corner_radius=16,
                                 font=t.fonts().step_number)
            badge.grid(row=0, column=0, rowspan=2, padx=(12, 12), pady=12)
            title_label = ctk.CTkLabel(item, text=title, font=t.fonts().body_bold, anchor="w")
            title_label.grid(row=0, column=1, sticky="sw", pady=(10, 0))
            sub_label = ctk.CTkLabel(item, text=subtitle, font=t.fonts().caption, anchor="w", height=20)
            sub_label.grid(row=1, column=1, sticky="nw", pady=(0, 10))
            bind_tree(item, "<Button-1>", lambda _e, i=index: self._clicked(i))
            bind_tree(item, "<Enter>", lambda _e, i=index: self._hover(i, True))
            bind_tree(item, "<Leave>", lambda _e, i=index: self._hover(i, False))
            self.items.append({"frame": item, "badge": badge, "title": title_label, "sub": sub_label,
                               "state": "upcoming", "enabled": False})
        self.rowconfigure(3 + len(steps), weight=1)
        self.footer = ctk.CTkLabel(self, text="", font=t.fonts().caption, text_color=t.TEXT_MUTED,
                                   anchor="w", justify="left", wraplength=t.SIDEBAR_WIDTH - 48)
        self.footer.grid(row=4 + len(steps), column=0, sticky="ew", padx=24, pady=(0, 20))

    def _clicked(self, index: int) -> None:
        item = self.items[index]
        if item["enabled"] and item["state"] != "current":
            self.on_select(index)

    def _hover(self, index: int, inside: bool) -> None:
        item = self.items[index]
        frame: ctk.CTkFrame = item["frame"]  # type: ignore[assignment]
        if item["state"] == "current":
            return
        frame.configure(fg_color=t.HOVER_NEUTRAL if inside and item["enabled"] else "transparent")

    def set_state(self, current: int, done: set[int], enabled: set[int]) -> None:
        for index, item in enumerate(self.items):
            frame: ctk.CTkFrame = item["frame"]  # type: ignore[assignment]
            badge: ctk.CTkLabel = item["badge"]  # type: ignore[assignment]
            title: ctk.CTkLabel = item["title"]  # type: ignore[assignment]
            sub: ctk.CTkLabel = item["sub"]  # type: ignore[assignment]
            is_enabled = index in enabled
            item["enabled"] = is_enabled
            if index == current:
                item["state"] = "current"
                frame.configure(fg_color=t.ACCENT_SOFT)
                badge.configure(text=str(index + 1), fg_color=t.ACCENT, text_color=t.TEXT_ON_ACCENT)
                title.configure(text_color=t.ACCENT_TEXT)
                sub.configure(text_color=t.ACCENT_TEXT)
            elif index in done:
                item["state"] = "done"
                frame.configure(fg_color="transparent")
                badge.configure(text="✓", fg_color=t.SUCCESS_SOFT, text_color=t.SUCCESS)
                title.configure(text_color=t.TEXT if is_enabled else t.TEXT_MUTED)
                sub.configure(text_color=t.TEXT_MUTED)
            else:
                item["state"] = "upcoming"
                frame.configure(fg_color="transparent")
                badge.configure(text=str(index + 1), fg_color=t.DISABLED_BG,
                                text_color=t.TEXT_SECONDARY if is_enabled else t.DISABLED_TEXT)
                title.configure(text_color=t.TEXT_SECONDARY if is_enabled else t.DISABLED_TEXT)
                sub.configure(text_color=t.TEXT_MUTED if is_enabled else t.DISABLED_TEXT)
            set_tree_cursor(frame, "hand2" if is_enabled and index != current else "")

    def set_footer(self, text: str) -> None:
        self.footer.configure(text=text)
