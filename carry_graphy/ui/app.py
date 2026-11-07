"""Main application window and workflow coordinator."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable

import customtkinter as ctk
import numpy as np

from ..core.calibration_io import (
    distortion_from_metadata,
    image_size_from_metadata,
    load_calibration_file,
    read_calibration_result,
    write_calibration_summary,
)
from ..core.homography import format_homography_summary
from ..core.settings import SettingsStore, get_section
from ..paths import DEFAULT_RESULTS_DIR
from . import theme as t
from .pages.board_page import build_board_page, create_board
from .pages.calibration_page import (
    append_log,
    build_calibration_page,
    handle_calibration_done,
    import_calibration_data,
    run_calibration_process,
    set_calibration_busy,
    start_calibration,
    update_calibration_progress,
)
from .pages.completion_page import build_completion_page
from .pages.homography_page import (
    build_homography_page,
    build_resolution_choice_page,
    choose_calibration_resolution,
    choose_reference_resolution,
    import_homography_data,
    open_point_selection,
)
from .pages.point_page import build_point_page, prepare_point_selection, save_homography_action
from .point_canvas import PointCanvas
from .widgets import (
    DirectoryMemory,
    FastScrollableFrame,
    StepSidebar,
    ToastManager,
    button,
    label,
    set_button_enabled,
    tag,
)

STEPS: list[tuple[str, str]] = [
    ("ボード作成", "任意・PNG出力"),
    ("キャリブレーション", "動画解析 / NPZ読込"),
    ("ホモグラフィ設定", "基準画像と実寸"),
    ("四隅の選択", "クリック＆ドラッグ調整"),
    ("完了", "行列の確認・出力"),
]


def create_root() -> ctk.CTk:
    """Create a CustomTkinter root window with optional TkinterDnD support."""
    t.apply_global_theme()
    try:
        from tkinterdnd2 import TkinterDnD

        class _DnDCTk(ctk.CTk, TkinterDnD.DnDWrapper):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                try:
                    self.TkdndVersion = TkinterDnD._require(self)
                except tk.TclError:
                    self.TkdndVersion = None

        return _DnDCTk(fg_color=t.BG)
    except Exception:
        return ctk.CTk(fg_color=t.BG)


class IntegratedApp:
    """Single-window camera calibration and homography application."""

    def __init__(self, root: tk.Tk | ctk.CTk, settings_store: SettingsStore | None = None) -> None:
        self.root = root
        t.apply_global_theme()
        t.init_fonts(root)
        self.root.title("カメラキャリブレーション・ホモグラフィ統合アプリ")
        self.root.minsize(1100, 780)
        try:
            self.root.configure(fg_color=t.BG)  # type: ignore[call-arg]
        except tk.TclError:
            self.root.configure(bg=t.BG)

        self.settings_store = settings_store or SettingsStore()
        saved = self.settings_store.load()
        geometry = saved.get("geometry")
        self.root.geometry(geometry if isinstance(geometry, str) and "x" in geometry else "1240x880")

        # ---- State -------------------------------------------------------
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.calibration_data: dict[str, object] | None = None
        self.image_path: Path | None = None
        self.output_path: Path | None = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion: np.ndarray | None = None
        self.undistorted_image: np.ndarray | None = None
        self.image_alignment: dict[str, object] = {}
        self.calibration_camera_matrix: np.ndarray | None = None
        self.calibration_image_size = (0, 0)
        self.points: list[tuple[float, float]] = []
        self.current_plane_size: tuple[float, float] = (1.0, 1.0)
        self.last_completion: tuple[np.ndarray, bool, bool] | None = None
        self.image_width = 0
        self.image_height = 0
        self.display_width = 0
        self.display_height = 0
        self.display_scale = 1.0
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        self.fit_scale = 1.0
        self.zoom_factor = 1.0
        self.viewport_width = 880
        self.viewport_height = 500
        self.photo: tk.PhotoImage | None = None
        self.board_preview_photo: tk.PhotoImage | None = None
        self.board_generated_settings: tuple[int, int, float, float, str, int] | None = None
        self.point_canvas: PointCanvas | None = None
        self._cached_point_canvas: PointCanvas | None = None
        self._point_image_dirty = False
        self._cached_pages: dict[str, tk.Misc] = {}
        self._transient_page: tk.Misc | None = None
        self._active_page: tk.Misc | None = None
        self._save_after_id: str | None = None
        self._prewarm_after_id: str | None = None
        self._poll_after_id: str | None = None
        self.current_step = 0
        self.is_calibrating = False
        self.advanced_calibration_open = bool(saved.get("advanced_calibration_open", False))

        # ---- Form variables (defaults merged with saved settings) --------
        cal_defaults = {
            "video": "",
            "output": str(DEFAULT_RESULTS_DIR / "calibration"),
            "squares_x": "7",
            "squares_y": "5",
            "square_length": "30",
            "marker_length": "22",
            "dictionary": "DICT_4X4_50",
            "top_n": "30",
            "stride": "5",
            "blur_threshold": "30",
            "min_corners": "8",
            "max_frames": "0",
            "low_coverage": "0.40",
        }
        board_defaults = {
            "dpi": "300",
            "output": str(DEFAULT_RESULTS_DIR / "charuco_board.png"),
        }
        hom_defaults = {
            "image": "",
            "width": "",
            "height": "",
            "output": str(DEFAULT_RESULTS_DIR / "homography_matrix.npy"),
        }
        saved_cal = get_section(saved, "calibration")
        saved_board = get_section(saved, "board")
        saved_hom = get_section(saved, "homography")
        self.calibration_vars = {
            k: tk.StringVar(master=root, value=str(saved_cal.get(k, v))) for k, v in cal_defaults.items()
        }
        self.board_vars = {
            k: tk.StringVar(master=root, value=str(saved_board.get(k, v))) for k, v in board_defaults.items()
        }
        self.homography_vars = {
            k: tk.StringVar(master=root, value=str(saved_hom.get(k, v))) for k, v in hom_defaults.items()
        }
        self.calibration_import_var = tk.StringVar(master=root, value=str(saved.get("calibration_import", "")))
        self.homography_import_var = tk.StringVar(master=root, value=str(saved.get("homography_import", "")))
        self.resolution_choice_var = tk.StringVar(master=root, value=str(saved.get("resolution_choice", "calibration")))
        self.loupe_var = tk.BooleanVar(master=root, value=bool(saved.get("loupe_enabled", True)))
        self.dir_memory = DirectoryMemory(get_section(saved, "directories"))

        # Status StringVars referenced across rebuilds
        self.step_label = tk.StringVar(master=root, value="ステップ 1 / 4　ボード作成（任意）")
        self.board_status = tk.StringVar(master=root, value="ボードを作成するか、そのまま次へ進んでください。")
        self.status = tk.StringVar(master=root, value="動画とボード設定を指定してください")
        self.calibration_progress = tk.DoubleVar(master=root, value=0.0)
        self.progress_detail = tk.StringVar(master=root, value="処理開始を待っています")
        self.progress_mode = "determinate"
        self.zoom_status = tk.StringVar(master=root, value="表示倍率: 100%")
        self.point_status = tk.StringVar(master=root, value="")

        # ---- Shell layout (sidebar | main area) --------------------------
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(2, weight=1)

        self.sidebar = StepSidebar(
            self.root,
            STEPS,
            on_select=self._on_sidebar_select,
            app_title="Carry-Graphy",
            app_subtitle="キャリブレーション & ホモグラフィ",
        )
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.set_footer("設定と入出力パスは自動で記憶されます。")

        ctk.CTkFrame(self.root, fg_color=t.BORDER, width=1, corner_radius=0).grid(
            row=0, column=1, sticky="ns"
        )

        self.main_area = ctk.CTkFrame(self.root, fg_color=t.BG, corner_radius=0)
        self.main_area.grid(row=0, column=2, sticky="nsew")
        self.main_area.rowconfigure(1, weight=1)
        self.main_area.columnconfigure(0, weight=1)

        # Page header
        self.header = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=t.PAD_PAGE_X, pady=(24, 12))
        self.header.columnconfigure(0, weight=1)
        self.header_badge = tag(self.header, "ステップ 1 / 4", "info")
        self.header_badge.grid(row=0, column=0, sticky="w")
        self.header_title = ctk.CTkLabel(
            self.header, text="", font=t.fonts().display, text_color=t.TEXT, anchor="w",
        )
        self.header_title.grid(row=1, column=0, sticky="w", pady=(6, 2))
        self.header_subtitle = label(self.header, "", "body", color=t.TEXT_SECONDARY)
        self.header_subtitle.grid(row=2, column=0, sticky="ew")

        # Dynamic body holder
        self.content = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=t.PAD_PAGE_X, pady=(4, 12))
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        # Bottom action bar
        ctk.CTkFrame(self.main_area, fg_color=t.BORDER, height=1, corner_radius=0).grid(
            row=2, column=0, sticky="ew"
        )
        self.action_bar = ctk.CTkFrame(self.main_area, fg_color=t.SURFACE, corner_radius=0, height=76)
        self.action_bar.grid(row=3, column=0, sticky="ew")
        self.action_bar.grid_propagate(False)
        self.action_bar.columnconfigure(1, weight=1)
        self.action_left = ctk.CTkFrame(self.action_bar, fg_color="transparent")
        self.action_left.grid(row=0, column=0, sticky="w", padx=t.PAD_PAGE_X, pady=14)
        self.action_right = ctk.CTkFrame(self.action_bar, fg_color="transparent")
        self.action_right.grid(row=0, column=2, sticky="e", padx=t.PAD_PAGE_X, pady=14)
        self.action_back_button: ctk.CTkButton | None = None
        self.action_primary_button: ctk.CTkButton | None = None
        self._pool_back_btn = button(self.action_left, "", None, variant="secondary", size="md")
        self._pool_sec_btns: list[ctk.CTkButton] = [
            button(self.action_right, "", None, variant="secondary", size="md"),
            button(self.action_right, "", None, variant="secondary", size="md"),
        ]
        self._pool_primary_btn = button(self.action_right, "", None, variant="primary", size="lg")

        self.toast = ToastManager(self.root)

        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.main_area.bind("<Destroy>", self._cancel_app_timers, add="+")

        self._build_board_page()
        if self.settings_store.last_error:
            self.toast.show(self.settings_store.last_error, title="設定の読み込み", tone="warning")
        self._prewarm_after_id = self.root.after(120, self._prewarm_next_page)
        self._poll_after_id = self.root.after(100, self._poll_messages)

    # ------------------------------------------------------------------
    # Shell helpers (used by page modules)
    # ------------------------------------------------------------------

    def _clear_content(self) -> None:
        self.point_canvas = None
        self._cached_point_canvas = None
        self._cached_pages.clear()
        self._transient_page = None
        self._active_page = None
        for child in self.content.winfo_children():
            child.destroy()
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

    def _create_page_container(self, *, scrollable: bool) -> tk.Misc:
        if scrollable:
            scroll = FastScrollableFrame(self.content, bg=t.BG)
            scroll.grid(row=0, column=0, sticky="nsew")
            scroll.columnconfigure(0, weight=1)
            return scroll
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        return frame

    def use_page(
        self,
        cache_key: str | None,
        *,
        step_index: int,
        step_badge: str,
        title: str,
        subtitle: str,
        scrollable: bool = True,
        background_build: bool = False,
    ) -> tuple[tk.Misc, bool]:
        """Return ``(page_container, created)`` using cached pages when available."""
        if background_build:
            assert cache_key is not None
            if cache_key in self._cached_pages:
                return self._cached_pages[cache_key], False
            container = self._create_page_container(scrollable=scrollable)
            container.grid_remove()  # type: ignore[attr-defined]
            self._cached_pages[cache_key] = container
            return container, True

        self._schedule_save_settings()
        if self._transient_page is not None:
            if self._active_page is self._transient_page:
                self._active_page = None
            self._transient_page.destroy()
            self._transient_page = None

        target_existing = self._cached_pages.get(cache_key) if cache_key is not None else None
        if (
            self._active_page is not None
            and self._active_page is not target_existing
            and self._active_page.winfo_exists()
        ):
            self._active_page.grid_remove()  # type: ignore[attr-defined]

        if cache_key != "point":
            self.point_canvas = None

        self.current_step = step_index
        self.step_label.set(f"{step_badge}　{title}")
        self.header_badge.configure(text=f" {step_badge} ")
        self.header_title.configure(text=title)
        self.header_subtitle.configure(text=subtitle)
        self.refresh_sidebar()

        if cache_key is not None:
            if target_existing is not None:
                target_existing.grid(row=0, column=0, sticky="nsew")  # type: ignore[call-arg]
                self._active_page = target_existing
                return target_existing, False
            container = self._create_page_container(scrollable=scrollable)
            self._cached_pages[cache_key] = container
            self._active_page = container
            return container, True

        container = self._create_page_container(scrollable=scrollable)
        self._transient_page = container
        self._active_page = container
        return container, True

    def begin_page(
        self,
        *,
        step_index: int,
        step_badge: str,
        title: str,
        subtitle: str,
        scrollable: bool = True,
    ) -> tk.Misc:
        container, _ = self.use_page(
            None,
            step_index=step_index,
            step_badge=step_badge,
            title=title,
            subtitle=subtitle,
            scrollable=scrollable,
        )
        return container

    def _prewarm_next_page(self) -> None:
        self._prewarm_after_id = None
        if not self.root.winfo_exists() or self.is_calibrating:
            return
        if "calibration" not in self._cached_pages:
            build_calibration_page(self, prewarm=True)
            self._prewarm_after_id = self.root.after(120, self._prewarm_next_page)
            return
        if "homography" not in self._cached_pages:
            build_homography_page(self, prewarm=True)

    def set_action_bar(
        self,
        *,
        primary: tuple[str, Callable[[], object]],
        back: tuple[str, Callable[[], object]] | None = None,
        secondary: list[tuple[str, Callable[[], object]]] | None = None,
    ) -> ctk.CTkButton:
        self._pool_back_btn.pack_forget()
        for btn in self._pool_sec_btns:
            btn.pack_forget()
        self._pool_primary_btn.pack_forget()

        self.action_back_button = None
        if back is not None:
            self._pool_back_btn.configure(text=back[0], command=back[1])
            set_button_enabled(self._pool_back_btn, True)
            self._pool_back_btn.pack(side="left")
            self.action_back_button = self._pool_back_btn

        sec_list = secondary or []
        while len(self._pool_sec_btns) < len(sec_list):
            self._pool_sec_btns.append(
                button(self.action_right, "", None, variant="secondary", size="md")
            )
        for idx, (text, cmd) in enumerate(sec_list):
            sec_btn = self._pool_sec_btns[idx]
            sec_btn.configure(text=text, command=cmd)
            set_button_enabled(sec_btn, True)
            sec_btn.pack(side="left", padx=(0, 10))
            setattr(self, f"action_secondary_{idx}", sec_btn)

        self._pool_primary_btn.configure(text=primary[0], command=primary[1])
        set_button_enabled(self._pool_primary_btn, True)
        self._pool_primary_btn.pack(side="left")
        self.action_primary_button = self._pool_primary_btn
        return self.action_primary_button

    def refresh_sidebar(self) -> None:
        done: set[int] = set()
        enabled: set[int] = {0, 1}
        if self.board_generated_settings is not None or self.current_step > 0:
            done.add(0)
        if self.calibration_data is not None:
            done.update({0, 1})
            enabled.add(2)
        if self.undistorted_image is not None:
            done.add(2)
            enabled.add(3)
        if self.last_completion is not None:
            done.update({2, 3, 4})
            enabled.add(4)
        if self.is_calibrating:
            enabled = {self.current_step}
        self.sidebar.set_state(self.current_step, done, enabled)

    def _on_sidebar_select(self, index: int) -> None:
        if self.is_calibrating:
            return
        if index == 0:
            self._build_board_page()
        elif index == 1:
            self._build_calibration_page()
        elif index == 2 and self.calibration_data is not None:
            self._build_homography_page()
        elif index == 3 and self.undistorted_image is not None:
            w_m, h_m = self.current_plane_size
            self._build_point_page(w_m, h_m)
        elif index == 4 and self.last_completion is not None:
            matrix, loaded, has_meta = self.last_completion
            self._build_completion_page(matrix, loaded=loaded, has_embedded_metadata=has_meta)

    # ------------------------------------------------------------------
    # Settings persistence & shortcuts
    # ------------------------------------------------------------------

    def _schedule_save_settings(self) -> None:
        if self._save_after_id is None and self.root.winfo_exists():
            self._save_after_id = self.root.after(300, self.save_settings)

    def save_settings(self) -> None:
        if self._save_after_id is not None:
            try:
                self.root.after_cancel(self._save_after_id)
            except tk.TclError:
                pass
            self._save_after_id = None
        try:
            geom = self.root.geometry()
        except tk.TclError:
            geom = "1240x880"
        self.settings_store.save({
            "geometry": geom,
            "calibration": {k: v.get() for k, v in self.calibration_vars.items()},
            "board": {k: v.get() for k, v in self.board_vars.items()},
            "homography": {k: v.get() for k, v in self.homography_vars.items()},
            "calibration_import": self.calibration_import_var.get(),
            "homography_import": self.homography_import_var.get(),
            "resolution_choice": self.resolution_choice_var.get(),
            "loupe_enabled": bool(self.loupe_var.get()),
            "advanced_calibration_open": self.advanced_calibration_open,
            "directories": dict(self.dir_memory.dirs),
        })

    def _cancel_app_timers(self, _event: tk.Event | None = None) -> None:
        for attr in ("_prewarm_after_id", "_save_after_id", "_poll_after_id"):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    def _on_close(self) -> None:
        self._cancel_app_timers()
        self.save_settings()
        self.root.destroy()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda _e: self._undo_point() if self.current_step == 3 else None)
        self.root.bind("<BackSpace>", self._on_backspace)
        self.root.bind("<KeyPress-space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)
        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")

    def _on_global_mousewheel(self, event: tk.Event) -> None:
        if not isinstance(self._active_page, FastScrollableFrame):
            return
        widget = event.widget
        if isinstance(widget, tk.Misc):
            w: tk.Misc | None = widget
            while w is not None:
                if self.point_canvas is not None and w is self.point_canvas.canvas:
                    return
                if isinstance(w, (ctk.CTkTextbox, tk.Text)):
                    return
                w = w.master
        if not event.delta:
            return
        delta_px = -int(round(event.delta / 120.0 * 80.0))
        if delta_px == 0:
            delta_px = -12 if event.delta > 0 else 12
        self._active_page.scroll_by_pixels(delta_px)

    def _on_backspace(self, event: tk.Event) -> None:
        if self.current_step != 3:
            return
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        self._undo_point()

    def _on_space_press(self, event: tk.Event) -> None:
        if self.point_canvas and not isinstance(event.widget, (tk.Entry, tk.Text)):
            self.point_canvas.set_space_held(True)

    def _on_space_release(self, _event: tk.Event) -> None:
        if self.point_canvas:
            self.point_canvas.set_space_held(False)

    # ------------------------------------------------------------------
    # Page & action methods (backward-compatible names)
    # ------------------------------------------------------------------

    def _build_board_page(self) -> None:
        build_board_page(self)

    def _create_board(self) -> None:
        create_board(self)

    def _choose_board_output(self) -> None:
        current = Path(self.board_vars["output"].get()).expanduser()
        path = filedialog.asksaveasfilename(
            title="ChArUcoボード画像の保存先",
            defaultextension=".png",
            initialdir=str(current.parent),
            initialfile=current.name,
            filetypes=(("PNG画像", "*.png"),),
            parent=self.root,
        )
        if path:
            self.board_vars["output"].set(path)
            self.dir_memory.remember("board_output", path)

    def _build_calibration_page(self) -> None:
        build_calibration_page(self)

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="キャリブレーション動画を選択",
            initialdir=self.dir_memory.get("video"),
            filetypes=(("動画", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("すべてのファイル", "*.*")),
            parent=self.root,
        )
        if path:
            self.calibration_vars["video"].set(path)
            self.dir_memory.remember("video", path)

    def _choose_calibration_output(self) -> None:
        path = filedialog.askdirectory(
            title="キャリブレーション結果の保存先",
            initialdir=self.dir_memory.get("calibration_output", self.calibration_vars["output"].get()),
            parent=self.root,
        )
        if path:
            self.calibration_vars["output"].set(path)
            self.dir_memory.remember("calibration_output", path)

    def _append_log(self, text: str) -> None:
        append_log(self, text)

    def _start_calibration(self) -> None:
        start_calibration(self)

    def _run_calibration_process(self, command: list[str], output: Path) -> None:
        run_calibration_process(self, command, output)

    def _update_calibration_progress(self, progress: dict[str, object]) -> None:
        update_calibration_progress(self, progress)

    def _import_calibration_data(self) -> None:
        import_calibration_data(self)

    def _load_calibration_file(self, path: Path) -> dict[str, object]:
        return load_calibration_file(path, resolve_metadata=self._choose_calibration_metadata)

    def _choose_calibration_metadata(self, data_path: Path, default_path: Path) -> Path | None:
        if default_path.is_file():
            return default_path
        selected = filedialog.askopenfilename(
            title="対応するキャリブレーションJSONを選択",
            filetypes=(("JSON", "*.json"), ("すべてのファイル", "*.*")),
            initialdir=str(data_path.parent),
            parent=self.root,
        )
        return Path(selected).expanduser() if selected else None

    @staticmethod
    def _distortion_from_metadata(metadata: dict[str, object]) -> np.ndarray | None:
        return distortion_from_metadata(metadata)

    @staticmethod
    def _image_size_from_metadata(metadata: dict[str, object]) -> np.ndarray | None:
        return image_size_from_metadata(metadata)

    def _read_calibration_result(self, output: Path) -> dict[str, object]:
        return read_calibration_result(output)

    @staticmethod
    def _write_calibration_summary(output: Path, data: dict[str, object]) -> Path:
        return write_calibration_summary(output, data)

    def _build_homography_page(self) -> None:
        build_homography_page(self)

    def _import_homography_data(self) -> None:
        import_homography_data(self)

    def _choose_reference_image(self) -> None:
        path = filedialog.askopenfilename(
            title="ホモグラフィの基準画像を選択",
            initialdir=self.dir_memory.get("reference_image"),
            filetypes=(("画像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("すべてのファイル", "*.*")),
            parent=self.root,
        )
        if path:
            self.homography_vars["image"].set(path)
            self.dir_memory.remember("reference_image", path)

    def _choose_homography_output(self) -> None:
        current = Path(self.homography_vars["output"].get()).expanduser()
        path = filedialog.asksaveasfilename(
            title="ホモグラフィ行列の保存先",
            defaultextension=".npy",
            initialdir=str(current.parent),
            initialfile=current.name,
            filetypes=(("NumPy行列", "*.npy"),),
            parent=self.root,
        )
        if path:
            self.homography_vars["output"].set(path)
            self.dir_memory.remember("homography_output", path)

    def _open_point_selection(self) -> None:
        open_point_selection(self)

    def _build_resolution_choice_page(
        self,
        image: np.ndarray,
        image_path: Path,
        output_path: Path,
        width_m: float,
        height_m: float,
        reference_size: tuple[int, int],
        calibration_size: tuple[int, int],
    ) -> None:
        build_resolution_choice_page(
            self, image, image_path, output_path, width_m, height_m, reference_size, calibration_size,
        )

    def _choose_calibration_resolution(
        self,
        image: np.ndarray,
        image_path: Path,
        output_path: Path,
        width_m: float,
        height_m: float,
        reference_size: tuple[int, int],
        calibration_size: tuple[int, int],
    ) -> None:
        choose_calibration_resolution(
            self, image, image_path, output_path, width_m, height_m, reference_size, calibration_size,
        )

    def _choose_reference_resolution(
        self,
        image: np.ndarray,
        image_path: Path,
        output_path: Path,
        width_m: float,
        height_m: float,
        reference_size: tuple[int, int],
        calibration_size: tuple[int, int],
    ) -> None:
        choose_reference_resolution(
            self, image, image_path, output_path, width_m, height_m, reference_size, calibration_size,
        )

    def _prepare_point_selection(
        self,
        image: np.ndarray,
        image_path: Path,
        output_path: Path,
        width_m: float,
        height_m: float,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        image_alignment: dict[str, object],
    ) -> None:
        prepare_point_selection(
            self, image, image_path, output_path, width_m, height_m,
            camera_matrix, distortion, image_alignment,
        )

    def _build_point_page(self, width_m: float, height_m: float) -> None:
        build_point_page(self, width_m, height_m)

    def _render_point_image(self, anchor: tuple[int, int] | None = None) -> None:
        if self.point_canvas:
            self.point_canvas.render_image(anchor)

    def _redraw_points(self) -> None:
        if self.point_canvas:
            self.point_canvas.redraw_points()

    def _change_zoom(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        if self.point_canvas:
            self.point_canvas.change_zoom(factor, anchor)

    def _fit_image(self) -> None:
        if self.point_canvas:
            self.point_canvas.fit()

    def _undo_point(self) -> None:
        if self.point_canvas:
            self.point_canvas.undo()

    def _reset_points(self) -> None:
        if self.point_canvas:
            self.point_canvas.reset()

    def _save_homography(self, width_m: float, height_m: float) -> None:
        save_homography_action(self, width_m, height_m)

    def _format_homography_summary(self, matrix: np.ndarray, metadata: dict[str, Any]) -> str:
        return format_homography_summary(matrix, metadata)

    def _build_completion_page(
        self, matrix: np.ndarray, loaded: bool = False, has_embedded_metadata: bool = False
    ) -> None:
        build_completion_page(self, matrix, loaded=loaded, has_embedded_metadata=has_embedded_metadata)

    # ------------------------------------------------------------------
    # Worker message loop
    # ------------------------------------------------------------------

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "calibration_error":
                    if hasattr(self, "progress_bar") and self.progress_bar.winfo_exists():
                        self.progress_bar.stop()
                    set_calibration_busy(self, False)
                    self.status.set("キャリブレーションを起動できませんでした")
                    self._append_log(str(payload))
                    self.toast.show(str(payload), title="起動エラー", tone="danger")
                    continue
                if kind == "calibration_progress":
                    self._update_calibration_progress(payload)  # type: ignore[arg-type]
                    continue
                if kind == "calibration_output":
                    self._append_log(str(payload))
                    continue
                if kind == "calibration_done":
                    return_code, stdout, stderr, output = payload  # type: ignore[misc]
                    handle_calibration_done(self, return_code, stdout, stderr, Path(output))
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self._poll_after_id = self.root.after(100, self._poll_messages)


def main() -> None:
    root = create_root()
    IntegratedApp(root)
    root.mainloop()
