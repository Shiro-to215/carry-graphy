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
from .widgets import DirectoryMemory, StepSidebar, ToastManager, button, label, tag

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

        self.toast = ToastManager(self.root)

        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_board_page()
        if self.settings_store.last_error:
            self.toast.show(self.settings_store.last_error, title="設定の読み込み", tone="warning")
        self.root.after(100, self._poll_messages)

    # ------------------------------------------------------------------
    # Shell helpers (used by page modules)
    # ------------------------------------------------------------------

    def _clear_content(self) -> None:
        self.point_canvas = None
        for child in self.content.winfo_children():
            child.destroy()
        for row in range(6):
            self.content.rowconfigure(row, weight=0)
        for col in range(4):
            self.content.columnconfigure(col, weight=0)
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

    def begin_page(
        self,
        *,
        step_index: int,
        step_badge: str,
        title: str,
        subtitle: str,
        scrollable: bool = True,
    ) -> ctk.CTkBaseClass:
        self.save_settings()
        self._clear_content()
        self.current_step = step_index
        self.step_label.set(f"{step_badge}　{title}")
        self.header_badge.configure(text=f" {step_badge} ")
        self.header_title.configure(text=title)
        self.header_subtitle.configure(text=subtitle)
        self.refresh_sidebar()

        if scrollable:
            scroll = ctk.CTkScrollableFrame(
                self.content,
                fg_color="transparent",
                corner_radius=0,
                scrollbar_button_color=t.BORDER_STRONG,
                scrollbar_button_hover_color=t.TEXT_MUTED,
            )
            scroll.grid(row=0, column=0, sticky="nsew")
            scroll.columnconfigure(0, weight=1)
            return scroll
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        return frame

    def set_action_bar(
        self,
        *,
        primary: tuple[str, Callable[[], object]],
        back: tuple[str, Callable[[], object]] | None = None,
        secondary: list[tuple[str, Callable[[], object]]] | None = None,
    ) -> ctk.CTkButton:
        for child in self.action_left.winfo_children():
            child.destroy()
        for child in self.action_right.winfo_children():
            child.destroy()
        self.action_back_button = None
        if back is not None:
            self.action_back_button = button(
                self.action_left, back[0], back[1], variant="secondary", size="md",
            )
            self.action_back_button.pack(side="left")
        for idx, (text, cmd) in enumerate(secondary or []):
            sec_btn = button(self.action_right, text, cmd, variant="secondary", size="md")
            sec_btn.pack(side="left", padx=(0, 10))
            setattr(self, f"action_secondary_{idx}", sec_btn)
        self.action_primary_button = button(
            self.action_right, primary[0], primary[1], variant="primary", size="lg",
        )
        self.action_primary_button.pack(side="left")
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

    def save_settings(self) -> None:
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

    def _on_close(self) -> None:
        self.save_settings()
        self.root.destroy()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda _e: self._undo_point() if self.current_step == 3 else None)
        self.root.bind("<BackSpace>", self._on_backspace)
        self.root.bind("<KeyPress-space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)

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
            self.root.after(100, self._poll_messages)


def main() -> None:
    root = create_root()
    IntegratedApp(root)
    root.mainloop()
