"""Step 4: Selecting and fine-tuning the four homography corners."""

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

import cv2
import customtkinter as ctk
import numpy as np

from ...core.homography import (
    POINT_LABELS,
    HomographyComputeError,
    HomographyPointError,
    build_metadata,
    compute_homography,
    save_homography,
)
from .. import theme as t
from ..point_canvas import PointCanvas, is_quad_valid
from ..widgets import button, label, set_button_enabled

if TYPE_CHECKING:
    from ..app import IntegratedApp


def prepare_point_selection(
    app: IntegratedApp,
    image: np.ndarray,
    image_path: Path,
    output_path: Path,
    width_m: float,
    height_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_alignment: dict[str, object],
) -> None:
    assert app.calibration_data is not None
    try:
        undistorted = cv2.undistort(image, camera_matrix, distortion)
    except cv2.error as exc:
        messagebox.showerror("画像補正エラー", str(exc), parent=app.root)
        return

    app.image_path = image_path
    app.output_path = output_path
    app.camera_matrix = camera_matrix
    app.distortion = distortion
    app.undistorted_image = undistorted
    app.image_alignment = image_alignment
    app.calibration_camera_matrix = app.calibration_data["camera_matrix"]  # type: ignore[assignment]
    app.calibration_image_size = (
        int(app.calibration_data["width"]), int(app.calibration_data["height"])
    )
    app.image_height, app.image_width = undistorted.shape[:2]
    app.points = []
    app.current_plane_size = (width_m, height_m)
    build_point_page(app, width_m, height_m)


def build_point_page(app: IntegratedApp, width_m: float, height_m: float) -> None:
    assert app.undistorted_image is not None
    app.current_plane_size = (width_m, height_m)
    container = app.begin_page(
        step_index=3,
        step_badge="ステップ 4 / 4  ・  四隅の選択",
        title="補正画像上で平面の四隅を選択",
        subtitle="左下 → 右下 → 右上 → 左上 の順にクリックしてください。置いた点はドラッグして何度でも位置を微調整できます。",
        scrollable=False,
    )
    container.rowconfigure(2, weight=1)
    container.columnconfigure(0, weight=1)

    # ---- Step chips (1: 左下 -> 2: 右下 -> 3: 右上 -> 4: 左上) ----------
    chips_bar = ctk.CTkFrame(
        container, fg_color=t.SURFACE, corner_radius=t.RADIUS_CARD,
        border_width=1, border_color=t.BORDER,
    )
    chips_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    for col in range(4):
        chips_bar.columnconfigure(col, weight=1, uniform="pt_chip")

    app.point_chips = []
    for idx, name in enumerate(POINT_LABELS):
        chip = ctk.CTkFrame(chips_bar, fg_color=t.SURFACE_ALT, corner_radius=10, border_width=1, border_color=t.BORDER)
        chip.grid(row=0, column=idx, sticky="ew", padx=(12 if idx == 0 else 6, 12 if idx == 3 else 6), pady=10)
        chip.columnconfigure(1, weight=1)
        badge = ctk.CTkLabel(
            chip, text=str(idx + 1), width=28, height=28, corner_radius=14,
            fg_color=t.DISABLED_BG, text_color=t.TEXT_SECONDARY, font=t.fonts().step_number,
        )
        badge.grid(row=0, column=0, rowspan=2, padx=(10, 8), pady=8)
        title_lbl = ctk.CTkLabel(chip, text=name, font=t.fonts().body_bold, text_color=t.TEXT, anchor="w")
        title_lbl.grid(row=0, column=1, sticky="sw", pady=(6, 0))
        coord_lbl = ctk.CTkLabel(chip, text="未選択", font=t.fonts().caption, text_color=t.TEXT_MUTED, anchor="w")
        coord_lbl.grid(row=1, column=1, sticky="nw", pady=(0, 6))
        app.point_chips.append({"frame": chip, "badge": badge, "title": title_lbl, "coord": coord_lbl})

    # ---- Toolbar (zoom, loupe toggle, undo/reset, shortcuts hint) -------
    toolbar = ctk.CTkFrame(
        container, fg_color=t.SURFACE, corner_radius=t.RADIUS_CARD,
        border_width=1, border_color=t.BORDER,
    )
    toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    toolbar.columnconfigure(6, weight=1)

    button(
        toolbar, "− 縮小", lambda: app._change_zoom(1 / 1.25), variant="secondary", size="sm", width=84,
    ).grid(row=0, column=0, padx=(12, 4), pady=8)
    button(
        toolbar, "＋ 拡大", lambda: app._change_zoom(1.25), variant="secondary", size="sm", width=84,
    ).grid(row=0, column=1, padx=4, pady=8)
    button(
        toolbar, "全体表示", app._fit_image, variant="secondary", size="sm", width=96,
    ).grid(row=0, column=2, padx=(4, 12), pady=8)

    label(toolbar, "", "body_bold", color=t.TEXT_SECONDARY, wrap=False, textvariable=app.zoom_status).grid(
        row=0, column=3, padx=(0, 16), pady=8, sticky="w",
    )

    loupe_switch = ctk.CTkSwitch(
        toolbar,
        text="拡大ルーペ",
        variable=app.loupe_var,
        font=t.fonts().body_bold,
        text_color=t.TEXT,
        progress_color=t.ACCENT,
        command=lambda: (app.point_canvas.refresh_loupe() if app.point_canvas else None, app.save_settings()),
    )
    loupe_switch.grid(row=0, column=4, padx=(0, 16), pady=8)

    app.undo_button = button(
        toolbar, "1点戻す (Ctrl+Z)", app._undo_point, variant="secondary", size="sm", width=136,
    )
    app.undo_button.grid(row=0, column=5, padx=(0, 6), pady=8)
    app.reset_button = button(
        toolbar, "すべてリセット", app._reset_points, variant="danger", size="sm", width=124,
    )
    app.reset_button.grid(row=0, column=6, sticky="w", padx=(0, 12), pady=8)

    label(
        toolbar,
        "ホイール: 拡大/縮小  ・  点をドラッグ: 位置調整  ・  右ボタンドラッグ / Space+ドラッグ: 移動",
        "caption",
        color=t.TEXT_MUTED,
        wrap=False,
    ).grid(row=0, column=7, sticky="e", padx=(8, 14), pady=8)

    # ---- Image canvas + scrollbars --------------------------------------
    canvas_card = ctk.CTkFrame(
        container, fg_color=t.SURFACE, corner_radius=t.RADIUS_CARD,
        border_width=1, border_color=t.BORDER,
    )
    canvas_card.grid(row=2, column=0, sticky="nsew")
    canvas_card.rowconfigure(0, weight=1)
    canvas_card.columnconfigure(0, weight=1)

    app.point_canvas = PointCanvas(
        canvas_card,
        app.undistorted_image,
        app.points,
        on_points_changed=lambda: _sync_point_ui(app),
        on_zoom_changed=lambda z: _sync_zoom_from_canvas(app, z),
        on_cursor_changed=lambda pt, hit: _sync_cursor_status(app, pt, hit),
        loupe_enabled=app.loupe_var,
    )
    app.canvas = app.point_canvas.canvas
    app.canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))

    v_scroll = ctk.CTkScrollbar(canvas_card, orientation="vertical", command=app.canvas.yview)
    v_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=(10, 0))
    h_scroll = ctk.CTkScrollbar(canvas_card, orientation="horizontal", command=app.canvas.xview)
    h_scroll.grid(row=1, column=0, sticky="ew", padx=(10, 0), pady=(4, 8))
    app.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    # ---- Status strip under the canvas ----------------------------------
    status_strip = ctk.CTkFrame(container, fg_color="transparent")
    status_strip.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    status_strip.columnconfigure(0, weight=1)
    app.point_status_label = label(
        status_strip, "", "body_bold", color=t.ACCENT_TEXT, textvariable=app.point_status,
    )
    app.point_status_label.grid(row=0, column=0, sticky="w")
    app.cursor_coord_label = ctk.CTkLabel(
        status_strip,
        text=f"画像サイズ: {app.image_width} × {app.image_height} px  /  平面: {width_m:g} × {height_m:g} m",
        font=t.fonts().mono,
        text_color=t.TEXT_MUTED,
    )
    app.cursor_coord_label.grid(row=0, column=1, sticky="e")

    # ---- Action bar -----------------------------------------------------
    app.save_button = app.set_action_bar(
        primary=("行列を計算して保存  ✓", lambda: app._save_homography(width_m, height_m)),
        back=("←  ホモグラフィ設定へ戻る", app._build_homography_page),
    )

    app.point_canvas.render_image()
    app.point_canvas.redraw_points()


def _sync_zoom_from_canvas(app: IntegratedApp, zoom: float) -> None:
    if app.point_canvas is not None:
        pc = app.point_canvas
        app.fit_scale = pc.fit_scale
        app.zoom_factor = pc.zoom_factor
        app.display_scale = pc.display_scale
        app.display_width = pc.display_width
        app.display_height = pc.display_height
        app.display_scale_x = pc.display_scale_x
        app.display_scale_y = pc.display_scale_y
        app.photo = pc.photo
    app.zoom_status.set(f"表示倍率: {zoom * 100:.0f}%")


def _sync_cursor_status(
    app: IntegratedApp, pt: tuple[float, float] | None, hit: int | None
) -> None:
    if not hasattr(app, "cursor_coord_label") or not app.cursor_coord_label.winfo_exists():
        return
    width_m, height_m = app.current_plane_size
    base = f"画像: {app.image_width}×{app.image_height} px  /  平面: {width_m:g}×{height_m:g} m"
    if pt is None:
        app.cursor_coord_label.configure(text=base)
    elif hit is not None:
        app.cursor_coord_label.configure(
            text=f"点 {hit + 1}（{POINT_LABELS[hit]}）をドラッグで移動可能  |  ({pt[0]:.1f}, {pt[1]:.1f}) px"
        )
    else:
        app.cursor_coord_label.configure(text=f"カーソル座標: ({pt[0]:.1f}, {pt[1]:.1f}) px  |  {base}")


def _sync_point_ui(app: IntegratedApp) -> None:
    count = len(app.points)
    valid = is_quad_valid(app.points)

    for idx, chip in enumerate(getattr(app, "point_chips", [])):
        frame: ctk.CTkFrame = chip["frame"]  # type: ignore[assignment]
        badge: ctk.CTkLabel = chip["badge"]  # type: ignore[assignment]
        title_lbl: ctk.CTkLabel = chip["title"]  # type: ignore[assignment]
        coord_lbl: ctk.CTkLabel = chip["coord"]  # type: ignore[assignment]
        if not frame.winfo_exists():
            continue
        if idx < count:
            px, py = app.points[idx]
            tone_bg = t.SUCCESS_SOFT if valid else t.DANGER_SOFT
            tone_fg = t.SUCCESS if valid else t.DANGER
            border = t.SUCCESS_BORDER if valid else t.DANGER_BORDER
            frame.configure(fg_color=tone_bg, border_color=border, border_width=1)
            badge.configure(text="✓" if valid else "!", fg_color=tone_fg, text_color=t.TEXT_ON_ACCENT)
            title_lbl.configure(text_color=t.TEXT)
            coord_lbl.configure(text=f"({px:.1f}, {py:.1f}) px", text_color=t.TEXT_SECONDARY)
        elif idx == count:
            frame.configure(fg_color=t.ACCENT_SOFT, border_color=t.ACCENT, border_width=2)
            badge.configure(text=str(idx + 1), fg_color=t.ACCENT, text_color=t.TEXT_ON_ACCENT)
            title_lbl.configure(text_color=t.ACCENT_TEXT)
            coord_lbl.configure(text="次にここをクリック", text_color=t.ACCENT_TEXT)
        else:
            frame.configure(fg_color=t.SURFACE_ALT, border_color=t.BORDER, border_width=1)
            badge.configure(text=str(idx + 1), fg_color=t.DISABLED_BG, text_color=t.TEXT_MUTED)
            title_lbl.configure(text_color=t.TEXT_MUTED)
            coord_lbl.configure(text="未選択", text_color=t.TEXT_MUTED)

    if count < 4:
        app.point_status.set(
            f"次に【 {count + 1} / 4 : {POINT_LABELS[count]} 】を画像上でクリックしてください（打った点はドラッグで動かせます）"
        )
        status_color = t.ACCENT_TEXT
    elif not valid:
        app.point_status.set(
            "四角形が交差しているか面積が小さすぎます。点をドラッグして「左下 → 右下 → 右上 → 左上」の凸四角形に直してください。"
        )
        status_color = t.DANGER
    else:
        app.point_status.set(
            "四隅を選択しました。位置をドラッグで微調整するか、右下の［行列を計算して保存］を押してください。"
        )
        status_color = t.SUCCESS

    if hasattr(app, "point_status_label") and app.point_status_label.winfo_exists():
        app.point_status_label.configure(text_color=status_color)
    if hasattr(app, "save_button") and app.save_button.winfo_exists():
        set_button_enabled(app.save_button, count == 4 and valid)
    if hasattr(app, "undo_button") and app.undo_button.winfo_exists():
        set_button_enabled(app.undo_button, count > 0)
    if hasattr(app, "reset_button") and app.reset_button.winfo_exists():
        set_button_enabled(app.reset_button, count > 0)


def save_homography_action(app: IntegratedApp, width_m: float, height_m: float) -> None:
    assert app.image_path is not None
    assert app.output_path is not None
    assert app.camera_matrix is not None
    assert app.distortion is not None
    assert app.calibration_camera_matrix is not None
    if len(app.points) != 4:
        return
    try:
        matrix, source, destination = compute_homography(app.points, width_m, height_m)
    except HomographyPointError as exc:
        messagebox.showerror("点の位置を確認", str(exc), parent=app.root)
        return
    except HomographyComputeError as exc:
        messagebox.showerror("計算エラー", str(exc), parent=app.root)
        return

    metadata = build_metadata(
        image_size=(app.image_width, app.image_height),
        image_path=app.image_path,
        camera_matrix=app.camera_matrix,
        calibration_camera_matrix=app.calibration_camera_matrix,
        calibration_image_size=app.calibration_image_size,
        image_alignment=app.image_alignment,
        distortion=app.distortion,
        source=source,
        destination=destination,
        width_m=width_m,
        height_m=height_m,
    )
    try:
        save_homography(app.output_path, matrix, metadata)
    except OSError as exc:
        messagebox.showerror("保存エラー", str(exc), parent=app.root)
        return
    app.save_settings()
    app._build_completion_page(matrix)
    app.toast.show(
        f"{app.output_path.name} と .txt を保存しました",
        title="ホモグラフィ行列を保存しました",
        tone="success",
    )
