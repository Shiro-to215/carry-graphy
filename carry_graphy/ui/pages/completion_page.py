"""Step 5: Completion summary page."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk
import numpy as np

from ...core.homography import format_matrix, warp_full_image_for_preview
from .. import theme as t
from ..imaging import fit_image, to_photo
from ..widgets import Card, StatTile, button, label

if TYPE_CHECKING:
    from ..app import IntegratedApp


def build_completion_page(
    app: IntegratedApp,
    matrix: np.ndarray,
    *,
    loaded: bool = False,
    has_embedded_metadata: bool = False,
) -> None:
    assert app.output_path is not None
    app.last_completion = (matrix, loaded, has_embedded_metadata)

    scroll = app.begin_page(
        step_index=4,
        step_badge="完了",
        title="既存のホモグラフィ行列を読み込みました" if loaded else "ホモグラフィ行列の保存が完了しました",
        subtitle=(
            "読み込んだ行列の内容を確認できます。設定画面へ戻って作り直すこともできます。"
            if loaded
            else "カメラ歪み補正後のピクセル座標から平面上のメートル座標へ変換する 3×3 行列を出力しました。"
        ),
    )

    # ---- Banner ---------------------------------------------------------
    banner = ctk.CTkFrame(
        scroll, fg_color=t.SUCCESS_SOFT, corner_radius=t.RADIUS_CARD,
        border_width=1, border_color=t.SUCCESS_BORDER,
    )
    banner.grid(row=0, column=0, sticky="ew", pady=(0, t.GAP))
    banner.columnconfigure(1, weight=1)
    ctk.CTkLabel(
        banner, text="✓", width=44, height=44, corner_radius=22,
        fg_color=t.SUCCESS, text_color=t.SURFACE, font=t.fonts().title,
    ).grid(row=0, column=0, rowspan=2, padx=(20, 14), pady=18)
    label(
        banner,
        "すべての工程が完了しました" if not loaded else "ホモグラフィ行列を確認しています",
        "title",
        color=t.SUCCESS,
    ).grid(row=0, column=1, sticky="sw", padx=(0, 16), pady=(16, 0))
    label(
        banner,
        f"保存場所: {app.output_path.parent}",
        "body",
        color=t.TEXT_SECONDARY,
    ).grid(row=1, column=1, sticky="nw", padx=(0, 16), pady=(2, 16))
    button(
        banner,
        "保存フォルダを開く",
        lambda: _open_in_explorer(app, app.output_path.parent),
        variant="secondary",
        size="sm",
    ).grid(row=0, column=2, rowspan=2, padx=(0, 20), pady=18)

    current_row = 1

    # ---- Card 0: Transformed Image Preview (if available) ---------------
    if app.undistorted_image is not None and len(app.points) == 4 and not loaded:
        w_m, h_m = app.current_plane_size
        warped = warp_full_image_for_preview(app.undistorted_image, app.points, w_m, h_m)
        if warped is not None and warped.size > 0:
            preview_card = Card(
                scroll,
                title="ホモグラフィ変換プレビュー",
                description="選択した四隅の実寸比率に基づき、画像全体を正面視点へ透視変換したプレビューです（ファイル出力はされません）。",
            )
            preview_card.grid(row=current_row, column=0, sticky="ew", pady=(0, t.GAP))
            preview_card.body.columnconfigure(0, weight=1)

            preview = fit_image(warped, 880, 360)
            photo = to_photo(preview, app.root)
            if photo is not None:
                app.homography_preview_photo = photo
                box = ctk.CTkFrame(
                    preview_card.body,
                    fg_color=t.SURFACE_ALT,
                    corner_radius=t.RADIUS_CONTROL,
                    border_width=1,
                    border_color=t.BORDER,
                )
                box.grid(row=0, column=0, sticky="ew")
                box.columnconfigure(0, weight=1)

                img_lbl = tk.Label(box, image=photo, bg=t.SURFACE_ALT)
                img_lbl.grid(row=0, column=0, padx=12, pady=12)

            current_row += 1

    # ---- Card 1: Output files -------------------------------------------
    files_card = Card(scroll, title="出力ファイルと設定情報")
    files_card.grid(row=current_row, column=0, sticky="ew", pady=(0, t.GAP))
    files_card.body.columnconfigure(0, weight=1)
    current_row += 1

    path_label = "読込元ファイル (.npy / .npz)" if loaded else "ホモグラフィ行列ファイル (.npy)"
    StatTile(
        files_card.body, path_label, str(app.output_path),
        note=(
            "設定情報（カメラ行列・歪み係数・四隅の座標・平面サイズ）が NPY 内に同梱されています。"
            if (not loaded or has_embedded_metadata)
            else "3×3 行列のみのファイルです。"
        ),
    ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

    txt_path = app.output_path.with_suffix(".txt")
    if not loaded or txt_path.is_file():
        StatTile(
            files_card.body, "確認用テキスト (.txt)", str(txt_path),
            note="行列と変換条件・座標対応をテキストエディタで確認できます。",
        ).grid(row=1, column=0, sticky="ew")

    # ---- Card 2: Matrix display -----------------------------------------
    matrix_card = Card(
        scroll,
        title="変換行列 H（3 × 3）",
        description="補正後画像のピクセル座標 (x, y, 1)ᵀ に左から掛けて正規化すると、左下原点の平面メートル座標 (X, Y) が得られます。",
    )
    matrix_card.grid(row=current_row, column=0, sticky="ew")
    matrix_card.body.columnconfigure(0, weight=1)

    matrix_text = format_matrix(matrix)
    box = ctk.CTkTextbox(
        matrix_card.body,
        height=110,
        corner_radius=t.RADIUS_CONTROL,
        border_width=1,
        border_color=t.BORDER,
        fg_color=t.SURFACE_ALT,
        text_color=t.TEXT,
        font=t.fonts().mono,
        activate_scrollbars=False,
    )
    box.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    box.insert("1.0", matrix_text)
    box.configure(state="disabled")

    copy_row = ctk.CTkFrame(matrix_card.body, fg_color="transparent")
    copy_row.grid(row=1, column=0, sticky="ew")
    button(
        copy_row, "行列テキストをコピー", lambda: _copy_matrix(app, matrix_text),
        variant="soft", size="sm",
    ).pack(side="left")

    # ---- Action bar -----------------------------------------------------
    secondary = [("キャリブレーションへ戻る", app._build_calibration_page)]
    if app.undistorted_image is not None and not loaded:
        w_m, h_m = app.current_plane_size
        secondary.insert(0, ("四隅の選択をやり直す", lambda: app._build_point_page(w_m, h_m)))
    app.set_action_bar(
        primary=("保存フォルダを開く", lambda: _open_in_explorer(app, app.output_path.parent)),
        back=("←  ホモグラフィ設定へ戻る", app._build_homography_page),
        secondary=secondary,
    )


def _copy_matrix(app: IntegratedApp, matrix_text: str) -> None:
    app.root.clipboard_clear()
    app.root.clipboard_append(matrix_text)
    app.toast.show("3×3 行列をクリップボードにコピーしました", title="コピー完了", tone="info")


def _open_in_explorer(app: IntegratedApp, folder: Path) -> None:
    target = folder if folder.exists() else folder.parent
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        app.toast.show(str(exc), title="フォルダを開けませんでした", tone="warning")
