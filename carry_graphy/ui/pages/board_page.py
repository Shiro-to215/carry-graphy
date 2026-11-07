"""Step 1: ChArUco board image creation (optional)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

import cv2
import customtkinter as ctk

from ...core.board import BoardInputError, BoardSizeError, parse_board_spec, save_board_image
from ...core.calibration import ARUCO_DICTIONARIES
from .. import theme as t
from ..imaging import fit_image, to_photo
from ..widgets import Callout, Card, ChoiceField, FormField, PathField, button, label

if TYPE_CHECKING:
    from ..app import IntegratedApp


def build_board_page(app: IntegratedApp) -> None:
    scroll = app.begin_page(
        step_index=0,
        step_badge="ステップ 1 / 4  ・  任意",
        title="ChArUcoボード画像の作成",
        subtitle="キャリブレーション撮影に使うボード画像を PNG で出力します。すでに印刷済みのボードをお持ちの場合は、そのまま右下の［キャリブレーションへ進む］を押してください。",
    )

    # ---- Card 1: Board geometry -----------------------------------------
    board_card = Card(
        scroll,
        title="ボードの寸法と辞書",
        description="ここで入力したマス数・長さ・ArUco辞書は、次のキャリブレーション設定にも自動で引き継がれます。",
        number=1,
    )
    board_card.grid(row=0, column=0, sticky="ew", pady=(0, t.GAP))
    for col in range(2):
        board_card.body.columnconfigure(col, weight=1, uniform="board_cols")

    FormField(
        board_card.body, "横のマス数", app.calibration_vars["squares_x"],
        unit="マス", hint="2以上の整数（例: 7）",
    ).grid(row=0, column=0, sticky="ew", padx=(0, 12), pady=(0, 14))
    FormField(
        board_card.body, "縦のマス数", app.calibration_vars["squares_y"],
        unit="マス", hint="2以上の整数（例: 5）",
    ).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=(0, 14))

    FormField(
        board_card.body, "マスの一辺", app.calibration_vars["square_length"],
        unit="mm", hint="黒白マスの実寸（例: 30）",
    ).grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(0, 14))
    FormField(
        board_card.body, "マーカーの一辺", app.calibration_vars["marker_length"],
        unit="mm", hint="マスの一辺より短い値（例: 22）",
    ).grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(0, 14))

    ChoiceField(
        board_card.body, "ArUco辞書", app.calibration_vars["dictionary"], ARUCO_DICTIONARIES,
        hint="特別な理由がなければ既定の DICT_4X4_50 のままで問題ありません。",
        width=260,
    ).grid(row=2, column=0, columnspan=2, sticky="w")

    # ---- Card 2: Print & output -----------------------------------------
    print_card = Card(
        scroll,
        title="印刷解像度と保存先",
        description="作成ボタンを押すと PNG 画像を書き出し、下にプレビューを表示します。",
        number=2,
    )
    print_card.grid(row=1, column=0, sticky="ew", pady=(0, t.GAP))
    print_card.body.columnconfigure(1, weight=1)

    FormField(
        print_card.body, "印刷解像度", app.board_vars["dpi"],
        unit="dpi", hint="72〜2400 dpi（通常は 300 dpi）", width=130,
    ).grid(row=0, column=0, sticky="nw", padx=(0, 20), pady=(0, 14))

    PathField(
        print_card.body,
        "出力ファイル (.png)",
        app.board_vars["output"],
        mode="save",
        dialog_title="ChArUcoボード画像の保存先",
        filetypes=(("PNG画像", "*.png"),),
        extensions=(".png",),
        default_extension=".png",
        memory=app.dir_memory,
        memory_key="board_output",
        dnd_enabled=True,
        hint="既定では「統合結果/charuco_board.png」に保存されます。",
        on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
    ).grid(row=0, column=1, sticky="ew", pady=(0, 14))

    Callout(
        print_card.body,
        "印刷時はプリンタ設定の「用紙サイズに合わせる（フィット）」をオフにし、実際のサイズ（100%）で印刷してください。縮尺が変わると実寸での計測結果がずれます。",
        title="印刷時の注意",
        tone="warning",
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))

    gen_row = ctk.CTkFrame(print_card.body, fg_color="transparent")
    gen_row.grid(row=2, column=0, columnspan=2, sticky="ew")
    button(gen_row, "ボード画像を作成して保存", app._create_board, variant="soft", size="md").pack(side="left")
    label(
        gen_row,
        app.board_status.get(),
        "body",
        color=t.TEXT_SECONDARY,
        textvariable=app.board_status,
    ).pack(side="left", padx=(16, 0))

    # ---- Card 3: Preview ------------------------------------------------
    preview_card = Card(scroll, title="プレビュー")
    preview_card.grid(row=2, column=0, sticky="ew")
    preview_card.body.columnconfigure(0, weight=1)

    preview_box = ctk.CTkFrame(
        preview_card.body, fg_color=t.SURFACE_ALT, corner_radius=t.RADIUS_CONTROL,
        border_width=1, border_color=t.BORDER, height=260,
    )
    preview_box.grid(row=0, column=0, sticky="ew")
    preview_box.grid_propagate(False)
    preview_box.columnconfigure(0, weight=1)
    preview_box.rowconfigure(0, weight=1)

    app.board_preview_label = tk.Label(
        preview_box,
        text="まだボード画像を作成していません。\n上の［ボード画像を作成して保存］を押すとここに表示されます。",
        bg=t.SURFACE_ALT,
        fg=t.TEXT_MUTED,
        font=t.fonts().canvas(12),
        justify="center",
    )
    app.board_preview_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
    if app.board_preview_photo is not None:
        app.board_preview_label.configure(image=app.board_preview_photo, text="")

    # ---- Action bar -----------------------------------------------------
    next_text = (
        "キャリブレーションへ進む  →"
        if app.board_generated_settings is not None
        else "キャリブレーションへ進む（ボード作成は任意）  →"
    )
    app.board_next_button = app.set_action_bar(
        primary=(next_text, app._build_calibration_page),
        secondary=[("ボード画像を作成", app._create_board)],
    )


def create_board(app: IntegratedApp) -> None:
    raw = {
        "squares_x": app.calibration_vars["squares_x"].get(),
        "squares_y": app.calibration_vars["squares_y"].get(),
        "square_length": app.calibration_vars["square_length"].get(),
        "marker_length": app.calibration_vars["marker_length"].get(),
        "dictionary": app.calibration_vars["dictionary"].get(),
        "dpi": app.board_vars["dpi"].get(),
    }
    try:
        spec = parse_board_spec(raw)
    except BoardInputError as exc:
        messagebox.showerror("入力エラー", str(exc), parent=app.root)
        return

    output = Path(app.board_vars["output"].get().strip()).expanduser()
    try:
        image = save_board_image(spec, output)
    except BoardInputError as exc:
        messagebox.showerror("入力エラー", str(exc), parent=app.root)
        return
    except BoardSizeError as exc:
        messagebox.showerror("画像サイズエラー", str(exc), parent=app.root)
        return
    except (cv2.error, OSError, ValueError) as exc:
        messagebox.showerror("ボード作成エラー", str(exc), parent=app.root)
        return

    app.dir_memory.remember("board_output", output)
    app.board_generated_settings = spec.as_tuple()
    preview = fit_image(image, 720, 236)
    photo = to_photo(preview, app.root)
    if photo is not None:
        app.board_preview_photo = photo
        if hasattr(app, "board_preview_label") and app.board_preview_label.winfo_exists():
            app.board_preview_label.configure(image=photo, text="")
    app.board_status.set(
        f"保存しました: {output}（{spec.squares_x}×{spec.squares_y} マス / {spec.square_length:g} mm / {spec.dpi} dpi）"
    )
    if hasattr(app, "board_next_button") and app.board_next_button.winfo_exists():
        app.board_next_button.configure(text="キャリブレーションへ進む  →")
    app.save_settings()
    app.toast.show(
        f"{output.name} を保存しました（{spec.squares_x}×{spec.squares_y} マス, {spec.dpi} dpi）",
        title="ボード画像を作成しました",
        tone="success",
    )
