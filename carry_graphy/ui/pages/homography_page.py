"""Step 3: Homography configuration and resolution-choice subpage."""

from __future__ import annotations

import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING

import cv2
import customtkinter as ctk
import numpy as np

from ...core.calibration_io import rmse_quality
from ...core.homography import (
    HOMOGRAPHY_FILE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SAME_RESOLUTION_ALIGNMENT,
    align_to_calibration_resolution,
    align_to_reference_resolution,
    load_homography_file,
    parse_plane_size,
    read_image_color,
)
from .. import theme as t
from ..widgets import (
    Callout,
    Card,
    Collapsible,
    FormField,
    OptionCard,
    PathField,
    StatTile,
    button,
    label,
)

if TYPE_CHECKING:
    from ..app import IntegratedApp


def _refresh_homography_summary(app: IntegratedApp) -> None:
    assert app.calibration_data is not None
    ui = getattr(app, "_homography_ui", None)
    if not ui:
        return
    camera_matrix: np.ndarray = app.calibration_data["camera_matrix"]  # type: ignore[assignment]
    distortion: np.ndarray = app.calibration_data["distortion"]  # type: ignore[assignment]
    width = app.calibration_data["width"]
    height = app.calibration_data["height"]
    rmse = app.calibration_data["rmse"]
    source = str(app.calibration_data.get("source") or "今回の解析結果")

    quality = rmse_quality(rmse)
    badge_text, badge_tone = {
        "good": ("精度 良好 (< 0.5 px)", "success"),
        "fair": ("精度 標準 (< 1.0 px)", "info"),
        "poor": ("誤差やや大 (≥ 1.0 px)", "warning"),
        None: ("既存データ", "info"),
    }[quality]

    summary_card: Card = ui["card"]
    if summary_card.badge_label is not None:
        bg, _, fg = t.TONES[badge_tone]
        summary_card.badge_label.configure(text=f" {badge_text} ", fg_color=bg, text_color=fg)

    res_tile: StatTile = ui["res_tile"]
    res_tile.set_content(f"{width} × {height} px")

    rmse_text = (
        f"{float(rmse):.4f} px"
        if isinstance(rmse, (int, float, np.number)) and np.isfinite(rmse)
        else "未記録（外部データ）"
    )
    rmse_tone = "success" if quality == "good" else ("warning" if quality == "poor" else None)
    rmse_tile: StatTile = ui["rmse_tile"]
    rmse_tile.set_content(rmse_text, tone=rmse_tone)

    focal_tile: StatTile = ui["focal_tile"]
    focal_tile.set_content(
        f"fx={camera_matrix[0, 0]:.1f}, fy={camera_matrix[1, 1]:.1f}",
        note=f"cx={camera_matrix[0, 2]:.1f}, cy={camera_matrix[1, 2]:.1f}",
    )

    detail: Collapsible = ui["detail"]
    if detail.subtitle_label is not None:
        detail.subtitle_label.configure(text=f"読込元: {Path(source).name}")

    detail_text = (
        f"読込元: {source}\n"
        f"K: fx={camera_matrix[0, 0]:.4f}, fy={camera_matrix[1, 1]:.4f}, "
        f"cx={camera_matrix[0, 2]:.4f}, cy={camera_matrix[1, 2]:.4f}\n"
        "歪み係数 k1, k2, p1, p2, k3: " + ", ".join(f"{value:.7g}" for value in distortion)
    )
    ui["detail_label"].configure(text=detail_text)


def build_homography_page(app: IntegratedApp, *, prewarm: bool = False) -> None:
    if not prewarm:
        assert app.calibration_data is not None
    scroll, created = app.use_page(
        "homography",
        step_index=2,
        step_badge="ステップ 3 / 4  ・  必須",
        title="ホモグラフィの設定",
        subtitle="歪みを補正する基準画像と、計測する平面の実際の幅・高さを指定します。すでにホモグラフィ行列（.npy / .npz）をお持ちの場合は、そのまま読み込んで完了画面へ進めます。",
        background_build=prewarm,
    )

    if created:
        # ---- Card 0: Loaded calibration summary -----------------------------
        summary_card = Card(
            scroll,
            title="適用中のキャリブレーション情報",
            badge=("既存データ", "info"),
        )
        summary_card.grid(row=0, column=0, sticky="ew", pady=(0, t.GAP))
        for col in range(3):
            summary_card.body.columnconfigure(col, weight=1, uniform="cal_stat")

        res_tile = StatTile(summary_card.body, "キャリブレーション解像度", "- × - px")
        res_tile.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        rmse_tile = StatTile(summary_card.body, "再投影誤差 (RMSE)", "-")
        rmse_tile.grid(row=0, column=1, sticky="ew", padx=8)

        focal_tile = StatTile(summary_card.body, "焦点距離 / 主点", "fx=-, fy=-", note="cx=-, cy=-")
        focal_tile.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        detail = Collapsible(
            summary_card.body,
            "カメラ行列 K・歪み係数・読込元の詳細を表示",
            "読込元: -",
            opened=False,
        )
        detail.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        detail_label = label(detail.body, "", "mono", color=t.TEXT_SECONDARY)
        detail_label.pack(anchor="w", fill="x")

        app._homography_ui = {  # type: ignore[attr-defined]
            "card": summary_card,
            "res_tile": res_tile,
            "rmse_tile": rmse_tile,
            "focal_tile": focal_tile,
            "detail": detail,
            "detail_label": detail_label,
        }

        # ---- Quick-skip for existing homography -----------------------------
        skip_card = Card(
            scroll,
            title="既存のホモグラフィ行列をお持ちですか？",
            description="すでに作成した homography_matrix.npy / .npz があれば、四隅の選択を省略して内容を確認できます。",
            badge=("選択をスキップ可能", "info"),
        )
        skip_card.grid(row=1, column=0, sticky="ew", pady=(0, t.GAP))
        skip_card.body.columnconfigure(0, weight=1)
        skip_row = ctk.CTkFrame(skip_card.body, fg_color="transparent")
        skip_row.grid(row=0, column=0, sticky="ew")
        skip_row.columnconfigure(0, weight=1)

        PathField(
            skip_row,
            "既存のホモグラフィ行列 (.npy / .npz)",
            app.homography_import_var,
            mode="open",
            dialog_title="既存のホモグラフィ行列を選択",
            filetypes=(
                ("NumPyファイル", "*.npy *.npz"),
                ("すべてのファイル", "*.*"),
            ),
            extensions=HOMOGRAPHY_FILE_EXTENSIONS,
            memory=app.dir_memory,
            memory_key="homography_import",
            dnd_enabled=True,
            placeholder="ここに homography_matrix.npy / .npz をドロップすると自動で読み込みます",
            on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
            on_change=lambda path: import_homography_path(app, Path(path)),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 12))

        button(
            skip_row,
            "行列を選んで完了へ  →",
            app._import_homography_data,
            variant="soft",
            size="md",
        ).grid(row=0, column=1, sticky="s", pady=(22, 0))

        # ---- Card 1: Reference image & plane dimensions ---------------------
        config_card = Card(
            scroll,
            title="基準画像と平面の実寸",
            description="基準画像を歪み補正したあと、次の画面で平面の四隅（左下 → 右下 → 右上 → 左上）をクリックします。",
            number=1,
        )
        config_card.grid(row=2, column=0, sticky="ew", pady=(0, t.GAP))
        config_card.body.columnconfigure(0, weight=1)

        PathField(
            config_card.body,
            "基準画像",
            app.homography_vars["image"],
            mode="open",
            dialog_title="ホモグラフィの基準画像を選択",
            filetypes=(
                ("画像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("すべてのファイル", "*.*"),
            ),
            extensions=IMAGE_EXTENSIONS,
            memory=app.dir_memory,
            memory_key="reference_image",
            dnd_enabled=True,
            required=True,
            hint="対応形式: PNG / JPG / BMP / TIFF（ドラッグ＆ドロップ対応）",
            on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 16))

        dim_frame = ctk.CTkFrame(config_card.body, fg_color="transparent")
        dim_frame.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        dim_frame.columnconfigure(0, weight=1, uniform="dim")
        dim_frame.columnconfigure(1, weight=1, uniform="dim")

        FormField(
            dim_frame, "平面の幅（横方向）", app.homography_vars["width"],
            unit="m", hint="左下 → 右下 の実寸距離（正の数値、例: 2.5）",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 12))
        FormField(
            dim_frame, "平面の高さ（縦方向）", app.homography_vars["height"],
            unit="m", hint="左下 → 左上 の実寸距離（正の数値、例: 1.8）",
        ).grid(row=0, column=1, sticky="ew", padx=(12, 0))

        Callout(
            config_card.body,
            "座標原点は「左下 (0, 0)」です。次の画面では 左下 → 右下 → 右上 → 左上（反時計回り）の順に4点を選択します。打った点はあとからドラッグで微調整できます。",
            title="座標系とクリック順",
            tone="info",
        ).grid(row=2, column=0, sticky="ew")

        # ---- Card 2: Output path --------------------------------------------
        out_card = Card(
            scroll,
            title="ホモグラフィ行列の保存先",
            description="3×3 行列と設定メタデータを1つにまとめた .npy と、人間が読める確認用の .txt を同じ場所に出力します。",
            number=2,
        )
        out_card.grid(row=3, column=0, sticky="ew")
        out_card.body.columnconfigure(0, weight=1)

        PathField(
            out_card.body,
            "保存先ファイル (.npy)",
            app.homography_vars["output"],
            mode="save",
            dialog_title="ホモグラフィ行列の保存先",
            filetypes=(("NumPy行列", "*.npy"),),
            extensions=(".npy",),
            default_extension=".npy",
            memory=app.dir_memory,
            memory_key="homography_output",
            dnd_enabled=True,
            hint="既定では「統合結果/homography/homography_matrix.npy」（および .txt）に保存されます。",
            on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
        ).grid(row=0, column=0, sticky="ew")

    if prewarm:
        return

    _refresh_homography_summary(app)

    # ---- Action bar -----------------------------------------------------
    app.set_action_bar(
        primary=("補正画像で四隅を選択へ進む  →", app._open_point_selection),
        back=("←  キャリブレーションへ戻る", app._build_calibration_page),
        secondary=[("既存の行列を読み込む…", app._import_homography_data)],
    )


def import_homography_data(app: IntegratedApp) -> None:
    initialdir = app.dir_memory.get("homography_import", app.homography_vars["output"].get())
    selected = filedialog.askopenfilename(
        title="既存のホモグラフィ行列を選択",
        initialdir=initialdir,
        filetypes=(
            ("NumPyファイル", "*.npy *.npz"),
            ("すべてのファイル", "*.*"),
        ),
        parent=app.root,
    )
    if not selected:
        return
    import_homography_path(app, Path(selected).expanduser())


def import_homography_path(app: IntegratedApp, path: Path) -> None:
    try:
        matrix, has_embedded_metadata = load_homography_file(path)
    except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        messagebox.showerror("読み込みエラー", str(exc), parent=app.root)
        return
    app.output_path = path
    app.homography_import_var.set(str(path))
    app.dir_memory.remember("homography_import", path)
    app.save_settings()
    app._build_completion_page(matrix, loaded=True, has_embedded_metadata=has_embedded_metadata)
    app.toast.show(f"{path.name} を読み込みました", title="ホモグラフィ行列を読み込みました", tone="success")


def open_point_selection(app: IntegratedApp) -> None:
    assert app.calibration_data is not None
    image_path = Path(app.homography_vars["image"].get().strip()).expanduser()
    if not app.homography_vars["image"].get().strip() or not image_path.is_file():
        messagebox.showerror("入力エラー", "基準画像を選択してください。", parent=app.root)
        return
    try:
        width_m, height_m = parse_plane_size(
            app.homography_vars["width"].get(),
            app.homography_vars["height"].get(),
        )
    except ValueError as exc:
        messagebox.showerror("入力エラー", str(exc), parent=app.root)
        return

    output_path = Path(app.homography_vars["output"].get().strip()).expanduser()
    if output_path.suffix.lower() != ".npy":
        messagebox.showerror("入力エラー", "保存先の拡張子は .npy にしてください。", parent=app.root)
        return
    image = read_image_color(image_path)
    if image is None:
        messagebox.showerror("画像エラー", f"画像を読み込めません: {image_path}", parent=app.root)
        return

    app.dir_memory.remember("reference_image", image_path)
    app.dir_memory.remember("homography_output", output_path)
    app.save_settings()

    image_height, image_width = image.shape[:2]
    expected_size = (int(app.calibration_data["width"]), int(app.calibration_data["height"]))
    if (image_width, image_height) != expected_size:
        build_resolution_choice_page(
            app, image, image_path, output_path, width_m, height_m,
            (image_width, image_height), expected_size,
        )
        return

    camera_matrix = app.calibration_data["camera_matrix"]
    distortion = app.calibration_data["distortion"]
    app._prepare_point_selection(
        image, image_path, output_path, width_m, height_m,
        camera_matrix, distortion,  # type: ignore[arg-type]
        image_alignment=dict(SAME_RESOLUTION_ALIGNMENT),
    )


def build_resolution_choice_page(
    app: IntegratedApp,
    image: np.ndarray,
    image_path: Path,
    output_path: Path,
    width_m: float,
    height_m: float,
    reference_size: tuple[int, int],
    calibration_size: tuple[int, int],
) -> None:
    ref_width, ref_height = reference_size
    cal_width, cal_height = calibration_size
    scale_to_cal = (cal_width / ref_width, cal_height / ref_height)
    scale_to_ref = (ref_width / cal_width, ref_height / cal_height)

    scroll = app.begin_page(
        step_index=2,
        step_badge="ステップ 3 / 4  ・  解像度の確認",
        title="基準画像とキャリブレーションの解像度が異なります",
        subtitle="どちらの画像サイズをホモグラフィ行列の入力座標の基準にするか選んでください。カードをクリックして選択し、右下のボタンで進みます。",
    )

    compare = Card(scroll, title="検出された画像サイズ")
    compare.grid(row=0, column=0, sticky="ew", pady=(0, t.GAP))
    compare.body.columnconfigure(0, weight=1, uniform="res")
    compare.body.columnconfigure(1, weight=1, uniform="res")
    StatTile(
        compare.body, "基準画像のサイズ", f"{ref_width} × {ref_height} px",
        note=image_path.name,
    ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
    StatTile(
        compare.body, "キャリブレーション時のサイズ", f"{cal_width} × {cal_height} px",
        tone="info",
    ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

    choice_var = tk.StringVar(value=app.resolution_choice_var.get() or "calibration")

    OptionCard(
        scroll,
        title="キャリブレーション画像の解像度に合わせる",
        description=(
            f"基準画像を {cal_width} × {cal_height} px にリサイズしてから歪み補正を行います。"
            "作成されるホモグラフィ行列の入力座標は、リサイズ後のキャリブレーション解像度になります。"
        ),
        details=f"リサイズ倍率: 横 {scale_to_cal[0]:.6g} 倍  /  縦 {scale_to_cal[1]:.6g} 倍",
        value="calibration",
        variable=choice_var,
        badge=("キャリブレーション基準", "info"),
    ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

    OptionCard(
        scroll,
        title="基準画像の解像度に合わせる（カメラ行列をスケーリング）",
        description=(
            f"基準画像のサイズ（{ref_width} × {ref_height} px）をそのまま保ち、"
            "カメラ行列の焦点距離・主点を画像サイズの比に合わせて調整します。"
            "同じカメラ・画角で撮影し、出力解像度だけが変わった画像に適しています。"
        ),
        details=f"カメラ行列スケーリング倍率: 横 {scale_to_ref[0]:.6g} 倍  /  縦 {scale_to_ref[1]:.6g} 倍",
        value="reference",
        variable=choice_var,
        badge=("元画像の解像度を維持", "neutral"),
    ).grid(row=2, column=0, sticky="ew", pady=(0, t.GAP))

    Callout(
        scroll,
        "キャリブレーション時と基準画像で撮影範囲（画角・トリミング）やカメラ自体が異なる場合、解像度を合わせるだけでは正しい補正値になりません。",
        title="ご注意",
        tone="warning",
    ).grid(row=3, column=0, sticky="ew")

    def proceed() -> None:
        app.resolution_choice_var.set(choice_var.get())
        if choice_var.get() == "reference":
            app._choose_reference_resolution(
                image, image_path, output_path, width_m, height_m, reference_size, calibration_size,
            )
        else:
            app._choose_calibration_resolution(
                image, image_path, output_path, width_m, height_m, reference_size, calibration_size,
            )

    app.set_action_bar(
        primary=("この解像度基準で四隅の選択へ進む  →", proceed),
        back=("←  画像・実寸の設定へ戻る", app._build_homography_page),
    )


def choose_calibration_resolution(
    app: IntegratedApp,
    image: np.ndarray,
    image_path: Path,
    output_path: Path,
    width_m: float,
    height_m: float,
    reference_size: tuple[int, int],
    calibration_size: tuple[int, int],
) -> None:
    assert app.calibration_data is not None
    resized, camera_matrix, alignment = align_to_calibration_resolution(
        image,
        app.calibration_data["camera_matrix"],  # type: ignore[arg-type]
        reference_size,
        calibration_size,
    )
    app._prepare_point_selection(
        resized, image_path, output_path, width_m, height_m,
        camera_matrix, app.calibration_data["distortion"], alignment,  # type: ignore[arg-type]
    )


def choose_reference_resolution(
    app: IntegratedApp,
    image: np.ndarray,
    image_path: Path,
    output_path: Path,
    width_m: float,
    height_m: float,
    reference_size: tuple[int, int],
    calibration_size: tuple[int, int],
) -> None:
    assert app.calibration_data is not None
    unchanged, camera_matrix, alignment = align_to_reference_resolution(
        image,
        app.calibration_data["camera_matrix"],  # type: ignore[arg-type]
        reference_size,
        calibration_size,
    )
    app._prepare_point_selection(
        unchanged, image_path, output_path, width_m, height_m,
        camera_matrix, app.calibration_data["distortion"], alignment,  # type: ignore[arg-type]
    )
