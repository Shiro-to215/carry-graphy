"""Step 2: Camera calibration (video analysis or existing file import)."""

from __future__ import annotations

import json
import subprocess
import threading
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from ...core.calibration import (
    ARUCO_DICTIONARIES,
    VIDEO_EXTENSIONS,
    build_worker_command,
    build_worker_config,
)
from ...core.calibration_io import (
    CALIBRATION_FILE_EXTENSIONS,
    load_calibration_file,
    read_calibration_result,
    write_calibration_summary,
)
from ...paths import APP_DIR
from .. import theme as t
from ..widgets import (
    Callout,
    Card,
    ChoiceField,
    Collapsible,
    FormField,
    PathField,
    button,
    label,
    set_button_enabled,
)

if TYPE_CHECKING:
    from ..app import IntegratedApp


def build_calibration_page(app: IntegratedApp) -> None:
    scroll = app.begin_page(
        step_index=1,
        step_badge="ステップ 2 / 4  ・  必須",
        title="カメラキャリブレーション",
        subtitle="ChArUcoボードを撮影した動画を解析してカメラ行列と歪み係数を計算します。過去のキャリブレーション結果（.npz / .npy）がある場合は、そのまま読み込んで次へ進めます。",
    )

    # ---- Quick-skip banner for existing calibration data ----------------
    skip_card = Card(
        scroll,
        title="既存のキャリブレーションデータをお持ちですか？",
        description="すでに計算済みの .npz または .npy があれば、動画の解析をスキップしてホモグラフィ設定へ進めます。下の枠にドラッグ＆ドロップすることもできます。",
        badge=("解析をスキップ可能", "info"),
    )
    skip_card.grid(row=0, column=0, sticky="ew", pady=(0, t.GAP))
    skip_card.body.columnconfigure(0, weight=1)

    skip_row = ctk.CTkFrame(skip_card.body, fg_color="transparent")
    skip_row.grid(row=0, column=0, sticky="ew")
    skip_row.columnconfigure(0, weight=1)

    app.calibration_import_field = PathField(
        skip_row,
        "既存のカメラ行列ファイル (.npz / .npy)",
        app.calibration_import_var,
        mode="open",
        dialog_title="カメラ行列を含むNPYまたはNPZを選択",
        filetypes=(
            ("NumPyファイル", "*.npz *.npy"),
            ("すべてのファイル", "*.*"),
        ),
        extensions=CALIBRATION_FILE_EXTENSIONS,
        memory=app.dir_memory,
        memory_key="calibration_import",
        dnd_enabled=True,
        placeholder="ここに calibration.npz / .npy をドロップすると自動で読み込みます",
        on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
        on_change=lambda path: import_calibration_path(app, Path(path)),
    )
    app.calibration_import_field.grid(row=0, column=0, sticky="ew", padx=(0, 12))

    app.load_calibration_button = button(
        skip_row,
        "ファイルを選んでホモグラフィへ  →",
        app._import_calibration_data,
        variant="soft",
        size="md",
    )
    app.load_calibration_button.grid(row=0, column=1, sticky="s", pady=(22, 0))

    # ---- Card 1: Video & output -----------------------------------------
    io_card = Card(
        scroll,
        title="解析する動画と保存先",
        description="動画ファイルを指定して［キャリブレーションを開始］を押すと、良質なフレームを自動選別してカメラ値を算出します。",
        number=1,
    )
    io_card.grid(row=1, column=0, sticky="ew", pady=(0, t.GAP))
    io_card.body.columnconfigure(0, weight=1)

    PathField(
        io_card.body,
        "キャリブレーション動画",
        app.calibration_vars["video"],
        mode="open",
        dialog_title="キャリブレーション動画を選択",
        filetypes=(
            ("動画", "*.mp4 *.avi *.mov *.mkv *.m4v"),
            ("すべてのファイル", "*.*"),
        ),
        extensions=VIDEO_EXTENSIONS,
        memory=app.dir_memory,
        memory_key="video",
        dnd_enabled=True,
        required=True,
        hint="対応形式: MP4 / AVI / MOV / MKV / M4V（ファイルを直接ドロップできます）",
        on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
    ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

    PathField(
        io_card.body,
        "結果の保存先フォルダ",
        app.calibration_vars["output"],
        mode="dir",
        dialog_title="キャリブレーション結果の保存先",
        memory=app.dir_memory,
        memory_key="calibration_output",
        dnd_enabled=True,
        hint="calibration.npz と calibration_summary.txt がこのフォルダに出力されます。",
        on_error=lambda msg: app.toast.show(msg, title="ドロップできません", tone="warning"),
    ).grid(row=1, column=0, sticky="ew")

    # ---- Card 2: Board parameters ---------------------------------------
    board_card = Card(
        scroll,
        title="ボード設定（撮影した実物に合わせてください）",
        description="長さの単位は統一してください（例: すべて mm）。ステップ1で設定した値が引き継がれています。",
        number=2,
    )
    board_card.grid(row=2, column=0, sticky="ew", pady=(0, t.GAP))
    for col in range(2):
        board_card.body.columnconfigure(col, weight=1, uniform="cal_board_cols")

    FormField(
        board_card.body, "横のマス数", app.calibration_vars["squares_x"], unit="マス",
    ).grid(row=0, column=0, sticky="ew", padx=(0, 12), pady=(0, 14))
    FormField(
        board_card.body, "縦のマス数", app.calibration_vars["squares_y"], unit="マス",
    ).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=(0, 14))
    FormField(
        board_card.body, "マスの一辺", app.calibration_vars["square_length"], unit="mm",
    ).grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(0, 14))
    FormField(
        board_card.body, "マーカーの一辺", app.calibration_vars["marker_length"], unit="mm",
    ).grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(0, 14))
    ChoiceField(
        board_card.body, "ArUco辞書", app.calibration_vars["dictionary"], ARUCO_DICTIONARIES, width=260,
    ).grid(row=2, column=0, columnspan=2, sticky="w")

    # ---- Collapsible: Advanced solver options ---------------------------
    advanced = Collapsible(
        scroll,
        "詳細パラメータ（フレーム選別・しきい値）",
        "通常は既定値のままで構いません。フレーム数が足りない場合やブレが多い動画のときだけ調整してください。",
        opened=app.advanced_calibration_open,
        on_toggle=lambda opened: setattr(app, "advanced_calibration_open", opened),
    )
    advanced.grid(row=3, column=0, sticky="ew", pady=(0, t.GAP))
    for col in range(3):
        advanced.body.columnconfigure(col, weight=1, uniform="adv_cols")

    FormField(
        advanced.body, "使用フレーム数", app.calibration_vars["top_n"],
        unit="枚", hint="スコア上位から選ぶ枚数（3以上）",
    ).grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=(0, 14))
    FormField(
        advanced.body, "解析間隔", app.calibration_vars["stride"],
        unit="フレームごと", hint="1で全フレーム、5で5枚おき",
    ).grid(row=0, column=1, sticky="ew", padx=10, pady=(0, 14))
    FormField(
        advanced.body, "ブレ判定しきい値", app.calibration_vars["blur_threshold"],
        hint="小さいほどブレた画像も許容（0以上）",
    ).grid(row=0, column=2, sticky="ew", padx=(10, 0), pady=(0, 14))

    FormField(
        advanced.body, "最小コーナー数", app.calibration_vars["min_corners"],
        unit="個", hint="1フレームで必要な検出数（1以上）",
    ).grid(row=1, column=0, sticky="ew", padx=(0, 10))
    FormField(
        advanced.body, "最大解析フレーム数", app.calibration_vars["max_frames"],
        unit="枚", hint="0 で動画の最後まで解析",
    ).grid(row=1, column=1, sticky="ew", padx=10)
    FormField(
        advanced.body, "低カバレッジ判定", app.calibration_vars["low_coverage"],
        hint="0〜1（未満なら高次歪み k3 を固定）",
    ).grid(row=1, column=2, sticky="ew", padx=(10, 0))

    # ---- Card 3: Progress & log -----------------------------------------
    progress_card = Card(
        scroll,
        title="進捗状況とログ",
        number=3,
    )
    progress_card.grid(row=4, column=0, sticky="ew")
    progress_card.body.columnconfigure(0, weight=1)

    status_row = ctk.CTkFrame(progress_card.body, fg_color="transparent")
    status_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    status_row.columnconfigure(0, weight=1)
    label(
        status_row, app.status.get(), "body_bold", color=t.TEXT, textvariable=app.status,
    ).grid(row=0, column=0, sticky="w")
    app.percent_label = ctk.CTkLabel(
        status_row, text="0%", font=t.fonts().subtitle, text_color=t.ACCENT_TEXT,
    )
    app.percent_label.grid(row=0, column=1, sticky="e")

    app.progress_bar = ctk.CTkProgressBar(
        progress_card.body, height=14, corner_radius=7,
        fg_color=t.SURFACE_ALT, progress_color=t.ACCENT,
    )
    app.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    app.progress_bar.set(max(0.0, min(1.0, app.calibration_progress.get() / 100.0)))
    app.progress_mode = "determinate"

    label(
        progress_card.body, app.progress_detail.get(), "caption",
        color=t.TEXT_SECONDARY, textvariable=app.progress_detail,
    ).grid(row=2, column=0, sticky="ew", pady=(0, 12))

    app.log = ctk.CTkTextbox(
        progress_card.body,
        height=140,
        corner_radius=t.RADIUS_CONTROL,
        border_width=1,
        border_color=t.BORDER,
        fg_color=t.SURFACE_ALT,
        text_color=t.TEXT_SECONDARY,
        font=t.fonts().mono,
        wrap="word",
    )
    app.log.grid(row=3, column=0, sticky="ew")
    app.log.configure(state="disabled")

    # ---- Action bar -----------------------------------------------------
    app.run_button = app.set_action_bar(
        primary=("キャリブレーションを開始  ▶", app._start_calibration),
        back=("←  ボード作成へ戻る", app._build_board_page),
        secondary=[("既存データを読み込む…", app._import_calibration_data)],
    )
    app.calibration_back_button = app.action_back_button


def append_log(app: IntegratedApp, text: str) -> None:
    if not hasattr(app, "log") or not app.log.winfo_exists():
        return
    app.log.configure(state="normal")
    app.log.insert("end", text.rstrip() + "\n")
    app.log.see("end")
    app.log.configure(state="disabled")


def set_calibration_busy(app: IntegratedApp, busy: bool) -> None:
    app.is_calibrating = busy
    for attr in ("run_button", "calibration_back_button", "load_calibration_button", "action_secondary_0"):
        widget = getattr(app, attr, None)
        if widget is not None and widget.winfo_exists():
            set_button_enabled(widget, not busy)
    app.refresh_sidebar()


def start_calibration(app: IntegratedApp) -> None:
    raw = {key: var.get() for key, var in app.calibration_vars.items()}
    try:
        worker_config = build_worker_config(raw)
    except ValueError as exc:
        messagebox.showerror("入力エラー", str(exc), parent=app.root)
        return

    video = Path(str(worker_config["video"]))
    output = Path(str(worker_config["output"]))
    app.dir_memory.remember("video", video)
    app.dir_memory.remember("calibration_output", output)
    app.save_settings()

    command = build_worker_command(worker_config)
    set_calibration_busy(app, True)
    app.status.set("解析中です。動画の長さによって数分かかることがあります…")
    app.calibration_progress.set(0.0)
    app.progress_detail.set("準備中: 動画と解析設定を確認しています")
    if hasattr(app, "percent_label") and app.percent_label.winfo_exists():
        app.percent_label.configure(text="0%")
    if hasattr(app, "progress_bar") and app.progress_bar.winfo_exists():
        app.progress_bar.stop()
        app.progress_bar.configure(mode="determinate")
        app.progress_bar.set(0.0)
        app.progress_mode = "determinate"
    append_log(app, f"統合アプリ内のキャリブレーションエンジンを実行: {video}")
    append_log(app, f"結果の保存先: {output}")
    threading.Thread(
        target=run_calibration_process,
        args=(app, command, output),
        daemon=True,
    ).start()


def run_calibration_process(app: IntegratedApp, command: list[str], output: Path) -> None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with subprocess.Popen(
            command,
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        ) as process:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if line.startswith("PROGRESS "):
                    try:
                        progress = json.loads(line[len("PROGRESS "):])
                    except json.JSONDecodeError:
                        app.messages.put(("calibration_output", line))
                    else:
                        app.messages.put(("calibration_progress", progress))
                elif line:
                    app.messages.put(("calibration_output", line))
            return_code = process.wait()
        app.messages.put(("calibration_done", (return_code, "", "", output)))
    except Exception as exc:
        app.messages.put(("calibration_error", str(exc)))


def update_calibration_progress(app: IntegratedApp, progress: dict[str, object]) -> None:
    overall_value = progress.get("overall")
    overall = float(overall_value) if overall_value is not None else app.calibration_progress.get()
    detail = str(progress.get("detail", "解析中です"))
    indeterminate = bool(progress.get("indeterminate", False))
    if hasattr(app, "progress_bar") and app.progress_bar.winfo_exists():
        if indeterminate and app.progress_mode != "indeterminate":
            app.progress_bar.configure(mode="indeterminate")
            app.progress_bar.start()
            app.progress_mode = "indeterminate"
        elif not indeterminate and app.progress_mode != "determinate":
            app.progress_bar.stop()
            app.progress_bar.configure(mode="determinate")
            app.progress_mode = "determinate"
        if not indeterminate:
            app.progress_bar.set(max(0.0, min(1.0, overall / 100.0)))
    app.calibration_progress.set(overall)
    if hasattr(app, "percent_label") and app.percent_label.winfo_exists():
        app.percent_label.configure(text="計算中…" if overall_value is None else f"{overall:.0f}%")
    phase = progress.get("phase")
    phase_text = f"（この工程 {float(phase):.0f}%）" if phase is not None else ""
    progress_text = "全体の進捗率は算出できません" if overall_value is None else f"全体の目安 {overall:.0f}%"
    app.progress_detail.set(f"{progress_text}　{detail} {phase_text}".rstrip())


def handle_calibration_done(
    app: IntegratedApp, return_code: int, stdout: str, stderr: str, output: Path
) -> None:
    if hasattr(app, "progress_bar") and app.progress_bar.winfo_exists():
        app.progress_bar.stop()
        app.progress_bar.configure(mode="determinate")
        app.progress_mode = "determinate"
    set_calibration_busy(app, False)
    output_text = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    append_log(app, output_text or "処理が終了しました。")
    if return_code != 0:
        app.status.set("処理に失敗しました。下のログを確認してください。")
        app.toast.show("キャリブレーションに失敗しました。ログを確認してください。", title="解析エラー", tone="danger")
        return
    try:
        app.calibration_data = read_calibration_result(Path(output))
        summary_path = write_calibration_summary(Path(output), app.calibration_data)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        app.status.set("結果を読み込めませんでした。ログを確認してください。")
        append_log(app, f"キャリブレーション結果の読み込みエラー: {exc}")
        return
    append_log(app, f"TXT出力: {summary_path}")
    rmse = app.calibration_data.get("rmse")
    rmse_msg = f"（再投影誤差 {float(rmse):.4f} px）" if isinstance(rmse, (int, float)) else ""
    app._build_homography_page()
    app.toast.show(
        f"カメラ行列を保存しました {rmse_msg}",
        title="キャリブレーション完了",
        tone="success",
    )


def import_calibration_data(app: IntegratedApp) -> None:
    initialdir = app.dir_memory.get("calibration_import", app.calibration_vars["output"].get())
    selected = filedialog.askopenfilename(
        title="カメラ行列を含むNPYまたはNPZを選択",
        initialdir=initialdir,
        filetypes=(
            ("NumPyファイル", "*.npz *.npy"),
            ("すべてのファイル", "*.*"),
        ),
        parent=app.root,
    )
    if not selected:
        return
    import_calibration_path(app, Path(selected).expanduser())


def import_calibration_path(app: IntegratedApp, path: Path) -> None:
    try:
        data = load_calibration_file(path, resolve_metadata=app._choose_calibration_metadata)
    except (OSError, EOFError, ValueError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        messagebox.showerror("読み込みエラー", str(exc), parent=app.root)
        return
    app.calibration_data = data
    app.calibration_import_var.set(str(path))
    app.dir_memory.remember("calibration_import", path)
    app.save_settings()
    app._build_homography_page()
    app.toast.show(
        f"{path.name}（{data['width']}×{data['height']} px）を読み込みました",
        title="キャリブレーションデータを読み込みました",
        tone="success",
    )
