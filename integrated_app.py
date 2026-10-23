#!/usr/bin/env python3
"""Single-window workflow for camera calibration and homography creation."""

from __future__ import annotations

import base64
import json
import math
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = APP_DIR / "統合結果"


@dataclass
class _CalibrationCandidate:
    frame_index: int
    sharpness: float
    coverage: float
    boundary_count: int
    boundary_fraction: float
    score: float
    corners: np.ndarray
    ids: np.ndarray


_CalibrationProgressCallback = Callable[[dict[str, object]], None]


def _emit_calibration_progress(
    callback: _CalibrationProgressCallback | None,
    stage: str,
    overall: float | None,
    detail: str,
    *,
    indeterminate: bool = False,
    phase: float | None = None,
) -> None:
    if callback is None:
        return
    callback({
        "stage": stage,
        "overall": max(0.0, min(100.0, overall)) if overall is not None else None,
        "phase": phase,
        "detail": detail,
        "indeterminate": indeterminate,
    })


def _calibration_analyze_frame_quality(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _calibration_detect_charuco(
    frame: np.ndarray, board: Any, min_corners: int
) -> tuple[np.ndarray, np.ndarray] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
        return None
    return (
        np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1),
    )


def _calibration_spatial_coverage(
    corners: np.ndarray, image_size: tuple[int, int], boundary_fraction: float = 0.15
) -> tuple[float, int, float]:
    width, height = image_size
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 0.0, 0, 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    coverage = min(1.0, max(0.0, hull_area / float(width * height)))
    x_band, y_band = width * boundary_fraction, height * boundary_fraction
    on_boundary = (
        (points[:, 0] <= x_band) | (points[:, 0] >= width - x_band)
        | (points[:, 1] <= y_band) | (points[:, 1] >= height - y_band)
    )
    count = int(np.count_nonzero(on_boundary))
    return coverage, count, count / len(points)


def _calibration_frame_score(sharpness: float, coverage: float, boundary_fraction: float) -> float:
    sharp_term = min(1.0, math.log1p(max(0.0, sharpness)) / math.log1p(1000.0))
    return coverage + 0.15 * sharp_term + 0.10 * boundary_fraction


def _calibration_collect_candidates(
    video_path: Path,
    board: Any,
    *,
    blur_threshold: float,
    min_corners: int,
    stride: int,
    limit: int,
    boundary_fraction: float,
    progress_callback: _CalibrationProgressCallback | None,
) -> tuple[list[_CalibrationCandidate], tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    candidates: list[_CalibrationCandidate] = []
    frame_index = 0
    image_size: tuple[int, int] | None = None
    reported_percent = -1
    last_reported = time.monotonic()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_total = min(frame_count, limit) if frame_count > 0 and limit > 0 else (limit or frame_count)
    try:
        while limit <= 0 or frame_index < limit:
            ok, frame = cap.read()
            if not ok:
                break
            current = frame_index
            frame_index += 1
            now = time.monotonic()
            percent = int(100 * frame_index / scan_total) if scan_total > 0 else -1
            if percent != reported_percent or now - last_reported >= 0.5:
                frame_detail = (
                    f"動画フレームを解析中: {frame_index:,} / {scan_total:,}"
                    if scan_total else f"動画フレームを解析中: {frame_index:,}（総フレーム数不明）"
                )
                fraction = frame_index / scan_total if scan_total > 0 else None
                _emit_calibration_progress(
                    progress_callback,
                    "scan",
                    98 * fraction if fraction is not None else None,
                    frame_detail,
                    indeterminate=fraction is None,
                    phase=100 * fraction if fraction is not None else None,
                )
                reported_percent = percent
                last_reported = now
            if current % stride:
                continue
            height, width = frame.shape[:2]
            image_size = (width, height)
            sharpness = _calibration_analyze_frame_quality(frame)
            if sharpness < blur_threshold:
                continue
            detected = _calibration_detect_charuco(frame, board, min_corners)
            if detected is None:
                continue
            corners, ids = detected
            coverage, boundary_count, boundary_share = _calibration_spatial_coverage(
                corners, image_size, boundary_fraction
            )
            candidates.append(_CalibrationCandidate(
                current,
                sharpness,
                coverage,
                boundary_count,
                boundary_share,
                _calibration_frame_score(sharpness, coverage, boundary_share),
                corners,
                ids,
            ))
    finally:
        cap.release()
    if scan_total > 0:
        _emit_calibration_progress(
            progress_callback,
            "scan",
            98,
            f"動画フレームの確認が完了しました: {frame_index:,} / {scan_total:,}",
            phase=100,
        )
    if image_size is None:
        raise RuntimeError("The video contains no readable frames")
    return candidates, image_size


def _calibration_is_monotonic(
    camera_matrix: np.ndarray, distortion: np.ndarray, image_size: tuple[int, int], samples: int = 2048
) -> tuple[bool, float]:
    width, height = image_size
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    corners = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=np.float64,
    )
    radii = np.sqrt(((corners[:, 0] - cx) / fx) ** 2 + ((corners[:, 1] - cy) / fy) ** 2)
    r = np.linspace(0.0, float(np.max(radii)), samples)
    coeffs = np.zeros(5, dtype=np.float64)
    flat = np.asarray(distortion, dtype=np.float64).ravel()
    coeffs[:min(5, len(flat))] = flat[:5]
    k1, k2, _, _, k3 = coeffs
    derivative = 1.0 + 3.0 * k1 * r**2 + 5.0 * k2 * r**4 + 7.0 * k3 * r**6
    minimum = float(np.min(derivative))
    return minimum > 0.0, minimum


def _calibration_run_solver(
    selected: list[_CalibrationCandidate], board: Any, image_size: tuple[int, int], flags: int
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    board_corners = (
        board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    )
    object_points, image_points = [], []
    for candidate in selected:
        ids = candidate.ids.ravel()
        object_points.append(np.asarray(board_corners, dtype=np.float32)[ids].reshape(-1, 1, 3))
        image_points.append(candidate.corners.astype(np.float32))
    return cv2.calibrateCamera(object_points, image_points, image_size, None, None, flags=flags)


def _calibration_worker(config: dict[str, object], progress_callback: _CalibrationProgressCallback) -> dict[str, object]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is missing; install opencv-contrib-python")
    dictionary_name = str(config["dictionary"])
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    squares_x, squares_y = int(config["squares_x"]), int(config["squares_y"])
    square_length, marker_length = float(config["square_length"]), float(config["marker_length"])
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    else:
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)

    _emit_calibration_progress(progress_callback, "prepare", 0, "動画情報を確認し、解析を準備しています")
    candidates, image_size = _calibration_collect_candidates(
        Path(str(config["video"])),
        board,
        blur_threshold=float(config["blur_threshold"]),
        min_corners=int(config["min_corners"]),
        stride=int(config["stride"]),
        limit=int(config["max_frames"]),
        boundary_fraction=0.15,
        progress_callback=progress_callback,
    )
    _emit_calibration_progress(
        progress_callback, "select", 98,
        f"検出した候補 {len(candidates):,} 件から使用フレームを選択しています", phase=0,
    )
    top_n = int(config["top_n"])
    selected = sorted(
        candidates,
        key=lambda item: (item.score, item.coverage, item.sharpness),
        reverse=True,
    )[:top_n]
    _emit_calibration_progress(
        progress_callback, "select", 98.3,
        f"使用フレーム {len(selected):,} 件を決定しました", phase=100,
    )
    if len(selected) < 3:
        raise RuntimeError(f"Only {len(selected)} usable frames found; at least 3 are needed")

    max_coverage = max(item.coverage for item in selected)
    flags = cv2.CALIB_FIX_K3 if max_coverage < float(config["low_coverage"]) else 0
    _emit_calibration_progress(
        progress_callback, "calibration", 98.3,
        f"{len(selected):,} フレームからカメラ行列を計算しています", indeterminate=True,
    )
    rms, camera_matrix, distortion, rvecs, tvecs = _calibration_run_solver(selected, board, image_size, flags)
    _emit_calibration_progress(progress_callback, "calibration", 99, "カメラ行列の計算が完了しました。歪みモデルを確認しています")
    monotonic, min_derivative = _calibration_is_monotonic(camera_matrix, distortion, image_size)
    fallback_used = False
    if not monotonic:
        fallback_used = True
        _emit_calibration_progress(
            progress_callback, "fallback", 99,
            "歪みモデルの条件に合わせ、制約を追加して再計算しています", indeterminate=True,
        )
        flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        rms, camera_matrix, distortion, rvecs, tvecs = _calibration_run_solver(
            selected, board, image_size, flags
        )
        monotonic, min_derivative = _calibration_is_monotonic(camera_matrix, distortion, image_size)
        _emit_calibration_progress(progress_callback, "fallback", 99.5, "再計算と歪みモデルの確認が完了しました")

    folding_check = {
        "radial_monotonic": bool(monotonic),
        "minimum_derivative": min_derivative,
        "fallback_used": fallback_used,
    }
    output = Path(str(config["output"]))
    _emit_calibration_progress(progress_callback, "save", 99.8, "キャリブレーション結果をファイルに保存しています")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "calibration.npz",
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(distortion, dtype=np.float64).reshape(1, -1),
        image_size=np.asarray(image_size, dtype=np.int32),
        rmse=np.asarray(float(rms), dtype=np.float64),
    )
    (output / "calibration.json").unlink(missing_ok=True)
    _emit_calibration_progress(progress_callback, "complete", 100, "キャリブレーション処理が完了しました", phase=100)
    if not monotonic:
        raise RuntimeError("Fallback calibration still failed the radial monotonicity check")
    return {
        "rmse": float(rms),
        "selected_frames": len(selected),
        "max_coverage": max_coverage,
        "folding_check": folding_check,
        "output": str(output.resolve()),
    }


def _calibration_worker_entry(config_text: str) -> int:
    try:
        config = json.loads(config_text)
        if not isinstance(config, dict):
            raise ValueError("解析設定の形式が不正です")

        def write_progress(progress: dict[str, object]) -> None:
            print("PROGRESS " + json.dumps(progress, ensure_ascii=False), file=sys.stderr, flush=True)

        summary = _calibration_worker(config, write_progress)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


class IntegratedApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("カメラキャリブレーション・ホモグラフィ統合アプリ")
        self.root.minsize(900, 760)
        self.root.geometry("1040x880")

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
        self.image_width = 0
        self.image_height = 0
        self.display_width = 0
        self.display_height = 0
        self.display_scale = 1.0
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        self.fit_scale = 1.0
        self.zoom_factor = 1.0
        self.viewport_width = 950
        self.viewport_height = 500
        self.photo: tk.PhotoImage | None = None
        self.board_preview_photo: tk.PhotoImage | None = None
        self.board_generated_settings: tuple[int, int, float, float, str, int] | None = None

        self.calibration_vars = {
            "video": tk.StringVar(),
            "output": tk.StringVar(value=str(DEFAULT_RESULTS_DIR / "calibration")),
            "squares_x": tk.StringVar(value="7"),
            "squares_y": tk.StringVar(value="5"),
            "square_length": tk.StringVar(value="30"),
            "marker_length": tk.StringVar(value="22"),
            "dictionary": tk.StringVar(value="DICT_4X4_50"),
            "top_n": tk.StringVar(value="30"),
            "stride": tk.StringVar(value="5"),
            "blur_threshold": tk.StringVar(value="30"),
            "min_corners": tk.StringVar(value="8"),
            "max_frames": tk.StringVar(value="0"),
            "low_coverage": tk.StringVar(value="0.40"),
        }
        self.board_vars = {
            "dpi": tk.StringVar(value="300"),
            "output": tk.StringVar(value=str(DEFAULT_RESULTS_DIR / "charuco_board.png")),
        }
        self.homography_vars = {
            "image": tk.StringVar(),
            "width": tk.StringVar(),
            "height": tk.StringVar(),
            "output": tk.StringVar(value=str(DEFAULT_RESULTS_DIR / "homography_matrix.npy")),
        }

        self.header = ttk.Frame(root, padding=(18, 14, 18, 10))
        self.header.pack(fill="x")
        ttk.Label(
            self.header,
            text="カメラキャリブレーション ＋ ホモグラフィ変換",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(anchor="w")
        self.step_label = tk.StringVar(value="ステップ 1 / 4　ボード作成（任意）")
        ttk.Label(self.header, textvariable=self.step_label).pack(anchor="w", pady=(5, 0))

        self.content = ttk.Frame(root, padding=(18, 4, 18, 18))
        self.content.pack(fill="both", expand=True)
        self._build_board_page()
        self.root.after(100, self._poll_messages)

    def _clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        for row in range(10):
            self.content.rowconfigure(row, weight=0)
        for column in range(5):
            self.content.columnconfigure(column, weight=0)

    @staticmethod
    def _add_field(
        parent: ttk.Frame | ttk.LabelFrame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
        width: int = 12,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=column + 1, sticky="ew", padx=(8, 14), pady=3
        )

    def _build_board_page(self) -> None:
        self._clear_content()
        self.step_label.set("ステップ 1 / 4　ボード作成（任意）")
        self.content.columnconfigure(1, weight=1)

        ttk.Label(
            self.content,
            text="ChArUcoボード画像を作成できます。この手順は任意です。すでにボードをお持ちの場合は、作成せずにキャリブレーションへ進めます。",
            wraplength=920,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        board = ttk.LabelFrame(self.content, text="ボード設定", padding=10)
        board.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        for column in range(4):
            board.columnconfigure(column, weight=1)
        self._add_field(board, "横のマス数", self.calibration_vars["squares_x"], 0, 0)
        self._add_field(board, "縦のマス数", self.calibration_vars["squares_y"], 0, 2)
        self._add_field(board, "マスの一辺 (mm)", self.calibration_vars["square_length"], 1, 0)
        self._add_field(board, "マーカーの一辺 (mm)", self.calibration_vars["marker_length"], 1, 2)
        ttk.Label(board, text="マス数と長さは、後のキャリブレーション設定にも引き継がれます。").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(3, 5)
        )
        ttk.Label(board, text="ArUco辞書").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(
            board,
            textvariable=self.calibration_vars["dictionary"],
            state="readonly",
            values=(
                "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100",
                "DICT_6X6_50", "DICT_6X6_100", "DICT_7X7_50", "DICT_7X7_100",
            ),
        ).grid(row=3, column=1, columnspan=3, sticky="ew", padx=8)

        print_settings = ttk.LabelFrame(self.content, text="印刷サイズと保存先", padding=10)
        print_settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        print_settings.columnconfigure(1, weight=1)
        self._add_field(print_settings, "解像度 (dpi)", self.board_vars["dpi"], 0, 0)
        ttk.Label(print_settings, text="出力ファイル (.png)").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(print_settings, textvariable=self.board_vars["output"]).grid(
            row=1, column=1, sticky="ew", padx=8
        )
        ttk.Button(print_settings, text="保存先…", command=self._choose_board_output).grid(row=1, column=2)
        ttk.Label(
            print_settings,
            text="印刷時は「用紙サイズに合わせる」をオフにして、実際のサイズ（100%）で印刷してください。",
            wraplength=850,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        actions = ttk.Frame(self.content)
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Button(actions, text="ボード画像を作成", command=self._create_board).pack(side="left")
        self.board_next_button = ttk.Button(
            actions,
            text="キャリブレーションへ進む（ボード作成は任意）",
            command=self._build_calibration_page,
        )
        self.board_next_button.pack(side="right")

        self.board_status = tk.StringVar(value="ボードを作成するか、スキップして次へ進んでください。")
        ttk.Label(self.content, textvariable=self.board_status, wraplength=920).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 4)
        )
        self.board_preview_label = ttk.Label(
            self.content,
            text="作成したボード画像のプレビューがここに表示されます。",
            anchor="center",
        )
        self.board_preview_label.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        self.content.rowconfigure(5, weight=1)

    def _import_calibration_data(self) -> None:
        selected = filedialog.askopenfilename(
            title="カメラ行列を含むNPYまたはNPZを選択",
            filetypes=(
                ("NumPyファイル", "*.npz *.npy"),
                ("すべてのファイル", "*.*"),
            ),
        )
        if not selected:
            return
        try:
            self.calibration_data = self._load_calibration_file(Path(selected).expanduser())
        except (OSError, EOFError, ValueError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            messagebox.showerror("読み込みエラー", str(exc), parent=self.root)
            return
        self._build_homography_page()

    def _load_calibration_file(self, path: Path) -> dict[str, object]:
        if path.suffix.lower() not in (".npz", ".npy"):
            raise ValueError("カメラ行列を含む .npz または .npy ファイルを選択してください。")

        metadata: dict[str, object] = {}
        metadata_path = path.with_suffix(".json")
        rmse: float | None = None

        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "camera_matrix" not in archive.files:
                    raise ValueError("NPZに camera_matrix がありません。カメラキャリブレーション結果を選択してください。")
                camera_matrix = np.asarray(archive["camera_matrix"], dtype=np.float64)
                distortion = None
                for key in ("distortion_coefficients", "distortion", "dist_coeffs"):
                    if key in archive.files:
                        distortion = np.asarray(archive[key], dtype=np.float64).ravel()
                        break
                image_size = np.asarray(archive["image_size"]).ravel() if "image_size" in archive.files else None
                if "rmse" in archive.files:
                    rmse_values = np.asarray(archive["rmse"], dtype=np.float64).ravel()
                    if rmse_values.size != 1 or not np.isfinite(rmse_values[0]):
                        raise ValueError("NPZのrmseは有限値1個である必要があります。")
                    rmse = float(rmse_values[0])
        else:
            camera_matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
            distortion = None
            image_size = None

        if distortion is None or image_size is None:
            metadata_path = self._choose_calibration_metadata(path, metadata_path)
            if metadata_path is None:
                raise ValueError(
                    "カメラの歪み係数とキャリブレーション解像度が必要です。"
                    "対応するJSONを選択するか、camera_matrix・distortion_coefficients・image_sizeを含むNPZを指定してください。"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"対応JSONの形式が不正です: {metadata_path}")
            if distortion is None:
                distortion = self._distortion_from_metadata(metadata)
            if image_size is None:
                image_size = self._image_size_from_metadata(metadata)
            if rmse is None and "rmse" in metadata:
                metadata_rmse = np.asarray(metadata["rmse"], dtype=np.float64).ravel()
                if metadata_rmse.size == 1 and np.isfinite(metadata_rmse[0]):
                    rmse = float(metadata_rmse[0])

        if distortion is None or image_size is None:
            raise ValueError("入力データに歪み係数または画像解像度がありません。")
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ValueError("カメラ行列は有限値の3×3配列である必要があります。")
        if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise ValueError("カメラ行列のfxとfyは正の値である必要があります。選択したNPYがカメラ行列か確認してください。")
        if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(
                "選択したNPYはカメラマトリックスではないようです。"
                "ホモグラフィ行列ではなく、キャリブレーションのカメラマトリックスを選択してください。"
            )
        if distortion.size < 5 or not np.isfinite(distortion[:5]).all():
            raise ValueError("歪み係数は有限値のk1, k2, p1, p2, k3が必要です。")
        width, height = int(image_size[0]), int(image_size[1])
        if width < 1 or height < 1:
            raise ValueError("キャリブレーション画像解像度が不正です。")
        return {
            "camera_matrix": camera_matrix,
            "distortion": np.asarray(distortion[:5], dtype=np.float64),
            "width": width,
            "height": height,
            "rmse": rmse,
            "source": str(path),
        }

    def _choose_calibration_metadata(self, data_path: Path, default_path: Path) -> Path | None:
        if default_path.is_file():
            return default_path
        selected = filedialog.askopenfilename(
            title="対応するキャリブレーションJSONを選択",
            filetypes=(("JSON", "*.json"), ("すべてのファイル", "*.*")),
            initialdir=str(data_path.parent),
        )
        return Path(selected).expanduser() if selected else None

    @staticmethod
    def _distortion_from_metadata(metadata: dict[str, object]) -> np.ndarray | None:
        values = metadata.get("distortion_coefficients")
        if isinstance(values, dict):
            try:
                return np.asarray([values[key] for key in ("k1", "k2", "p1", "p2", "k3")], dtype=np.float64)
            except KeyError:
                return None
        for key in ("distortion_coefficients_k1_k2_p1_p2_k3", "opencv_distortion_vector"):
            if key in metadata:
                return np.asarray(metadata[key], dtype=np.float64).ravel()
        return None

    @staticmethod
    def _image_size_from_metadata(metadata: dict[str, object]) -> np.ndarray | None:
        for key in ("calibration_image_size", "image_size"):
            if key not in metadata:
                continue
            value = metadata[key]
            if isinstance(value, dict) and "width" in value and "height" in value:
                return np.asarray([value["width"], value["height"]], dtype=np.int64)
            size = np.asarray(value).ravel()
            if size.size >= 2:
                return size[:2]
        return None

    def _choose_board_output(self) -> None:
        current = Path(self.board_vars["output"].get()).expanduser()
        path = filedialog.asksaveasfilename(
            title="ChArUcoボード画像の保存先",
            defaultextension=".png",
            initialdir=str(current.parent),
            initialfile=current.name,
            filetypes=(("PNG画像", "*.png"),),
        )
        if path:
            self.board_vars["output"].set(path)

    def _create_board(self) -> None:
        try:
            squares_x = int(self.calibration_vars["squares_x"].get())
            squares_y = int(self.calibration_vars["squares_y"].get())
            square_length = float(self.calibration_vars["square_length"].get())
            marker_length = float(self.calibration_vars["marker_length"].get())
            dpi = int(self.board_vars["dpi"].get())
            if not np.isfinite([square_length, marker_length]).all():
                raise ValueError("マスとマーカーの長さには有限の数値を入力してください。")
            if squares_x < 2 or squares_y < 2:
                raise ValueError("ボードのマス数は横・縦ともに2以上にしてください。")
            if not 0 < marker_length < square_length:
                raise ValueError("マーカーの一辺は、マスの一辺より短くしてください。")
            if not 72 <= dpi <= 2400:
                raise ValueError("解像度は72〜2400 dpiで指定してください。")
        except ValueError as exc:
            messagebox.showerror("入力エラー", str(exc) or "ボード設定を確認してください。", parent=self.root)
            return

        output = Path(self.board_vars["output"].get().strip()).expanduser()
        if output.suffix.lower() != ".png":
            messagebox.showerror("入力エラー", "ボード画像の保存先は .png にしてください。", parent=self.root)
            return

        pixel_width = squares_x * square_length * dpi / 25.4
        pixel_height = squares_y * square_length * dpi / 25.4
        if (
            not np.isfinite([pixel_width, pixel_height]).all()
            or pixel_width < 1
            or pixel_height < 1
            or pixel_width * pixel_height > 60_000_000
        ):
            messagebox.showerror(
                "画像サイズエラー",
                "指定した設定ではボード画像が大きすぎます。dpiまたはボード寸法を小さくしてください。",
                parent=self.root,
            )
            return
        image_width = int(pixel_width)
        image_height = int(pixel_height)

        dictionary_name = self.calibration_vars["dictionary"].get()
        try:
            if not hasattr(cv2, "aruco"):
                raise ValueError("OpenCV ArUcoモジュールがありません。opencv-contrib-pythonを確認してください。")
            dictionary_id = getattr(cv2.aruco, dictionary_name, None)
            if dictionary_id is None:
                raise ValueError(f"不明なArUco辞書です: {dictionary_name}")
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_length, marker_length, dictionary
            )
            image = board.generateImage((image_width, image_height))
            image = cv2.copyMakeBorder(
                image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), image):
                raise OSError(f"画像を保存できませんでした: {output}")
        except (cv2.error, OSError, ValueError) as exc:
            messagebox.showerror("ボード作成エラー", str(exc), parent=self.root)
            return

        settings = (squares_x, squares_y, square_length, marker_length, dictionary_name, dpi)
        self.board_generated_settings = settings
        preview_scale = min(700 / image.shape[1], 300 / image.shape[0], 1.0)
        preview = cv2.resize(
            image,
            (max(1, round(image.shape[1] * preview_scale)), max(1, round(image.shape[0] * preview_scale))),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".png", preview)
        if ok:
            self.board_preview_photo = tk.PhotoImage(
                data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="png"
            )
            self.board_preview_label.configure(image=self.board_preview_photo, text="")
        self.board_status.set(
            f"ボード画像を保存しました: {output}\n"
            f"サイズ: {squares_x} × {squares_y} マス、{square_length:g} mm、{dpi} dpi。"
        )
        self.board_next_button.configure(text="キャリブレーションへ進む")

    def _build_calibration_page(self) -> None:
        self._clear_content()
        self.step_label.set("ステップ 2 / 4　カメラキャリブレーション")
        self.content.columnconfigure(1, weight=1)

        ttk.Label(
            self.content,
            text="動画を解析してカメラ値を計算するか、既存のキャリブレーションデータを読み込んでホモグラフィへ進みます。",
            wraplength=920,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(self.content, text="キャリブレーション動画").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(self.content, textvariable=self.calibration_vars["video"]).grid(
            row=1, column=1, sticky="ew", padx=8
        )
        ttk.Button(self.content, text="選択…", command=self._choose_video).grid(row=1, column=2)

        ttk.Label(self.content, text="結果の保存先").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(self.content, textvariable=self.calibration_vars["output"]).grid(
            row=2, column=1, sticky="ew", padx=8
        )
        ttk.Button(self.content, text="選択…", command=self._choose_calibration_output).grid(
            row=2, column=2
        )

        board = ttk.LabelFrame(self.content, text="ボード設定（使用する実物に合わせてください）", padding=10)
        board.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 5))
        for column in range(4):
            board.columnconfigure(column, weight=1)
        self._add_field(board, "横のマス数", self.calibration_vars["squares_x"], 0, 0)
        self._add_field(board, "縦のマス数", self.calibration_vars["squares_y"], 0, 2)
        self._add_field(board, "マスの一辺", self.calibration_vars["square_length"], 1, 0)
        self._add_field(board, "マーカーの一辺", self.calibration_vars["marker_length"], 1, 2)
        ttk.Label(board, text="長さの単位は統一してください（例: mm）").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(3, 5)
        )
        ttk.Label(board, text="ArUco辞書").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(
            board,
            textvariable=self.calibration_vars["dictionary"],
            state="readonly",
            values=(
                "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100",
                "DICT_6X6_50", "DICT_6X6_100", "DICT_7X7_50", "DICT_7X7_100",
            ),
        ).grid(row=3, column=1, columnspan=3, sticky="ew", padx=8)

        options = ttk.LabelFrame(self.content, text="解析設定", padding=10)
        options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        for column in range(4):
            options.columnconfigure(column, weight=1)
        self._add_field(options, "使用フレーム数", self.calibration_vars["top_n"], 0, 0)
        self._add_field(options, "解析間隔", self.calibration_vars["stride"], 0, 2)
        self._add_field(options, "ブレ判定しきい値", self.calibration_vars["blur_threshold"], 1, 0)
        self._add_field(options, "最小コーナー数", self.calibration_vars["min_corners"], 1, 2)
        self._add_field(options, "最大解析フレーム数", self.calibration_vars["max_frames"], 2, 0)
        self._add_field(options, "低カバレッジ判定", self.calibration_vars["low_coverage"], 2, 2)
        ttk.Label(options, text="最大解析フレーム数 0 は動画全体を解析します").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(3, 0)
        )

        actions = ttk.Frame(self.content)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(9, 5))
        self.calibration_back_button = ttk.Button(
            actions, text="ボード作成ツールへ戻る", command=self._build_board_page
        )
        self.calibration_back_button.pack(side="left")
        self.run_button = ttk.Button(
            actions, text="キャリブレーションを開始", command=self._start_calibration
        )
        self.run_button.pack(side="right", padx=(8, 0))
        self.load_calibration_button = ttk.Button(
            actions,
            text="既存データを読み込んでホモグラフィへ",
            command=self._import_calibration_data,
        )
        self.load_calibration_button.pack(side="right")
        self.status = tk.StringVar(value="動画とボード設定を指定してください")
        ttk.Label(self.content, textvariable=self.status).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )
        self.log = tk.Text(self.content, height=6, wrap="word", state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew")
        self.calibration_progress = tk.DoubleVar(value=0.0)
        self.progress_detail = tk.StringVar(value="処理開始を待っています")
        self.progress_mode = "determinate"
        self.progress_bar = ttk.Progressbar(
            self.content,
            variable=self.calibration_progress,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        ttk.Label(self.content, textvariable=self.progress_detail, wraplength=920).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        self.content.rowconfigure(7, weight=1)

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="キャリブレーション動画を選択",
            filetypes=(
                ("動画", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("すべてのファイル", "*.*"),
            ),
        )
        if path:
            self.calibration_vars["video"].set(path)

    def _choose_calibration_output(self) -> None:
        path = filedialog.askdirectory(title="キャリブレーション結果の保存先")
        if path:
            self.calibration_vars["output"].set(path)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start_calibration(self) -> None:
        video = Path(self.calibration_vars["video"].get().strip()).expanduser()
        if not video.is_file():
            messagebox.showerror("入力エラー", "キャリブレーション動画を選択してください。", parent=self.root)
            return

        try:
            integer_values = {
                key: int(self.calibration_vars[key].get())
                for key in ("squares_x", "squares_y", "top_n", "stride", "min_corners", "max_frames")
            }
            square_length = float(self.calibration_vars["square_length"].get())
            marker_length = float(self.calibration_vars["marker_length"].get())
            blur_threshold = float(self.calibration_vars["blur_threshold"].get())
            low_coverage = float(self.calibration_vars["low_coverage"].get())
            numeric_values = (square_length, marker_length, blur_threshold, low_coverage)
            if not np.isfinite(numeric_values).all():
                raise ValueError("数値欄には有限の値を入力してください。")
            if integer_values["squares_x"] < 2 or integer_values["squares_y"] < 2:
                raise ValueError("ボードのマス数は横・縦ともに2以上にしてください。")
            if integer_values["top_n"] < 3 or integer_values["stride"] < 1:
                raise ValueError("使用フレーム数は3以上、解析間隔は1以上にしてください。")
            if integer_values["min_corners"] < 1 or integer_values["max_frames"] < 0:
                raise ValueError("最小コーナー数と最大解析フレーム数を確認してください。")
            if not 0 < marker_length < square_length:
                raise ValueError("マーカーの一辺は、マスの一辺より短くしてください。")
            if blur_threshold < 0 or not 0 <= low_coverage <= 1:
                raise ValueError("ブレ判定は0以上、低カバレッジ判定は0〜1で指定してください。")
        except ValueError as exc:
            messagebox.showerror("入力エラー", str(exc) or "数値の入力を確認してください。", parent=self.root)
            return

        output = Path(self.calibration_vars["output"].get().strip() or str(DEFAULT_RESULTS_DIR / "calibration")).expanduser()
        if not output.is_absolute():
            output = APP_DIR / output
        video = video.resolve()
        output = output.resolve()
        worker_config = {
            "video": str(video),
            "output": str(output),
            "squares_x": integer_values["squares_x"],
            "squares_y": integer_values["squares_y"],
            "square_length": square_length,
            "marker_length": marker_length,
            "dictionary": self.calibration_vars["dictionary"].get(),
            "top_n": integer_values["top_n"],
            "stride": integer_values["stride"],
            "blur_threshold": blur_threshold,
            "min_corners": integer_values["min_corners"],
            "max_frames": integer_values["max_frames"],
            "low_coverage": low_coverage,
        }
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--calibration-worker",
            json.dumps(worker_config, ensure_ascii=False),
        ]
        self.run_button.configure(state="disabled")
        self.calibration_back_button.configure(state="disabled")
        self.load_calibration_button.configure(state="disabled")
        self.status.set("解析中です。動画の長さによって時間がかかります…")
        self.calibration_progress.set(0.0)
        self.progress_detail.set("準備中: 動画と解析設定を確認しています")
        if self.progress_mode == "indeterminate":
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_mode = "determinate"
        self._append_log(f"統合アプリ内のキャリブレーションエンジンを実行: {video}")
        self._append_log(f"結果の保存先: {output}")
        threading.Thread(
            target=self._run_calibration_process,
            args=(command, output),
            daemon=True,
        ).start()

    def _run_calibration_process(self, command: list[str], output: Path) -> None:
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
            ) as process:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if line.startswith("PROGRESS "):
                        try:
                            progress = json.loads(line[len("PROGRESS "):])
                        except json.JSONDecodeError:
                            self.messages.put(("calibration_output", line))
                        else:
                            self.messages.put(("calibration_progress", progress))
                    elif line:
                        self.messages.put(("calibration_output", line))
                return_code = process.wait()
            self.messages.put((
                "calibration_done",
                (return_code, "", "", output),
            ))
        except Exception as exc:  # Surface process launch errors in the UI.
            self.messages.put(("calibration_error", str(exc)))

    def _update_calibration_progress(self, progress: dict[str, object]) -> None:
        overall_value = progress.get("overall")
        overall = float(overall_value) if overall_value is not None else self.calibration_progress.get()
        detail = str(progress.get("detail", "解析中です"))
        indeterminate = bool(progress.get("indeterminate", False))
        if indeterminate and self.progress_mode != "indeterminate":
            self.progress_bar.stop()
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(12)
            self.progress_mode = "indeterminate"
        elif not indeterminate and self.progress_mode != "determinate":
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_mode = "determinate"
        self.calibration_progress.set(overall)
        phase = progress.get("phase")
        if phase is None:
            phase_text = ""
        else:
            phase_text = f"（この工程 {float(phase):.0f}%）"
        if overall_value is None:
            progress_text = "全体の進捗率は算出できません"
        else:
            progress_text = f"全体の目安 {overall:.0f}%"
        self.progress_detail.set(f"{progress_text}　{detail} {phase_text}".rstrip())

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "calibration_error":
                    if hasattr(self, "progress_bar"):
                        self.progress_bar.stop()
                    self.run_button.configure(state="normal")
                    self.calibration_back_button.configure(state="normal")
                    self.load_calibration_button.configure(state="normal")
                    self.status.set("キャリブレーションを起動できませんでした")
                    self._append_log(str(payload))
                    continue

                if kind == "calibration_progress":
                    self._update_calibration_progress(payload)  # type: ignore[arg-type]
                    continue

                if kind == "calibration_output":
                    self._append_log(str(payload))
                    continue

                if kind == "calibration_done":
                    return_code, stdout, stderr, output = payload  # type: ignore[misc]
                    self.progress_bar.stop()
                    self.run_button.configure(state="normal")
                    self.calibration_back_button.configure(state="normal")
                    self.load_calibration_button.configure(state="normal")
                    output_text = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
                    self._append_log(output_text or "処理が終了しました。")
                    if return_code != 0:
                        self.status.set("処理に失敗しました。ログを確認してください。")
                        continue
                    try:
                        self.calibration_data = self._read_calibration_result(Path(output))
                        summary_path = self._write_calibration_summary(
                            Path(output), self.calibration_data
                        )
                    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                        self.status.set("結果を読み込めませんでした。ログを確認してください。")
                        self._append_log(f"キャリブレーション結果の読み込みエラー: {exc}")
                        continue
                    self._append_log(f"TXT出力: {summary_path}")
                    self._build_homography_page()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _read_calibration_result(self, output: Path) -> dict[str, object]:
        result_path = output / "calibration.npz"
        if not result_path.is_file():
            raise ValueError(f"結果ファイルがありません: {result_path}")
        return self._load_calibration_file(result_path)

    @staticmethod
    def _write_calibration_summary(output: Path, data: dict[str, object]) -> Path:
        camera_matrix = data["camera_matrix"]
        distortion = data["distortion"]
        rmse = data.get("rmse")
        rmse_line = (
            f"再投影誤差: {float(rmse):.4f} px"
            if isinstance(rmse, (int, float, np.number)) and np.isfinite(rmse)
            else "再投影誤差: 未記録"
        )
        matrix_rows = [
            "[ " + ",  ".join(str(float(value)) for value in row) + " ]"
            for row in camera_matrix
        ]
        distortion_lines = [
            f"{name}: {float(value)}"
            for name, value in zip(("k1", "k2", "p1", "p2", "k3"), distortion)
        ]
        text = (
            f"解像度（横 × 縦）: {data['width']} × {data['height']} px\n"
            + rmse_line
            + "\n\n"
            "カメラマトリックス:\n"
            + "\n".join(matrix_rows)
            + "\n\n歪み定数:\n"
            + "\n".join(distortion_lines)
            + "\n"
        )
        summary_path = output / "calibration_summary.txt"
        summary_path.write_text(text, encoding="utf-8")
        return summary_path

    def _build_homography_page(self) -> None:
        assert self.calibration_data is not None
        self._clear_content()
        self.step_label.set("ステップ 3 / 4　ホモグラフィの設定")
        self.content.columnconfigure(1, weight=1)
        ttk.Label(
            self.content,
            text="キャリブレーション結果を読み込みました。基準画像と平面の実寸を指定してください。",
            wraplength=920,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        camera_matrix = self.calibration_data["camera_matrix"]
        distortion = self.calibration_data["distortion"]
        width = self.calibration_data["width"]
        height = self.calibration_data["height"]
        rmse = self.calibration_data["rmse"]
        if isinstance(rmse, (int, float, np.number)) and np.isfinite(rmse):
            calibration_status = f"再投影誤差: {rmse:.4f} px"
        else:
            calibration_status = "既存のキャリブレーションデータを使用"
        summary = (
            f"キャリブレーション解像度: {width} × {height} px    {calibration_status}\n"
            f"K: fx={camera_matrix[0, 0]:.4f}, fy={camera_matrix[1, 1]:.4f}, "
            f"cx={camera_matrix[0, 2]:.4f}, cy={camera_matrix[1, 2]:.4f}\n"
            "歪み係数 k1, k2, p1, p2, k3: "
            + ", ".join(f"{value:.7g}" for value in distortion)
        )
        ttk.Label(self.content, text=summary, justify="left", relief="groove", padding=9).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )

        ttk.Label(self.content, text="基準画像").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(self.content, textvariable=self.homography_vars["image"]).grid(
            row=2, column=1, sticky="ew", padx=8
        )
        ttk.Button(self.content, text="選択…", command=self._choose_reference_image).grid(row=2, column=2)

        dimensions = ttk.LabelFrame(self.content, text="ポール平面の実寸", padding=10)
        dimensions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
        self._add_field(dimensions, "幅 (m)", self.homography_vars["width"], 0, 0)
        self._add_field(dimensions, "高さ (m)", self.homography_vars["height"], 0, 2)
        ttk.Label(dimensions, text="座標原点は左下です。四隅は左下 → 右下 → 右上 → 左上の順に選択します。").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )

        ttk.Label(self.content, text="行列の保存先 (.npy)").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(self.content, textvariable=self.homography_vars["output"]).grid(
            row=4, column=1, sticky="ew", padx=8
        )
        ttk.Button(self.content, text="保存先…", command=self._choose_homography_output).grid(
            row=4, column=2
        )

        buttons = ttk.Frame(self.content)
        buttons.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="キャリブレーション設定へ戻る", command=self._build_calibration_page).pack(side="left")
        ttk.Button(
            buttons,
            text="既存のホモグラフィ行列を読み込んで次へ",
            command=self._import_homography_data,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="補正画像で四隅を選択", command=self._open_point_selection).pack(side="right")

    def _import_homography_data(self) -> None:
        selected = filedialog.askopenfilename(
            title="既存のホモグラフィ行列を選択",
            filetypes=(
                ("NumPyファイル", "*.npy *.npz"),
                ("すべてのファイル", "*.*"),
            ),
        )
        if not selected:
            return
        path = Path(selected).expanduser()
        has_embedded_metadata = False
        try:
            if path.suffix.lower() == ".npy":
                loaded_array = np.load(path, allow_pickle=False)
                if loaded_array.dtype.names and "homography_matrix" in loaded_array.dtype.names:
                    matrix = np.asarray(loaded_array["homography_matrix"], dtype=np.float64)
                    if "metadata_json" in loaded_array.dtype.names:
                        raw_metadata = loaded_array["metadata_json"]
                        metadata_text = str(raw_metadata.item() if raw_metadata.shape == () else raw_metadata.flat[0])
                        metadata = json.loads(metadata_text)
                        if not isinstance(metadata, dict):
                            raise ValueError("NPY内の設定情報の形式が正しくありません。")
                        has_embedded_metadata = True
                else:
                    matrix = np.asarray(loaded_array, dtype=np.float64)
            elif path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    matrix_key = next(
                        (key for key in ("homography_matrix", "homography", "H", "matrix") if key in archive.files),
                        None,
                    )
                    if matrix_key is None and len(archive.files) == 1:
                        matrix_key = archive.files[0]
                    if matrix_key is None:
                        raise ValueError("NPZにホモグラフィ行列がありません。")
                    matrix = np.asarray(archive[matrix_key], dtype=np.float64)
            else:
                raise ValueError(".npy または .npz のホモグラフィ行列を選択してください。")
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("ホモグラフィ行列は有限値の3×3配列である必要があります。")
        except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            messagebox.showerror("読み込みエラー", str(exc), parent=self.root)
            return
        self.output_path = path
        self._build_completion_page(matrix, loaded=True, has_embedded_metadata=has_embedded_metadata)

    def _choose_reference_image(self) -> None:
        path = filedialog.askopenfilename(
            title="ホモグラフィの基準画像を選択",
            filetypes=(
                ("画像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("すべてのファイル", "*.*"),
            ),
        )
        if path:
            self.homography_vars["image"].set(path)

    def _choose_homography_output(self) -> None:
        current = Path(self.homography_vars["output"].get()).expanduser()
        path = filedialog.asksaveasfilename(
            title="ホモグラフィ行列の保存先",
            defaultextension=".npy",
            initialdir=str(current.parent),
            initialfile=current.name,
            filetypes=(("NumPy行列", "*.npy"),),
        )
        if path:
            self.homography_vars["output"].set(path)

    def _open_point_selection(self) -> None:
        assert self.calibration_data is not None
        image_path = Path(self.homography_vars["image"].get().strip()).expanduser()
        if not image_path.is_file():
            messagebox.showerror("入力エラー", "基準画像を選択してください。", parent=self.root)
            return
        try:
            width_m = float(self.homography_vars["width"].get())
            height_m = float(self.homography_vars["height"].get())
        except ValueError:
            messagebox.showerror("入力エラー", "平面の幅と高さを数値で入力してください。", parent=self.root)
            return
        if not np.isfinite([width_m, height_m]).all() or width_m <= 0 or height_m <= 0:
            messagebox.showerror("入力エラー", "平面の幅と高さは有限の正の値にしてください。", parent=self.root)
            return

        output_path = Path(self.homography_vars["output"].get().strip()).expanduser()
        if output_path.suffix.lower() != ".npy":
            messagebox.showerror("入力エラー", "保存先の拡張子は .npy にしてください。", parent=self.root)
            return
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("画像エラー", f"画像を読み込めません: {image_path}", parent=self.root)
            return

        image_height, image_width = image.shape[:2]
        expected_size = (self.calibration_data["width"], self.calibration_data["height"])
        if (image_width, image_height) != expected_size:
            self._build_resolution_choice_page(
                image,
                image_path,
                output_path,
                width_m,
                height_m,
                (image_width, image_height),
                expected_size,
            )
            return

        camera_matrix = self.calibration_data["camera_matrix"]
        distortion = self.calibration_data["distortion"]
        self._prepare_point_selection(
            image,
            image_path,
            output_path,
            width_m,
            height_m,
            camera_matrix,
            distortion,
            image_alignment={"mode": "same_resolution", "scale_xy": [1.0, 1.0]},
        )

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
        self._clear_content()
        self.step_label.set("ステップ 3 / 4　解像度の基準を選択")
        self.content.columnconfigure(0, weight=1)
        ref_width, ref_height = reference_size
        cal_width, cal_height = calibration_size
        scale_to_cal = (cal_width / ref_width, cal_height / ref_height)
        scale_to_ref = (ref_width / cal_width, ref_height / cal_height)

        ttk.Label(
            self.content,
            text="基準画像とキャリブレーション時の画像サイズが異なります。どちらの画像サイズをホモグラフィの座標基準にするか選んでください。",
            wraplength=920,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Label(
            self.content,
            text=(
                f"基準画像: {ref_width} × {ref_height} px\n"
                f"キャリブレーション画像: {cal_width} × {cal_height} px"
            ),
            justify="left",
            relief="groove",
            padding=9,
        ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

        calibration_choice = ttk.LabelFrame(
            self.content, text="キャリブレーション画像の解像度に合わせる", padding=12
        )
        calibration_choice.grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Label(
            calibration_choice,
            text=(
                f"基準画像を {cal_width} × {cal_height} px にリサイズしてから補正します。\n"
                f"倍率: 横 {scale_to_cal[0]:.6g}、縦 {scale_to_cal[1]:.6g}\n"
                "作成される行列の入力座標は、リサイズ後のキャリブレーション解像度です。"
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(
            calibration_choice,
            text="キャリブレーション解像度を使う",
            command=lambda: self._choose_calibration_resolution(
                image, image_path, output_path, width_m, height_m,
                reference_size, calibration_size,
            ),
        ).pack(anchor="e")

        reference_choice = ttk.LabelFrame(
            self.content, text="基準画像の解像度に合わせる", padding=12
        )
        reference_choice.grid(row=3, column=0, sticky="ew", pady=6)
        ttk.Label(
            reference_choice,
            text=(
                "基準画像のサイズを保ち、カメラ行列の横方向・縦方向の値を画像サイズに合わせて調整します。\n"
                f"倍率: 横 {scale_to_ref[0]:.6g}、縦 {scale_to_ref[1]:.6g}\n"
                "同じカメラ・画角で撮影し、解像度だけを変更した画像に適した方法です。"
            ),
            justify="left",
            wraplength=900,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(
            reference_choice,
            text="基準画像の解像度を使う",
            command=lambda: self._choose_reference_resolution(
                image, image_path, output_path, width_m, height_m,
                reference_size, calibration_size,
            ),
        ).pack(anchor="e")

        ttk.Label(
            self.content,
            text="撮影範囲やカメラが異なる場合、解像度の調整だけでは正しい補正値になりません。",
            wraplength=920,
        ).grid(row=4, column=0, sticky="w", pady=(10, 6))
        ttk.Button(
            self.content,
            text="画像・実寸の設定へ戻る",
            command=self._build_homography_page,
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))

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
        cal_width, cal_height = calibration_size
        ref_width, ref_height = reference_size
        interpolation = cv2.INTER_AREA if cal_width < ref_width or cal_height < ref_height else cv2.INTER_CUBIC
        resized = cv2.resize(image, calibration_size, interpolation=interpolation)
        camera_matrix = self.calibration_data["camera_matrix"]
        distortion = self.calibration_data["distortion"]
        alignment = {
            "mode": "reference_resized_to_calibration_resolution",
            "source_image_size": [ref_width, ref_height],
            "processing_image_size": [cal_width, cal_height],
            "scale_xy": [cal_width / ref_width, cal_height / ref_height],
        }
        self._prepare_point_selection(
            resized, image_path, output_path, width_m, height_m,
            camera_matrix, distortion, alignment,
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
        ref_width, ref_height = reference_size
        cal_width, cal_height = calibration_size
        original_camera_matrix = self.calibration_data["camera_matrix"]
        camera_matrix = original_camera_matrix.copy()
        scale_x, scale_y = ref_width / cal_width, ref_height / cal_height
        camera_matrix[0, :] *= scale_x
        camera_matrix[1, :] *= scale_y
        distortion = self.calibration_data["distortion"]
        alignment = {
            "mode": "calibration_intrinsics_scaled_to_reference_resolution",
            "calibration_image_size": [cal_width, cal_height],
            "processing_image_size": [ref_width, ref_height],
            "scale_xy": [scale_x, scale_y],
        }
        self._prepare_point_selection(
            image, image_path, output_path, width_m, height_m,
            camera_matrix, distortion, alignment,
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
        try:
            undistorted = cv2.undistort(image, camera_matrix, distortion)
        except cv2.error as exc:
            messagebox.showerror("画像補正エラー", str(exc), parent=self.root)
            return

        self.image_path = image_path
        self.output_path = output_path
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.undistorted_image = undistorted
        self.image_alignment = image_alignment
        self.calibration_camera_matrix = self.calibration_data["camera_matrix"]
        self.calibration_image_size = (
            self.calibration_data["width"], self.calibration_data["height"]
        )
        self.image_height, self.image_width = undistorted.shape[:2]
        self.points = []
        self._build_point_page(width_m, height_m)

    def _build_point_page(self, width_m: float, height_m: float) -> None:
        assert self.undistorted_image is not None
        self._clear_content()
        self.step_label.set("ステップ 4 / 4　補正画像上で四隅を選択")
        self.content.rowconfigure(2, weight=1)
        self.content.columnconfigure(0, weight=1)

        ttk.Label(
            self.content,
            text="左下 → 右下 → 右上 → 左上の順で四隅をクリックしてください。拡大表示中も元画像の座標に換算します。",
            wraplength=950,
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        original_height, original_width = self.undistorted_image.shape[:2]
        self.fit_scale = min(
            self.viewport_width / original_width,
            self.viewport_height / original_height,
            1.0,
        )
        self.zoom_factor = 1.0

        zoom_controls = ttk.Frame(self.content)
        zoom_controls.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Button(zoom_controls, text="−", width=3, command=lambda: self._change_zoom(1 / 1.25)).pack(side="left")
        ttk.Button(zoom_controls, text="＋", width=3, command=lambda: self._change_zoom(1.25)).pack(
            side="left", padx=(4, 8)
        )
        ttk.Button(zoom_controls, text="全体表示", command=self._fit_image).pack(side="left")
        self.zoom_status = tk.StringVar()
        ttk.Label(zoom_controls, textvariable=self.zoom_status).pack(side="left", padx=(10, 0))
        ttk.Label(zoom_controls, text="ホイール: 拡大/縮小　中央ボタンでドラッグ: 移動").pack(side="right")

        image_frame = ttk.Frame(self.content)
        image_frame.grid(row=2, column=0, sticky="nsew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            image_frame,
            width=self.viewport_width,
            height=self.viewport_height,
            highlightthickness=0,
            background="#202020",
            xscrollincrement=24,
            yscrollincrement=24,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical_scroll = ttk.Scrollbar(image_frame, orient="vertical", command=self.canvas.yview)
        vertical_scroll.grid(row=0, column=1, sticky="ns")
        horizontal_scroll = ttk.Scrollbar(image_frame, orient="horizontal", command=self.canvas.xview)
        horizontal_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal_scroll.set, yscrollcommand=vertical_scroll.set)

        self.point_status = tk.StringVar()
        ttk.Label(self.content, textvariable=self.point_status).grid(row=3, column=0, sticky="w", pady=(4, 0))
        controls = ttk.Frame(self.content)
        controls.grid(row=4, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(controls, text="1点戻る", command=self._undo_point).pack(side="left")
        ttk.Button(controls, text="全点リセット", command=self._reset_points).pack(side="left", padx=(7, 0))
        ttk.Button(controls, text="設定へ戻る", command=self._build_homography_page).pack(side="right")
        self.save_button = ttk.Button(
            controls,
            text="行列を計算して保存",
            command=lambda: self._save_homography(width_m, height_m),
            state="disabled",
        )
        self.save_button.pack(side="right", padx=(0, 7))
        self.canvas.bind("<Button-1>", self._add_point)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_from_event(event, 1.25))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_from_event(event, 1 / 1.25))
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan_image)
        self._render_point_image()
        self._redraw_points()

    def _add_point(self, event: tk.Event) -> None:
        if len(self.points) >= 4:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        if not (0 <= canvas_x < self.display_width and 0 <= canvas_y < self.display_height):
            return
        x = min(max(canvas_x / self.display_scale_x, 0), self.image_width - 1)
        y = min(max(canvas_y / self.display_scale_y, 0), self.image_height - 1)
        self.points.append((x, y))
        self._redraw_points()

    def _render_point_image(self, anchor: tuple[int, int] | None = None) -> None:
        assert self.undistorted_image is not None
        old_image_x = old_image_y = None
        if anchor is not None and hasattr(self, "canvas") and self.canvas.winfo_exists():
            old_image_x = self.canvas.canvasx(anchor[0]) / self.display_scale_x
            old_image_y = self.canvas.canvasy(anchor[1]) / self.display_scale_y

        self.display_scale = self.fit_scale * self.zoom_factor
        self.display_width = max(1, round(self.image_width * self.display_scale))
        self.display_height = max(1, round(self.image_height * self.display_scale))
        self.display_scale_x = self.display_width / self.image_width
        self.display_scale_y = self.display_height / self.image_height
        interpolation = cv2.INTER_AREA if self.display_scale < 1 else cv2.INTER_LINEAR
        shown = cv2.resize(
            self.undistorted_image,
            (self.display_width, self.display_height),
            interpolation=interpolation,
        )
        ok, encoded = cv2.imencode(".png", shown)
        if not ok:
            messagebox.showerror("画像エラー", "補正画像を表示できませんでした。", parent=self.root)
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="png")
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="image")
        scroll_width = max(self.display_width, self.viewport_width)
        scroll_height = max(self.display_height, self.viewport_height)
        self.canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        self.zoom_status.set(f"表示倍率: {self.zoom_factor * 100:.0f}%（全体表示を100%として表示）")

        if anchor is not None and old_image_x is not None and old_image_y is not None:
            self.canvas.update_idletasks()
            target_x = old_image_x * self.display_scale_x - anchor[0]
            target_y = old_image_y * self.display_scale_y - anchor[1]
            max_x = max(0, self.display_width - self.canvas.winfo_width())
            max_y = max(0, self.display_height - self.canvas.winfo_height())
            target_x = min(max(target_x, 0), max_x)
            target_y = min(max(target_y, 0), max_y)
            self.canvas.xview_moveto(target_x / scroll_width if scroll_width else 0)
            self.canvas.yview_moveto(target_y / scroll_height if scroll_height else 0)
        else:
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
        self.canvas.tag_raise("marker")

    def _change_zoom(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        if anchor is None and hasattr(self, "canvas") and self.canvas.winfo_exists():
            anchor = (self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)
        self.zoom_factor = min(max(self.zoom_factor * factor, 0.5), 6.0)
        self._render_point_image(anchor)
        self._redraw_points()

    def _zoom_from_event(self, event: tk.Event, factor: float) -> str:
        self._change_zoom(factor, (event.x, event.y))
        return "break"

    def _on_mousewheel(self, event: tk.Event) -> str:
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        return self._zoom_from_event(event, factor)

    def _fit_image(self) -> None:
        self.zoom_factor = 1.0
        self._render_point_image()
        self._redraw_points()

    def _start_pan(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _pan_image(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw_points()

    def _reset_points(self) -> None:
        self.points.clear()
        self._redraw_points()

    def _redraw_points(self) -> None:
        if not hasattr(self, "canvas") or not self.canvas.winfo_exists():
            return
        labels = ("左下", "右下", "右上", "左上")
        self.canvas.delete("marker")
        for index, (x, y) in enumerate(self.points):
            shown_x = x * self.display_scale_x
            shown_y = y * self.display_scale_y
            self.canvas.create_oval(
                shown_x - 5, shown_y - 5, shown_x + 5, shown_y + 5,
                fill="red", outline="white", tags="marker",
            )
            self.canvas.create_text(
                shown_x + 10, shown_y - 10,
                text=f"{index + 1}: {labels[index]}",
                fill="red", anchor="sw", tags="marker",
            )
        if len(self.points) < 4:
            self.point_status.set(f"{len(self.points) + 1}/4: {labels[len(self.points)]}をクリック")
        else:
            self.point_status.set("四隅を選択しました。行列を計算して保存できます。")
        self.save_button.configure(state="normal" if len(self.points) == 4 else "disabled")

    def _save_homography(self, width_m: float, height_m: float) -> None:
        assert self.image_path is not None
        assert self.output_path is not None
        assert self.camera_matrix is not None
        assert self.distortion is not None
        assert self.calibration_camera_matrix is not None
        if len(self.points) != 4:
            return

        source = np.asarray(self.points, dtype=np.float32)
        contour = source.reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 10:
            messagebox.showerror(
                "点の位置を確認",
                "左下 → 右下 → 右上 → 左上の順に、四隅を選択してください。",
                parent=self.root,
            )
            return

        destination = np.asarray(
            [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
            dtype=np.float32,
        )
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is None or not np.isfinite(matrix).all():
            messagebox.showerror("計算エラー", "ホモグラフィ行列を計算できませんでした。", parent=self.root)
            return

        metadata = {
            "format": "homography_with_metadata_v1",
            "coordinate_system": "vertical_plane, meters; origin=left_bottom",
            "image_size": [self.image_width, self.image_height],
            "image_file": str(self.image_path),
            "camera_matrix": self.camera_matrix.tolist(),
            "calibration_camera_matrix": self.calibration_camera_matrix.tolist(),
            "calibration_image_size": list(self.calibration_image_size),
            "image_alignment": self.image_alignment,
            "distortion_coefficients_k1_k2_p1_p2_k3": self.distortion.tolist(),
            "image_points_undistorted_px": source.tolist(),
            "plane_points_m": destination.tolist(),
            "plane_width_m": width_m,
            "plane_height_m": height_m,
            "point_order": ["左下", "右下", "右上", "左上"],
        }
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            record_dtype = np.dtype([
                ("homography_matrix", np.float64, (3, 3)),
                ("metadata_json", f"<U{max(1, len(metadata_text))}"),
            ])
            packaged = np.zeros((), dtype=record_dtype)
            packaged["homography_matrix"] = matrix
            packaged["metadata_json"] = metadata_text
            np.save(self.output_path, packaged, allow_pickle=False)
            self.output_path.with_suffix(".txt").write_text(
                self._format_homography_summary(matrix, metadata),
                encoding="utf-8",
            )
        except OSError as exc:
            messagebox.showerror("保存エラー", str(exc), parent=self.root)
            return
        self._build_completion_page(matrix)

    def _format_homography_summary(self, matrix: np.ndarray, metadata: dict[str, Any]) -> str:
        matrix_rows = [
            "[ " + ",  ".join(str(float(value)) for value in row) + " ]"
            for row in matrix
        ]
        alignment = metadata["image_alignment"]
        alignment_mode = alignment.get("mode")
        alignment_labels = {
            "same_resolution": "キャリブレーション解像度と同一（調整なし）",
            "reference_resized_to_calibration_resolution": "基準画像をキャリブレーション解像度にリサイズ",
            "calibration_intrinsics_scaled_to_reference_resolution": "カメラ行列を基準画像の解像度にスケーリング",
        }
        alignment_text = alignment_labels.get(alignment_mode, str(alignment_mode))
        camera_matrix_rows = "\n".join(
            "[ " + ",  ".join(str(float(value)) for value in row) + " ]"
            for row in metadata["camera_matrix"]
        )
        calibration_matrix_rows = "\n".join(
            "[ " + ",  ".join(str(float(value)) for value in row) + " ]"
            for row in metadata["calibration_camera_matrix"]
        )
        distortion = metadata["distortion_coefficients_k1_k2_p1_p2_k3"]
        source_points = "\n".join(
            f"  {label}: ({point[0]:.6f}, {point[1]:.6f}) px"
            for label, point in zip(metadata["point_order"], metadata["image_points_undistorted_px"])
        )
        plane_points = "\n".join(
            f"  {label}: ({point[0]:.6f}, {point[1]:.6f}) m"
            for label, point in zip(metadata["point_order"], metadata["plane_points_m"])
        )
        return (
            "ホモグラフィ行列（3×3）:\n"
            + "\n".join(matrix_rows)
            + "\n\n変換元・変換先と単位:\n"
            + "補正後画像のピクセル座標 → 平面上のメートル座標\n\n"
            + "座標系・原点: 垂直平面上、左下を原点\n"
            + f"座標基準の画像解像度: {self.image_width} × {self.image_height} px\n"
            + f"基準画像: {metadata['image_file']}\n"
            + f"解像度合わせ: {alignment_text}\n\n"
            + "キャリブレーション時の画像解像度: "
            + f"{metadata['calibration_image_size'][0]} × {metadata['calibration_image_size'][1]} px\n"
            + "ホモグラフィ計算に使用したカメラマトリックス:\n"
            + camera_matrix_rows
            + "\nキャリブレーション元カメラマトリックス:\n"
            + calibration_matrix_rows
            + "\n歪み係数 (k1, k2, p1, p2, k3): "
            + ", ".join(str(float(value)) for value in distortion)
            + "\n\n画像上の選択点（補正後 px）:\n"
            + source_points
            + "\n平面上の対応点 (m):\n"
            + plane_points
            + f"\n平面サイズ: 幅 {metadata['plane_width_m']} m × 高さ {metadata['plane_height_m']} m\n"
            + "点の順序: " + " → ".join(metadata["point_order"]) + "\n"
        )

    def _build_completion_page(
        self, matrix: np.ndarray, loaded: bool = False, has_embedded_metadata: bool = False
    ) -> None:
        assert self.output_path is not None
        self._clear_content()
        self.step_label.set("完了")
        ttk.Label(
            self.content,
            text="既存のホモグラフィ行列を読み込みました。" if loaded else "ホモグラフィ行列を保存しました。",
            font=("TkDefaultFont", 15, "bold"),
        ).pack(anchor="w", pady=(6, 12))
        path_label = "読込元" if loaded else "行列"
        ttk.Label(self.content, text=f"{path_label}: {self.output_path}", wraplength=950, justify="left").pack(
            anchor="w", pady=4
        )
        if not loaded or has_embedded_metadata:
            ttk.Label(
                self.content,
                text="設定情報: NPYファイル内に含まれています",
                wraplength=950,
                justify="left",
            ).pack(anchor="w", pady=4)
        if not loaded or self.output_path.with_suffix(".txt").is_file():
            ttk.Label(
                self.content,
                text=f"TXT: {self.output_path.with_suffix('.txt')}",
                wraplength=950,
                justify="left",
            ).pack(anchor="w", pady=4)
        matrix_text = "\n".join("[ " + ",  ".join(f"{value:.8g}" for value in row) + " ]" for row in matrix)
        ttk.Label(self.content, text=f"変換行列 H:\n{matrix_text}", justify="left", relief="groove", padding=10).pack(
            anchor="w", fill="x", pady=(12, 16)
        )
        actions = ttk.Frame(self.content)
        actions.pack(anchor="w", pady=(4, 0))
        ttk.Button(actions, text="ホモグラフィ設定へ戻る", command=self._build_homography_page).pack(side="left")
        ttk.Button(
            actions,
            text="キャリブレーションへ戻る",
            command=self._build_calibration_page,
        ).pack(side="left", padx=(8, 0))


def main() -> None:
    root = tk.Tk()
    IntegratedApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--calibration-worker":
        raise SystemExit(_calibration_worker_entry(sys.argv[2]))
    main()
