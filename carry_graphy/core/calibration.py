"""ChArUco video calibration engine.

The engine runs in a separate process (``integrated_app.py --calibration-worker``)
so that the GUI stays responsive. Progress is reported on stderr as
``PROGRESS {json}`` lines and the final summary is printed on stdout.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..paths import APP_DIR, DEFAULT_RESULTS_DIR, ENTRY_SCRIPT

WORKER_FLAG = "--calibration-worker"
ARUCO_DICTIONARIES = (
    "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_7X7_50", "DICT_7X7_100",
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


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


CalibrationProgressCallback = Callable[[dict[str, object]], None]


def _emit_calibration_progress(
    callback: CalibrationProgressCallback | None,
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
    progress_callback: CalibrationProgressCallback | None,
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


def run_calibration(config: dict[str, object], progress_callback: CalibrationProgressCallback) -> dict[str, object]:
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


def calibration_worker_entry(config_text: str) -> int:
    try:
        config = json.loads(config_text)
        if not isinstance(config, dict):
            raise ValueError("解析設定の形式が不正です")

        def write_progress(progress: dict[str, object]) -> None:
            print("PROGRESS " + json.dumps(progress, ensure_ascii=False), file=sys.stderr, flush=True)

        summary = run_calibration(config, write_progress)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


# --------------------------------------------------------------------------
# GUI-side helpers (validation and process command construction)
# --------------------------------------------------------------------------

def build_worker_config(values: dict[str, str]) -> dict[str, object]:
    """Validate raw form values and return the worker configuration.

    Raises ``ValueError`` with a user-facing Japanese message on invalid input.
    """
    video = Path(values.get("video", "").strip()).expanduser()
    if not values.get("video", "").strip() or not video.is_file():
        raise ValueError("キャリブレーション動画を選択してください。")

    try:
        integer_values = {
            key: int(values[key])
            for key in ("squares_x", "squares_y", "top_n", "stride", "min_corners", "max_frames")
        }
        square_length = float(values["square_length"])
        marker_length = float(values["marker_length"])
        blur_threshold = float(values["blur_threshold"])
        low_coverage = float(values["low_coverage"])
    except ValueError:
        raise ValueError("数値の入力を確認してください。") from None
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

    output = Path(values.get("output", "").strip() or str(DEFAULT_RESULTS_DIR / "calibration")).expanduser()
    if not output.is_absolute():
        output = APP_DIR / output
    return {
        "video": str(video.resolve()),
        "output": str(output.resolve()),
        "squares_x": integer_values["squares_x"],
        "squares_y": integer_values["squares_y"],
        "square_length": square_length,
        "marker_length": marker_length,
        "dictionary": values.get("dictionary", "DICT_4X4_50"),
        "top_n": integer_values["top_n"],
        "stride": integer_values["stride"],
        "blur_threshold": blur_threshold,
        "min_corners": integer_values["min_corners"],
        "max_frames": integer_values["max_frames"],
        "low_coverage": low_coverage,
    }


def build_worker_command(config: dict[str, object]) -> list[str]:
    return [
        sys.executable,
        str(ENTRY_SCRIPT),
        WORKER_FLAG,
        json.dumps(config, ensure_ascii=False),
    ]
