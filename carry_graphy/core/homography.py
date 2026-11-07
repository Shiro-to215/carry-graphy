"""Homography creation, persistence and resolution alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

POINT_LABELS = ("左下", "右下", "右上", "左上")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
HOMOGRAPHY_FILE_EXTENSIONS = (".npy", ".npz")


def read_image_color(path: Path) -> np.ndarray | None:
    """Read a BGR image from a path that may contain non-ASCII characters on Windows."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

ALIGNMENT_LABELS = {
    "same_resolution": "キャリブレーション解像度と同一（調整なし）",
    "reference_resized_to_calibration_resolution": "基準画像をキャリブレーション解像度にリサイズ",
    "calibration_intrinsics_scaled_to_reference_resolution": "カメラ行列を基準画像の解像度にスケーリング",
}


class HomographyPointError(ValueError):
    """Selected points are not a valid convex quadrilateral."""


class HomographyComputeError(ValueError):
    """OpenCV could not compute a finite homography."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_homography_file(path: Path) -> tuple[np.ndarray, bool]:
    """Return ``(matrix, has_embedded_metadata)`` from a NPY/NPZ file."""
    has_embedded_metadata = False
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
    return matrix, has_embedded_metadata


# --------------------------------------------------------------------------
# Input validation and resolution alignment
# --------------------------------------------------------------------------

def parse_plane_size(width_text: str, height_text: str) -> tuple[float, float]:
    try:
        width_m = float(width_text)
        height_m = float(height_text)
    except ValueError:
        raise ValueError("平面の幅と高さを数値で入力してください。") from None
    if not np.isfinite([width_m, height_m]).all() or width_m <= 0 or height_m <= 0:
        raise ValueError("平面の幅と高さは有限の正の値にしてください。")
    return width_m, height_m


def align_to_calibration_resolution(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    reference_size: tuple[int, int],
    calibration_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Resize the reference image to the calibration resolution."""
    cal_width, cal_height = calibration_size
    ref_width, ref_height = reference_size
    interpolation = cv2.INTER_AREA if cal_width < ref_width or cal_height < ref_height else cv2.INTER_CUBIC
    resized = cv2.resize(image, calibration_size, interpolation=interpolation)
    alignment = {
        "mode": "reference_resized_to_calibration_resolution",
        "source_image_size": [ref_width, ref_height],
        "processing_image_size": [cal_width, cal_height],
        "scale_xy": [cal_width / ref_width, cal_height / ref_height],
    }
    return resized, camera_matrix, alignment


def align_to_reference_resolution(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    reference_size: tuple[int, int],
    calibration_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Scale the camera intrinsics to the reference image resolution."""
    ref_width, ref_height = reference_size
    cal_width, cal_height = calibration_size
    scaled = camera_matrix.copy()
    scale_x, scale_y = ref_width / cal_width, ref_height / cal_height
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    alignment = {
        "mode": "calibration_intrinsics_scaled_to_reference_resolution",
        "calibration_image_size": [cal_width, cal_height],
        "processing_image_size": [ref_width, ref_height],
        "scale_xy": [scale_x, scale_y],
    }
    return image, scaled, alignment


SAME_RESOLUTION_ALIGNMENT: dict[str, object] = {"mode": "same_resolution", "scale_xy": [1.0, 1.0]}


# --------------------------------------------------------------------------
# Computation and saving
# --------------------------------------------------------------------------

def compute_homography(
    points: list[tuple[float, float]], width_m: float, height_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(matrix, source_points, destination_points)``."""
    if len(points) != 4:
        raise HomographyPointError("四隅を4点すべて選択してください。")
    source = np.asarray(points, dtype=np.float32)
    contour = source.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 10:
        raise HomographyPointError("左下 → 右下 → 右上 → 左上の順に、四隅を選択してください。")
    destination = np.asarray(
        [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
        dtype=np.float32,
    )
    matrix, _ = cv2.findHomography(source, destination, method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise HomographyComputeError("ホモグラフィ行列を計算できませんでした。")
    return matrix, source, destination


def build_metadata(
    *,
    image_size: tuple[int, int],
    image_path: Path,
    camera_matrix: np.ndarray,
    calibration_camera_matrix: np.ndarray,
    calibration_image_size: tuple[int, int],
    image_alignment: dict[str, object],
    distortion: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    width_m: float,
    height_m: float,
) -> dict[str, Any]:
    return {
        "format": "homography_with_metadata_v1",
        "coordinate_system": "vertical_plane, meters; origin=left_bottom",
        "image_size": [image_size[0], image_size[1]],
        "image_file": str(image_path),
        "camera_matrix": camera_matrix.tolist(),
        "calibration_camera_matrix": calibration_camera_matrix.tolist(),
        "calibration_image_size": list(calibration_image_size),
        "image_alignment": image_alignment,
        "distortion_coefficients_k1_k2_p1_p2_k3": distortion.tolist(),
        "image_points_undistorted_px": source.tolist(),
        "plane_points_m": destination.tolist(),
        "plane_width_m": width_m,
        "plane_height_m": height_m,
        "point_order": list(POINT_LABELS),
    }


def save_homography(output_path: Path, matrix: np.ndarray, metadata: dict[str, Any]) -> Path:
    """Write the packaged NPY and the readable TXT. Returns the TXT path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    record_dtype = np.dtype([
        ("homography_matrix", np.float64, (3, 3)),
        ("metadata_json", f"<U{max(1, len(metadata_text))}"),
    ])
    packaged = np.zeros((), dtype=record_dtype)
    packaged["homography_matrix"] = matrix
    packaged["metadata_json"] = metadata_text
    np.save(output_path, packaged, allow_pickle=False)
    text_path = output_path.with_suffix(".txt")
    text_path.write_text(format_homography_summary(matrix, metadata), encoding="utf-8")
    return text_path


def format_homography_summary(matrix: np.ndarray, metadata: dict[str, Any]) -> str:
    matrix_rows = [
        "[ " + ",  ".join(str(float(value)) for value in row) + " ]"
        for row in matrix
    ]
    image_width, image_height = metadata["image_size"]
    alignment = metadata["image_alignment"]
    alignment_mode = alignment.get("mode")
    alignment_text = ALIGNMENT_LABELS.get(alignment_mode, str(alignment_mode))
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
        + f"座標基準の画像解像度: {image_width} × {image_height} px\n"
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


def format_matrix(matrix: np.ndarray, precision: str = ".8g") -> str:
    return "\n".join("[ " + ",  ".join(f"{value:{precision}}" for value in row) + " ]" for row in matrix)
