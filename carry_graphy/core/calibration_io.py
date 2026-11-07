"""Reading and summarising camera calibration data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

CALIBRATION_FILE_EXTENSIONS = (".npz", ".npy")

# Called when the companion JSON (same stem) is missing. Receives the data
# path and the default JSON path; returns a JSON path or None to give up.
MetadataResolver = Callable[[Path, Path], "Path | None"]


def distortion_from_metadata(metadata: dict[str, object]) -> np.ndarray | None:
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


def image_size_from_metadata(metadata: dict[str, object]) -> np.ndarray | None:
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


def load_calibration_file(path: Path, resolve_metadata: MetadataResolver | None = None) -> dict[str, object]:
    """Load a calibration NPZ/NPY and validate it.

    Raises ``OSError``, ``EOFError``, ``ValueError``, ``KeyError``, ``TypeError``,
    ``json.JSONDecodeError`` or ``zipfile.BadZipFile`` on failure.
    """
    if path.suffix.lower() not in CALIBRATION_FILE_EXTENSIONS:
        raise ValueError("カメラ行列を含む .npz または .npy ファイルを選択してください。")

    metadata: dict[str, object] = {}
    metadata_path: Path | None = path.with_suffix(".json")
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
        assert metadata_path is not None
        if not metadata_path.is_file():
            metadata_path = resolve_metadata(path, metadata_path) if resolve_metadata else None
        if metadata_path is None:
            raise ValueError(
                "カメラの歪み係数とキャリブレーション解像度が必要です。"
                "対応するJSONを選択するか、camera_matrix・distortion_coefficients・image_sizeを含むNPZを指定してください。"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"対応JSONの形式が不正です: {metadata_path}")
        if distortion is None:
            distortion = distortion_from_metadata(metadata)
        if image_size is None:
            image_size = image_size_from_metadata(metadata)
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


def read_calibration_result(output: Path) -> dict[str, object]:
    result_path = output / "calibration.npz"
    if not result_path.is_file():
        raise ValueError(f"結果ファイルがありません: {result_path}")
    return load_calibration_file(result_path)


def write_calibration_summary(output: Path, data: dict[str, object]) -> Path:
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


def rmse_quality(rmse: object) -> str | None:
    """Classify a reprojection error: 'good', 'fair', 'poor' or None if unknown."""
    if not isinstance(rmse, (int, float, np.number)) or not np.isfinite(rmse):
        return None
    if rmse < 0.5:
        return "good"
    if rmse < 1.0:
        return "fair"
    return "poor"
