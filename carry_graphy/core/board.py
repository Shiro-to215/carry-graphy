"""ChArUco board image generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_BOARD_PIXELS = 60_000_000
BOARD_MARGIN_PX = 50


class BoardInputError(ValueError):
    """Invalid board settings entered by the user."""


class BoardSizeError(ValueError):
    """The requested board would produce an excessively large image."""


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str
    dpi: int

    def as_tuple(self) -> tuple[int, int, float, float, str, int]:
        return (
            self.squares_x, self.squares_y, self.square_length,
            self.marker_length, self.dictionary, self.dpi,
        )


def parse_board_spec(values: dict[str, str]) -> BoardSpec:
    try:
        squares_x = int(values["squares_x"])
        squares_y = int(values["squares_y"])
        square_length = float(values["square_length"])
        marker_length = float(values["marker_length"])
        dpi = int(values["dpi"])
    except ValueError:
        raise BoardInputError("ボード設定を確認してください。数値を入力する欄に文字が入っています。") from None
    if not np.isfinite([square_length, marker_length]).all():
        raise BoardInputError("マスとマーカーの長さには有限の数値を入力してください。")
    if squares_x < 2 or squares_y < 2:
        raise BoardInputError("ボードのマス数は横・縦ともに2以上にしてください。")
    if not 0 < marker_length < square_length:
        raise BoardInputError("マーカーの一辺は、マスの一辺より短くしてください。")
    if not 72 <= dpi <= 2400:
        raise BoardInputError("解像度は72〜2400 dpiで指定してください。")
    return BoardSpec(squares_x, squares_y, square_length, marker_length, values["dictionary"], dpi)


def board_pixel_size(spec: BoardSpec) -> tuple[int, int]:
    pixel_width = spec.squares_x * spec.square_length * spec.dpi / 25.4
    pixel_height = spec.squares_y * spec.square_length * spec.dpi / 25.4
    if (
        not np.isfinite([pixel_width, pixel_height]).all()
        or pixel_width < 1
        or pixel_height < 1
        or pixel_width * pixel_height > MAX_BOARD_PIXELS
    ):
        raise BoardSizeError("指定した設定ではボード画像が大きすぎます。dpiまたはボード寸法を小さくしてください。")
    return int(pixel_width), int(pixel_height)


def generate_board_image(spec: BoardSpec) -> np.ndarray:
    """Render the board (with a white margin). Raises cv2.error / ValueError."""
    image_width, image_height = board_pixel_size(spec)
    if not hasattr(cv2, "aruco"):
        raise ValueError("OpenCV ArUcoモジュールがありません。opencv-contrib-pythonを確認してください。")
    dictionary_id = getattr(cv2.aruco, spec.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"不明なArUco辞書です: {spec.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y), spec.square_length, spec.marker_length, dictionary
    )
    image = board.generateImage((image_width, image_height))
    return cv2.copyMakeBorder(
        image,
        BOARD_MARGIN_PX, BOARD_MARGIN_PX, BOARD_MARGIN_PX, BOARD_MARGIN_PX,
        cv2.BORDER_CONSTANT, value=[255, 255, 255],
    )


def save_board_image(spec: BoardSpec, output: Path) -> np.ndarray:
    if output.suffix.lower() != ".png":
        raise BoardInputError("ボード画像の保存先は .png にしてください。")
    board_pixel_size(spec)  # validate size before heavy work
    image = generate_board_image(spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"画像を保存できませんでした: {output}")
    output.write_bytes(encoded.tobytes())
    return image
