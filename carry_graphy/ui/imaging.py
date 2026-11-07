"""Conversion helpers between OpenCV images and Tk photo images."""

from __future__ import annotations

import base64
import tkinter as tk

import cv2
import numpy as np


def to_photo(image: np.ndarray, master: tk.Misc | None = None) -> tk.PhotoImage | None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return tk.PhotoImage(master=master, data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="png")


def fit_image(image: np.ndarray, max_width: int, max_height: int, allow_upscale: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)
