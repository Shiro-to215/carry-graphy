"""Conversion helpers between OpenCV images and Tk photo images."""

from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np


def to_photo(image: np.ndarray, master: tk.Misc | None = None) -> tk.PhotoImage | None:
    """Convert an OpenCV image to a Tk PhotoImage using uncompressed binary PPM."""
    if image.size == 0:
        return None
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(master=master, data=header + rgb.tobytes(), format="PPM")


def fit_image(image: np.ndarray, max_width: int, max_height: int, allow_upscale: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)
