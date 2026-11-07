"""Interactive image canvas for selecting and dragging the four homography corners.

Features:
* Click to place up to 4 points (左下 → 右下 → 右上 → 左上).
* Drag an existing point to reposition it (with live polygon and convexity feedback).
* Magnifier loupe around the cursor while dragging or while hovering over the image.
* Mouse wheel zoom (anchored at the cursor) and middle-drag / Space+drag pan.
* Auto-fit when the canvas is resized at 100% zoom.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable

import cv2
import numpy as np

from ..core.homography import POINT_LABELS
from . import theme as t
from .imaging import to_photo

HIT_RADIUS_PX = 14
MARKER_RADIUS_PX = 7
LOUPE_SIZE_PX = 156
LOUPE_SOURCE_RADIUS = 18  # +/- pixels in source image -> ~4.3x magnification on the source


def is_quad_valid(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        return True
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(contour) and abs(cv2.contourArea(contour)) >= 10)


class PointCanvas:
    """Manages a ``tk.Canvas`` showing an undistorted reference image and 4 corners."""

    def __init__(
        self,
        parent: tk.Misc,
        image: np.ndarray,
        points: list[tuple[float, float]],
        *,
        on_points_changed: Callable[[], None],
        on_zoom_changed: Callable[[float], None],
        on_cursor_changed: Callable[[tuple[float, float] | None, int | None], None] | None = None,
        loupe_enabled: tk.BooleanVar | None = None,
    ) -> None:
        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        self.points = points
        self.on_points_changed = on_points_changed
        self.on_zoom_changed = on_zoom_changed
        self.on_cursor_changed = on_cursor_changed
        self.loupe_enabled = loupe_enabled

        self.viewport_width = 880
        self.viewport_height = 500
        self.fit_scale = min(self.viewport_width / self.image_width, self.viewport_height / self.image_height, 1.0)
        self.zoom_factor = 1.0
        self.display_scale = self.fit_scale
        self.display_width = max(1, round(self.image_width * self.display_scale))
        self.display_height = max(1, round(self.image_height * self.display_scale))
        self.display_scale_x = self.display_width / self.image_width
        self.display_scale_y = self.display_height / self.image_height

        self.photo: tk.PhotoImage | None = None
        self.loupe_photo: tk.PhotoImage | None = None
        self.dragging_index: int | None = None
        self.hover_index: int | None = None
        self.space_held = False
        self.panning = False
        self._last_cursor_event: tuple[int, int] | None = None

        self.canvas = tk.Canvas(
            parent,
            width=self.viewport_width,
            height=self.viewport_height,
            highlightthickness=1,
            highlightbackground=t.BORDER_STRONG,
            background=t.CANVAS_BG,
            xscrollincrement=24,
            yscrollincrement=24,
        )
        self._bind_events()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def canvas_to_image(self, canvas_x: float, canvas_y: float, *, clamp: bool = True) -> tuple[float, float] | None:
        if not clamp and not (0 <= canvas_x < self.display_width and 0 <= canvas_y < self.display_height):
            return None
        x = canvas_x / self.display_scale_x
        y = canvas_y / self.display_scale_y
        if clamp:
            x = min(max(x, 0.0), float(self.image_width - 1))
            y = min(max(y, 0.0), float(self.image_height - 1))
        return x, y

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.display_scale_x, y * self.display_scale_y

    def hit_test(self, canvas_x: float, canvas_y: float) -> int | None:
        best_index: int | None = None
        best_dist = HIT_RADIUS_PX + 1.0
        for index, (x, y) in enumerate(self.points):
            px, py = self.image_to_canvas(x, y)
            dist = math.hypot(canvas_x - px, canvas_y - py)
            if dist <= HIT_RADIUS_PX and dist < best_dist:
                best_dist = dist
                best_index = index
        return best_index

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_image(self, anchor: tuple[int, int] | None = None) -> None:
        if not self.canvas.winfo_exists():
            return
        old_image_x = old_image_y = None
        if anchor is not None:
            old_image_x = self.canvas.canvasx(anchor[0]) / self.display_scale_x
            old_image_y = self.canvas.canvasy(anchor[1]) / self.display_scale_y

        self.display_scale = self.fit_scale * self.zoom_factor
        self.display_width = max(1, round(self.image_width * self.display_scale))
        self.display_height = max(1, round(self.image_height * self.display_scale))
        self.display_scale_x = self.display_width / self.image_width
        self.display_scale_y = self.display_height / self.image_height
        interpolation = cv2.INTER_AREA if self.display_scale < 1 else cv2.INTER_LINEAR
        shown = cv2.resize(self.image, (self.display_width, self.display_height), interpolation=interpolation)
        self.photo = to_photo(shown, self.canvas)
        if self.photo is None:
            return
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="image")
        scroll_width = max(self.display_width, self.viewport_width)
        scroll_height = max(self.display_height, self.viewport_height)
        self.canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        self.on_zoom_changed(self.zoom_factor)

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
        self.canvas.tag_raise("loupe")

    def redraw_points(self) -> None:
        if not self.canvas.winfo_exists():
            return
        self.canvas.delete("marker")
        valid = is_quad_valid(self.points)
        line_color = t.ACCENT if valid else t.DANGER
        screen_pts = [self.image_to_canvas(x, y) for x, y in self.points]

        # Connecting edges so self-intersection is immediately visible
        if len(screen_pts) >= 2:
            loop = screen_pts + ([screen_pts[0]] if len(screen_pts) == 4 else [])
            for (x0, y0), (x1, y1) in zip(loop, loop[1:]):
                self.canvas.create_line(x0, y0, x1, y1, fill="#FFFFFF", width=5, tags="marker")
                self.canvas.create_line(
                    x0, y0, x1, y1, fill=line_color, width=2.5,
                    dash=() if valid else (6, 4), tags="marker",
                )

        font = t.fonts().canvas(11, "bold")
        for index, (shown_x, shown_y) in enumerate(screen_pts):
            active = index in (self.dragging_index, self.hover_index)
            fill = line_color
            radius = MARKER_RADIUS_PX + (2 if active else 0)
            if active:
                self.canvas.create_oval(
                    shown_x - radius - 5, shown_y - radius - 5,
                    shown_x + radius + 5, shown_y + radius + 5,
                    outline="#FFFFFF", width=2, tags="marker",
                )
            self.canvas.create_oval(
                shown_x - radius, shown_y - radius, shown_x + radius, shown_y + radius,
                fill=fill, outline="#FFFFFF", width=2.5, tags="marker",
            )
            # Crosshair at the exact sub-pixel point
            self.canvas.create_line(shown_x - 3, shown_y, shown_x + 3, shown_y, fill="#FFFFFF", width=1.5, tags="marker")
            self.canvas.create_line(shown_x, shown_y - 3, shown_x, shown_y + 3, fill="#FFFFFF", width=1.5, tags="marker")

            # Pill label clamped inside the image so top/right points remain readable
            tag_text = f" {index + 1}: {POINT_LABELS[index]} "
            dx = -12 if shown_x > self.display_width - 110 else 12
            dy = 14 if shown_y < 36 else -14
            anchor = ("e" if dx < 0 else "w")
            text_id = self.canvas.create_text(
                shown_x + dx, shown_y + dy, text=tag_text, fill="#FFFFFF",
                font=font, anchor=anchor, tags="marker",
            )
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg_id = self.canvas.create_rectangle(
                    bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2,
                    fill=fill, outline="#FFFFFF", width=1.5, tags="marker",
                )
                self.canvas.tag_lower(bg_id, text_id)

        self.canvas.tag_raise("loupe")
        self.on_points_changed()

    def _draw_loupe(self, event_x: int, event_y: int, img_x: float, img_y: float) -> None:
        self.canvas.delete("loupe")
        if self.loupe_enabled is not None and not self.loupe_enabled.get():
            return
        if self.panning:
            return
        r = LOUPE_SOURCE_RADIUS
        cx, cy = int(round(img_x)), int(round(img_y))
        padded = cv2.copyMakeBorder(self.image, r, r, r, r, cv2.BORDER_CONSTANT, value=(35, 38, 45))
        patch = padded[cy:cy + 2 * r + 1, cx:cx + 2 * r + 1]
        if patch.size == 0:
            return
        zoomed = cv2.resize(patch, (LOUPE_SIZE_PX, LOUPE_SIZE_PX), interpolation=cv2.INTER_NEAREST)
        self.loupe_photo = to_photo(zoomed, self.canvas)
        if self.loupe_photo is None:
            return

        # Place in the top-right of the visible viewport; flip to top-left if the cursor is nearby
        vw = max(self.canvas.winfo_width(), 300)
        left_x = self.canvas.canvasx(16)
        right_x = self.canvas.canvasx(vw - LOUPE_SIZE_PX - 16)
        top_y = self.canvas.canvasy(16)
        box_x = left_x if (event_x > vw - LOUPE_SIZE_PX - 64 and event_y < LOUPE_SIZE_PX + 72) else right_x
        box_y = top_y

        half = LOUPE_SIZE_PX / 2
        mid_x, mid_y = box_x + half, box_y + half
        self.canvas.create_rectangle(
            box_x - 3, box_y - 3, box_x + LOUPE_SIZE_PX + 3, box_y + LOUPE_SIZE_PX + 30,
            fill=t.TEXT, outline="#FFFFFF", width=2, tags="loupe",
        )
        self.canvas.create_image(box_x, box_y, image=self.loupe_photo, anchor="nw", tags="loupe")
        # Crosshair inside the loupe
        self.canvas.create_line(box_x, mid_y, box_x + LOUPE_SIZE_PX, mid_y, fill="#FFFFFF", width=2, tags="loupe")
        self.canvas.create_line(mid_x, box_y, mid_x, box_y + LOUPE_SIZE_PX, fill="#FFFFFF", width=2, tags="loupe")
        self.canvas.create_line(box_x, mid_y, box_x + LOUPE_SIZE_PX, mid_y, fill=t.ACCENT, width=1, tags="loupe")
        self.canvas.create_line(mid_x, box_y, mid_x, box_y + LOUPE_SIZE_PX, fill=t.ACCENT, width=1, tags="loupe")
        self.canvas.create_rectangle(box_x, box_y, box_x + LOUPE_SIZE_PX, box_y + LOUPE_SIZE_PX,
                                     outline="#FFFFFF", width=1.5, tags="loupe")
        target = self.dragging_index if self.dragging_index is not None else (
            len(self.points) if len(self.points) < 4 else None
        )
        caption = f"{POINT_LABELS[target]}  ({img_x:.0f}, {img_y:.0f})" if target is not None else f"({img_x:.0f}, {img_y:.0f})"
        self.canvas.create_text(
            mid_x, box_y + LOUPE_SIZE_PX + 14, text=caption, fill="#FFFFFF",
            font=t.fonts().canvas(10, "bold"), tags="loupe",
        )

    # ------------------------------------------------------------------
    # Zoom & pan
    # ------------------------------------------------------------------

    def change_zoom(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        if anchor is None and self.canvas.winfo_exists():
            anchor = (self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)
        self.zoom_factor = min(max(self.zoom_factor * factor, 0.5), 6.0)
        self.render_image(anchor)
        self.redraw_points()

    def fit(self) -> None:
        self.zoom_factor = 1.0
        self.render_image()
        self.redraw_points()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.redraw_points()

    def reset(self) -> None:
        self.points.clear()
        self.redraw_points()

    def refresh_loupe(self) -> None:
        if self._last_cursor_event is None:
            self.canvas.delete("loupe")
            return
        ex, ey = self._last_cursor_event
        pt = self.canvas_to_image(self.canvas.canvasx(ex), self.canvas.canvasy(ey), clamp=False)
        if pt is None:
            self.canvas.delete("loupe")
        else:
            self._draw_loupe(ex, ey, pt[0], pt[1])

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<Configure>", self._on_configure)
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", self._on_leave)
        c.bind("<MouseWheel>", self._on_mousewheel)
        c.bind("<Button-4>", lambda e: self._zoom_from_event(e, 1.25))
        c.bind("<Button-5>", lambda e: self._zoom_from_event(e, 1 / 1.25))
        c.bind("<ButtonPress-2>", self._start_pan)
        c.bind("<B2-Motion>", self._pan_image)
        c.bind("<ButtonRelease-2>", self._end_pan)
        c.bind("<ButtonPress-3>", self._start_pan)
        c.bind("<B3-Motion>", self._pan_image)
        c.bind("<ButtonRelease-3>", self._end_pan)

    def _on_configure(self, event: tk.Event) -> None:
        if event.width < 80 or event.height < 80:
            return
        if (event.width, event.height) == (self.viewport_width, self.viewport_height):
            return
        self.viewport_width = event.width
        self.viewport_height = event.height
        new_fit = min(self.viewport_width / self.image_width, self.viewport_height / self.image_height, 1.0)
        if abs(new_fit - self.fit_scale) > 1e-3:
            self.fit_scale = new_fit
            self.render_image()
            self.redraw_points()

    def _on_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.space_held:
            self._start_pan(event)
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        hit = self.hit_test(cx, cy)
        if hit is not None:
            self.dragging_index = hit
            self.hover_index = hit
            self.canvas.configure(cursor="fleur")
            self._move_dragged_point(event.x, event.y)
            return
        if len(self.points) >= 4:
            return
        pt = self.canvas_to_image(cx, cy, clamp=False)
        if pt is None:
            return
        self.points.append(pt)
        self.dragging_index = len(self.points) - 1
        self.hover_index = self.dragging_index
        self.redraw_points()
        self._draw_loupe(event.x, event.y, pt[0], pt[1])

    def _on_drag(self, event: tk.Event) -> None:
        if self.panning:
            self._pan_image(event)
            return
        if self.dragging_index is None:
            return
        self._move_dragged_point(event.x, event.y)

    def _move_dragged_point(self, event_x: int, event_y: int) -> None:
        assert self.dragging_index is not None
        cx = self.canvas.canvasx(event_x)
        cy = self.canvas.canvasy(event_y)
        pt = self.canvas_to_image(cx, cy, clamp=True)
        assert pt is not None
        self.points[self.dragging_index] = pt
        self._last_cursor_event = (event_x, event_y)
        self.redraw_points()
        self._draw_loupe(event_x, event_y, pt[0], pt[1])
        if self.on_cursor_changed:
            self.on_cursor_changed(pt, self.dragging_index)

    def _on_release(self, event: tk.Event) -> None:
        if self.panning:
            self._end_pan(event)
            return
        if self.dragging_index is not None:
            self.dragging_index = None
            self.redraw_points()
            self._on_motion(event)

    def _on_motion(self, event: tk.Event) -> None:
        self._last_cursor_event = (event.x, event.y)
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        hit = self.hit_test(cx, cy)
        if hit != self.hover_index:
            self.hover_index = hit
            self.redraw_points()
        pt = self.canvas_to_image(cx, cy, clamp=False)
        if self.space_held:
            self.canvas.configure(cursor="fleur")
        elif hit is not None:
            self.canvas.configure(cursor="fleur")
        elif pt is not None and len(self.points) < 4:
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(cursor="")
        if pt is not None:
            self._draw_loupe(event.x, event.y, pt[0], pt[1])
        else:
            self.canvas.delete("loupe")
        if self.on_cursor_changed:
            self.on_cursor_changed(pt, hit)

    def _on_leave(self, _event: tk.Event) -> None:
        self._last_cursor_event = None
        if self.hover_index is not None and self.dragging_index is None:
            self.hover_index = None
            self.redraw_points()
        self.canvas.delete("loupe")
        if self.on_cursor_changed:
            self.on_cursor_changed(None, None)

    def _zoom_from_event(self, event: tk.Event, factor: float) -> str:
        self.change_zoom(factor, (event.x, event.y))
        return "break"

    def _on_mousewheel(self, event: tk.Event) -> str:
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        return self._zoom_from_event(event, factor)

    def _start_pan(self, event: tk.Event) -> None:
        self.panning = True
        self.canvas.delete("loupe")
        self.canvas.configure(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def _pan_image(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _end_pan(self, _event: tk.Event) -> None:
        self.panning = False
        self.canvas.configure(cursor="fleur" if self.space_held else "")

    def set_space_held(self, held: bool) -> None:
        self.space_held = held
        if not self.panning and self.canvas.winfo_exists():
            self.canvas.configure(cursor="fleur" if held else "")
