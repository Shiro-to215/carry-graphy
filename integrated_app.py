#!/usr/bin/env python3
"""Single-window workflow for camera calibration and homography creation.

The implementation lives in the :mod:`carry_graphy` package:
- ``carry_graphy.core``: calibration engine, ChArUco board generation,
  homography math/IO, and persistent settings.
- ``carry_graphy.ui``: CustomTkinter interface, drag-and-drop, interactive
  point canvas with magnifier loupe, and step pages.
"""

from __future__ import annotations

import sys

from carry_graphy.core.calibration import (
    _calibration_analyze_frame_quality,
    _calibration_collect_candidates,
    _calibration_detect_charuco,
    _calibration_frame_score,
    _calibration_is_monotonic,
    _calibration_run_solver,
    _calibration_spatial_coverage,
    _CalibrationCandidate,
    _emit_calibration_progress,
    calibration_worker_entry as _calibration_worker_entry,
    run_calibration as _calibration_worker,
)
from carry_graphy.paths import APP_DIR, DEFAULT_RESULTS_DIR


def main() -> None:
    from carry_graphy.ui.app import main as _ui_main

    _ui_main()


def __getattr__(name: str):
    if name == "IntegratedApp":
        from carry_graphy.ui.app import IntegratedApp

        return IntegratedApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "APP_DIR",
    "DEFAULT_RESULTS_DIR",
    "IntegratedApp",
    "_CalibrationCandidate",
    "_calibration_analyze_frame_quality",
    "_calibration_collect_candidates",
    "_calibration_detect_charuco",
    "_calibration_frame_score",
    "_calibration_is_monotonic",
    "_calibration_run_solver",
    "_calibration_spatial_coverage",
    "_calibration_worker",
    "_calibration_worker_entry",
    "_emit_calibration_progress",
    "main",
]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--calibration-worker":
        raise SystemExit(_calibration_worker_entry(sys.argv[2]))
    main()
