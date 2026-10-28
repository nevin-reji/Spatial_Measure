"""Saved settings (marker size, calibration file, mode, ...), kept in settings.json in the project folder.

app.py writes it whenever a setting changes; measure.py reads the marker size and calibration from it when no
flag is given, so the printed marker only has to be measured and entered once.
"""
from __future__ import annotations

import json
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(PROJECT_DIR, "settings.json")

DEFAULTS = {
    "marker_mm": 50.0, "calibration": None, "use_clahe": True, "refine": True, "smoothing": True,
    "scan": True, "mode": "table", "distance_mm": 400.0, "hfov_deg": 65.0, "ai": True, "camera": 0,
}


def load(path: str = PATH) -> dict:
    s = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    calib = resolve(s["calibration"])
    if calib is None and os.path.isfile(os.path.join(PROJECT_DIR, "calibration.npz")):
        calib = os.path.join(PROJECT_DIR, "calibration.npz")
    s["calibration"] = calib
    return s


def save(settings: dict, path: str = PATH):
    s = {k: settings[k] for k in DEFAULTS if k in settings}
    if s.get("calibration"):  # inside the project: store it relative, so the folder can be copied to another PC
        rel = os.path.relpath(os.path.abspath(s["calibration"]), PROJECT_DIR)
        if not rel.startswith(".."):
            s["calibration"] = rel.replace(os.sep, "/")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, path)


def resolve(calib: str | None) -> str | None:
    """Absolute path of a saved calibration file, or None if it no longer exists."""
    if not calib:
        return None
    p = calib if os.path.isabs(calib) else os.path.join(PROJECT_DIR, calib)
    return os.path.normpath(p) if os.path.isfile(p) else None
