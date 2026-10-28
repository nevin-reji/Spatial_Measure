"""Saved settings: a round trip keeps every value, and a calibration inside the project is stored relative."""
import json
import os

from spatial_measure import settings


def test_round_trip(tmp_path):
    path = str(tmp_path / "settings.json")
    calib = os.path.join(settings.PROJECT_DIR, "targets", "aruco_id0_50mm.png")  # any existing file in the project
    s = settings.load(path)
    s.update(marker_mm=49.6, mode="hand", ai=False, calibration=calib)
    settings.save(s, path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["calibration"] == "targets/aruco_id0_50mm.png"
    back = settings.load(path)
    assert back["marker_mm"] == 49.6 and back["mode"] == "hand" and back["ai"] is False
    assert back["calibration"] == os.path.normpath(calib)


def test_missing_or_broken_file_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert settings.load(str(path))["marker_mm"] == settings.DEFAULTS["marker_mm"]
    assert settings.load(str(tmp_path / "none.json"))["mode"] == "table"
