"""Synthetic end-to-end checks: render a known flat scene, project it through a known
camera pose, and verify the pipeline recovers the true dimensions and distance."""
import math

import cv2
import numpy as np
import pytest

from spatial_measure import SpatialDimensionEstimator

PPM = 8  # canvas resolution, px per mm
CANVAS_MM = (500, 350)
MARKER_MM = 50.0
MARKER_TL_MM = (40, 40)
CARD = dict(center=(250, 175), size=(85.60, 53.98), angle=20)  # credit card
DISK = dict(center=(400, 260), diameter=50.0)

K = np.array([[1142.38, 0, 961.04], [0, 1139.12, 538.22], [0, 0, 1]])  # paper Section 4.1
IMG_SIZE = (1920, 1080)


def render_plane():
    w, h = CANVAS_MM[0] * PPM, CANVAS_MM[1] * PPM
    canvas = np.full((h, w, 3), 235, np.uint8)
    side = int(MARKER_MM * PPM)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, side)
    x0, y0 = MARKER_TL_MM[0] * PPM, MARKER_TL_MM[1] * PPM
    canvas[y0 : y0 + side, x0 : x0 + side] = marker[:, :, None]
    rect = ((CARD["center"][0] * PPM, CARD["center"][1] * PPM), (CARD["size"][0] * PPM, CARD["size"][1] * PPM), CARD["angle"])
    cv2.fillPoly(canvas, [cv2.boxPoints(rect).astype(np.int32)], (40, 60, 160), cv2.LINE_AA)
    cv2.circle(canvas, (DISK["center"][0] * PPM, DISK["center"][1] * PPM), int(DISK["diameter"] / 2 * PPM), (60, 60, 60), -1, cv2.LINE_AA)
    return canvas


def camera_view(pitch_deg, distance_mm):
    """Image of the plane seen by camera K at the given pitch; plane origin at canvas centre."""
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    t = np.array([0, 0, distance_mm])
    to_plane_mm = np.array([[1 / PPM, 0, -CANVAS_MM[0] / 2], [0, 1 / PPM, -CANVAS_MM[1] / 2], [0, 0, 1]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ to_plane_mm
    # Blur first so the downsampling warp doesn't alias (a real lens low-pass filters too).
    plane = cv2.GaussianBlur(render_plane(), (0, 0), 2)
    img = cv2.warpPerspective(plane, H, IMG_SIZE, flags=cv2.INTER_LINEAR, borderValue=(90, 90, 90))
    marker_centre = np.array([MARKER_TL_MM[0] + MARKER_MM / 2 - CANVAS_MM[0] / 2, MARKER_TL_MM[1] + MARKER_MM / 2 - CANVAS_MM[1] / 2, 0])
    return img, float(np.linalg.norm(R @ marker_centre + t))


def find(measurements, length, width, tol):
    return [m for m in measurements if abs(m.length_mm - length) < tol and abs(m.width_mm - width) < tol]


@pytest.mark.parametrize("pitch", [0, 25, 45])
def test_dimensions_and_distance(pitch):
    img, true_dist = camera_view(pitch, 600)
    est = SpatialDimensionEstimator(marker_length_mm=MARKER_MM, camera_matrix=K, dist_coeffs=np.zeros(5))
    r = est.process(img)

    assert r.marker_found and r.marker_id == 7
    assert len(find(r.measurements, *CARD["size"], tol=1.0)) == 1, [(m.length_mm, m.width_mm) for m in r.measurements]
    assert len(find(r.measurements, DISK["diameter"], DISK["diameter"], tol=1.0)) == 1
    assert r.distance_pnp_mm == pytest.approx(true_dist, rel=0.02)
    assert r.tilt_deg == pytest.approx(pitch, abs=2)


def test_without_calibration_still_measures():
    img, _ = camera_view(30, 600)
    r = SpatialDimensionEstimator(marker_length_mm=MARKER_MM).process(img)
    assert len(find(r.measurements, *CARD["size"], tol=1.0)) == 1
    assert r.distance_pnp_mm is None


def test_no_marker():
    blank = np.full((480, 640, 3), 200, np.uint8)
    r = SpatialDimensionEstimator().process(blank)
    assert not r.marker_found and r.measurements == []
