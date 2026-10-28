"""Tap-to-measure: the tapped object is outlined, measured in mm, and followed via its marker-plane seed."""
import cv2
import numpy as np

from spatial_measure import SpatialDimensionEstimator


def _scene():
    est = SpatialDimensionEstimator(marker_length_mm=50)
    s = np.full((900, 1200), 255, np.uint8)
    marker = cv2.imread("targets/aruco_id0_50mm.png", 0)
    s[100:350, 100:350] = cv2.resize(marker, (250, 250), interpolation=cv2.INTER_NEAREST)
    c, _ = est.detect_marker(s)
    ppm = np.linalg.norm(c[0] - c[1]) / 50
    s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    w, h = int(85.6 * ppm), int(54 * ppm)
    card = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        card[:, x] = (200 - 40 * x / w, 120, 60)
    s[250:250 + h, 500:500 + w] = card
    cv2.circle(s, (300, 700), int(15 * ppm), (40, 160, 40), -1)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [1200, 0], [1200, 900], [0, 900]]),
                                     np.float32([[80, 40], [1150, 90], [1120, 860], [40, 800]]))
    img = cv2.warpPerspective(s, Hm, (1200, 900), borderValue=(255, 255, 255))
    tap = lambda p: tuple(map(float, cv2.perspectiveTransform(np.float32([[p]]), Hm)[0, 0]))
    return img, tap, (500 + w / 2, 250 + h / 2 + 25)


def test_tap_measures_card_and_disc():
    img, tap, card_pt = _scene()
    r = SpatialDimensionEstimator(marker_length_mm=50).process(img, seed_px=tap(card_pt))
    assert r.selection_state == "ok" and r.selected.outline_image is not None
    assert abs(r.selected.length_mm - 85.6) < 2.5 and abs(r.selected.width_mm - 54.0) < 2.5
    r = SpatialDimensionEstimator(marker_length_mm=50).process(img, seed_px=tap((300, 700)))
    assert abs(r.selected.length_mm - 30.0) < 1.5


def test_selection_is_tracked_and_marker_is_rejected():
    img, tap, card_pt = _scene()
    est = SpatialDimensionEstimator(marker_length_mm=50)
    r = est.process(img, seed_px=tap(card_pt))
    again = est.process(img, seed_mm=r.seed_mm)
    assert again.selection_state == "ok" and abs(again.selected.length_mm - r.selected.length_mm) < 0.5
    assert est.process(img, seed_px=tap((150, 150))).selection_state != "ok"
