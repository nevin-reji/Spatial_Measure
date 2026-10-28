"""Object held in the hand: the tapped object is followed in the image while it moves and rotates."""
import cv2
import numpy as np

from spatial_measure import SpatialDimensionEstimator


def test_moving_rotating_object_is_followed_and_measured():
    est = SpatialDimensionEstimator(marker_length_mm=50)
    t = np.full((900, 1200), 255, np.uint8)
    t[100:350, 100:350] = cv2.resize(cv2.imread("targets/aruco_id0_50mm.png", 0), (250, 250), interpolation=cv2.INTER_NEAREST)
    c, _ = est.detect_marker(t)
    ppm = np.linalg.norm(c[0] - c[1]) / 50
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [1200, 0], [1200, 900], [0, 900]]),
                                     np.float32([[80, 40], [1150, 90], [1120, 860], [40, 800]]))
    w, h = int(85.6 * ppm), int(54 * ppm)

    def frame(i):
        s = cv2.cvtColor(t.copy(), cv2.COLOR_GRAY2BGR)
        cx, cy = 480 + i * 12, 350 + 30 * np.sin(i / 5)
        cv2.fillPoly(s, [cv2.boxPoints(((cx, cy), (w, h), i * 1.5)).astype(np.int32)], (170, 110, 60))
        return cv2.warpPerspective(s, Hm, (1200, 900), borderValue=(255, 255, 255)), (cx, cy)

    img, (cx, cy) = frame(0)
    seed = tuple(map(float, cv2.perspectiveTransform(np.float32([[[cx, cy]]]), Hm)[0, 0]))
    lengths = []
    for i in range(25):
        r = est.process(frame(i)[0], seed_px=seed, follow=True, skin=True)
        assert r.selected is not None, f"lost the object at frame {i}"
        seed = r.seed_px
        lengths.append(r.selected.length_mm)
    assert all(abs(L - 85.6) < 3 for L in lengths)


def test_no_marker_mode_scales_with_the_assumed_distance():
    import math
    W, H = 1280, 720
    f = (W / 2) / math.tan(math.radians(65) / 2)
    ppm = f / 400  # the card really is 400 mm away
    img = np.full((H, W, 3), (200, 205, 210), np.uint8)
    cv2.fillPoly(img, [cv2.boxPoints(((640, 360), (int(85.6 * ppm), int(54 * ppm)), 20)).astype(np.int32)], (170, 110, 60))
    est = SpatialDimensionEstimator(marker_length_mm=50)
    right = est.process_nomarker(img, (640.0, 360.0), 400.0, 65.0).selected
    assert abs(right.length_mm - 85.6) < 2.5 and abs(right.width_mm - 54.0) < 2.5
    wrong = est.process_nomarker(img, (640.0, 360.0), 500.0, 65.0).selected  # 25 % too far -> 25 % too big
    assert abs(wrong.length_mm / right.length_mm - 1.25) < 0.03
