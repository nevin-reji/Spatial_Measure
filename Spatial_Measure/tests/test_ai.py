"""Learned outline (MobileSAM). Skipped when ultralytics or the weights are not installed."""
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")
from spatial_measure import SpatialDimensionEstimator  # noqa: E402
from spatial_measure.segmenter import WEIGHTS, SamSegmenter  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="models/mobile_sam.pt not downloaded")


def test_ai_outline_measures_the_tapped_object_not_the_background():
    est = SpatialDimensionEstimator(marker_length_mm=50)
    est.segmenter = SamSegmenter()
    W, H = 1280, 720
    f = (W / 2) / np.tan(np.radians(65) / 2)
    ppm = f / 400
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(190, 14, (H, W, 3)), 0, 255).astype(np.uint8)  # cluttered, noisy background
    cv2.fillPoly(img, [cv2.boxPoints(((640, 360), (int(85.6 * ppm), int(54 * ppm)), 20)).astype(np.int32)], (170, 110, 60))
    m = est.process_nomarker(img, (640.0, 360.0), 400.0, 65.0).selected
    assert m is not None and abs(m.length_mm - 85.6) < 4 and abs(m.width_mm - 54.0) < 4
