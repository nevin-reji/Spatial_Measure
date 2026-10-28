"""Learned promptable segmentation: click a point, get the mask of the object under it.

MobileSAM (a distilled Segment Anything model, trained on the SA-1B dataset: 11 M images, 1.1 B masks) is run
through ultralytics. Its job here is only to say *which pixels are the object*; the geometry (homography, scale,
minAreaRect) stays exactly as before, so the measurement is still metric.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mobile_sam.pt")


class SamSegmenter:
    def __init__(self, weights: str = WEIGHTS):
        import torch
        from ultralytics import SAM

        os.makedirs(os.path.dirname(weights), exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SAM(weights)  # downloads the official MobileSAM weights on first use

    def segment(self, frame_bgr: np.ndarray, point: tuple[float, float], extra_points=()) -> np.ndarray | None:
        """Mask of the object at `point`. extra_points: more positive clicks on the *same* object (taken from the
        previous frame's mask), which keeps the answer on the whole object instead of drifting to a part of it."""
        pts = [[float(point[0]), float(point[1])]] + [[float(x), float(y)] for x, y in extra_points]
        res = self.model.predict(frame_bgr, points=[pts], labels=[[1] * len(pts)], device=self.device, verbose=False)
        if not res or res[0].masks is None or len(res[0].masks.data) == 0:
            return None
        m = res[0].masks.data[0].cpu().numpy()
        if m.shape != frame_bgr.shape[:2]:
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8)
