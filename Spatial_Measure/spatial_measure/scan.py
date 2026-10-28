"""Measuring objects that are bigger than one camera frame.

Everything is expressed in millimetres on the marker plane (origin = marker top-left corner), which
does not move when the camera does. Two pieces make that work:

* PlaneTracker keeps knowing where the plane is while the marker is out of view. Whenever the marker
  is visible it gives an exact image->mm homography; when it is not, ORB features on the table are matched
  against stored keyframes whose homography is known, and the two homographies are chained.
* ScanMap accumulates, frame after frame, which parts of the plane have been *seen* and which of those
  belong to the tapped object, then measures the object from the consensus. It also knows whether the
  object's outline touches an unseen area, i.e. whether the size so far is only a lower bound.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np


class PlaneTracker:
    def __init__(self, marker_mm: float, max_keyframes: int = 40, feature_width: int = 960):
        self.marker_mm = marker_mm
        self.max_keyframes = max_keyframes
        self.feature_width = feature_width
        self._orb = cv2.ORB_create(1200, scaleFactor=1.2, nlevels=6, fastThreshold=12)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self):
        self.keyframes: list[dict] = []
        self.M: np.ndarray | None = None  # image (undistorted px) -> plane mm, for the latest frame
        self.ready = False
        self.frames_since_marker = 0
        self.last_center: np.ndarray | None = None

    # ------------------------------------------------------------------ features

    def _features(self, gray: np.ndarray):
        s = min(1.0, self.feature_width / gray.shape[1])
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        kps, des = self._orb.detectAndCompute(small, None)
        if des is None or len(kps) < 12:
            return None, None
        return (np.float32([k.pt for k in kps]) / s), des

    @staticmethod
    def _mm(M: np.ndarray, pts) -> np.ndarray:
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), M).reshape(-1, 2)

    def _maybe_keyframe(self, gray: np.ndarray, M: np.ndarray):
        h, w = gray.shape[:2]
        c = self._mm(M, [(w / 2, h / 2)])[0]
        a, b = self._mm(M, [(0, h / 2), (w, h / 2)])
        span = float(np.linalg.norm(a - b))
        if not np.isfinite(span) or span <= 0:
            return
        if any(np.linalg.norm(k["center"] - c) < 0.25 * span for k in self.keyframes):
            return
        pts, des = self._features(gray)
        if pts is None:
            return
        self.keyframes.append({"pts": pts, "des": des, "M": M.copy(), "center": c})
        if len(self.keyframes) > self.max_keyframes:
            self.keyframes.pop(1 if len(self.keyframes) > 2 else 0)  # keep the first (marker) keyframe

    # ------------------------------------------------------------------- public

    def anchor(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """The marker is visible: exact plane homography."""
        L = self.marker_mm
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [L, 0], [L, L], [0, L]]))
        self.M, self.ready, self.frames_since_marker = M, True, 0
        h, w = gray.shape[:2]
        self.last_center = self._mm(M, [(w / 2, h / 2)])[0]
        self._maybe_keyframe(gray, M)
        return M

    def track(self, gray: np.ndarray) -> np.ndarray | None:
        """The marker is out of view: chain a homography from the best-matching keyframe."""
        if not self.ready or not self.keyframes:
            return None
        pts, des = self._features(gray)
        if pts is None:
            return None
        order = sorted(self.keyframes, key=lambda k: np.linalg.norm(k["center"] - self.last_center)
                       if self.last_center is not None else 0.0)
        for kf in order[:4]:
            pairs = self._bf.knnMatch(des, kf["des"], k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
            if len(good) < 20:
                continue
            src = pts[[m.queryIdx for m in good]]
            dst = kf["pts"][[m.trainIdx for m in good]]
            Hk, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if Hk is None or int(inl.sum()) < 18 or inl.mean() < 0.35:
                continue
            scale = float(np.sqrt(abs(np.linalg.det(Hk[:2, :2]))))
            if not 0.4 < scale < 2.5:
                continue
            M = kf["M"] @ Hk
            h, w = gray.shape[:2]
            c = self._mm(M, [(w / 2, h / 2)])[0]
            if not np.all(np.isfinite(c)):
                continue
            self.M, self.last_center = M, c
            self.frames_since_marker += 1
            self._maybe_keyframe(gray, M)
            return M
        return None

    def virtual_corners(self, M: np.ndarray) -> np.ndarray:
        """Where the marker's corners would be in the current image (possibly outside it)."""
        L = self.marker_mm
        return self._mm(np.linalg.inv(M), [(0, 0), (L, 0), (L, L), (0, L)]).astype(np.float32)


class ScanMap:
    def __init__(self, ppm: float = 2.0, half_mm: float = 1000.0, freeze: int = 60):
        self.ppm, self.freeze = ppm, freeze
        self.G = int(2 * half_mm * ppm)
        self.c = half_mm * ppm
        self.reset()

    def reset(self):
        self.obj = np.zeros((self.G, self.G), np.uint8)
        self.obs = np.zeros((self.G, self.G), np.uint8)
        self.bbox: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 of the object votes
        self._cons = None  # latest consensus object (x0, y0, x1, y1, mask), for consistency checks

    def _rasterize(self, cnt, valid, canvas_ppm, origin):
        """Frame's object mask and observed mask, warped into the global map. Returns (x0, y0, x1, y1, obs, obj)."""
        k = self.ppm / canvas_ppm
        tx, ty = self.c - origin[0] * k, self.c - origin[1] * k
        ch, cw = valid.shape[:2]
        x0, y0 = max(0, int(np.floor(tx))), max(0, int(np.floor(ty)))
        x1, y1 = min(self.G, int(np.ceil(k * cw + tx))), min(self.G, int(np.ceil(k * ch + ty)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        A = np.float64([[k, 0, tx - x0], [0, k, ty - y0]])
        size = (x1 - x0, y1 - y0)
        obs_m = cv2.warpAffine((valid > 0).astype(np.uint8), A, size, flags=cv2.INTER_NEAREST)
        full = np.zeros((ch, cw), np.uint8)
        cv2.drawContours(full, [cnt], -1, 1, cv2.FILLED)
        obj_m = cv2.warpAffine(full, A, size, flags=cv2.INTER_NEAREST) & obs_m
        return x0, y0, x1, y1, obs_m, obj_m

    def consistent(self, cnt, valid, canvas_ppm, origin) -> bool:
        """Does this frame's object outline agree with what earlier frames established for the parts it sees again?
        Rejects a frame that latched onto a mark or letter instead of the object."""
        if self._cons is None:
            return True
        r = self._rasterize(cnt, valid, canvas_ppm, origin)
        if r is None:
            return True
        x0, y0, x1, y1, obs_m, obj_m = r
        cx0, cy0, cx1, cy1, comp = self._cons
        ix0, iy0, ix1, iy1 = max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)
        if ix1 <= ix0 or iy1 <= iy0:
            return True
        known = comp[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0].astype(bool) & obs_m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(bool)
        n = int(known.sum())
        if n < 200:
            return True
        hit = int((known & obj_m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(bool)).sum())
        return hit / n >= 0.6

    def add(self, cnt: np.ndarray, valid: np.ndarray, canvas_ppm: float, origin) -> None:
        """cnt: object outline in canvas px; valid: canvas mask of real pixels; origin: canvas px of mm (0, 0)."""
        r = self._rasterize(cnt, valid, canvas_ppm, origin)
        if r is None:
            return
        x0, y0, x1, y1, obs_m, obj_m = r
        r_obs, r_obj = self.obs[y0:y1, x0:x1], self.obj[y0:y1, x0:x1]
        live = r_obs < self.freeze
        np.add(r_obs, obs_m, out=r_obs, where=live)
        np.add(r_obj, obj_m, out=r_obj, where=live)
        bx, by, bw, bh = cv2.boundingRect(obj_m)
        if bw and bh:
            nb = (x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh)
            self.bbox = nb if self.bbox is None else (min(self.bbox[0], nb[0]), min(self.bbox[1], nb[1]),
                                                     max(self.bbox[2], nb[2]), max(self.bbox[3], nb[3]))

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_px, np.float64) - self.c) / self.ppm

    def measure(self, seed_mm) -> dict | None:
        if self.bbox is None:
            return None
        pad = 8
        x0, y0 = max(0, self.bbox[0] - pad), max(0, self.bbox[1] - pad)
        x1, y1 = min(self.G, self.bbox[2] + pad), min(self.G, self.bbox[3] + pad)
        obj, obs = self.obj[y0:y1, x0:x1].astype(np.uint16), self.obs[y0:y1, x0:x1]
        cons = ((obj * 2 > obs) & (obs > 0)).astype(np.uint8)
        cons = cv2.morphologyEx(cons, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cons, connectivity=8)
        if n < 2:
            return None
        sx = int(round(seed_mm[0] * self.ppm + self.c)) - x0 if seed_mm is not None else -1
        sy = int(round(seed_mm[1] * self.ppm + self.c)) - y0 if seed_mm is not None else -1
        if 0 <= sx < lab.shape[1] and 0 <= sy < lab.shape[0] and lab[sy, sx] > 0:
            idx = int(lab[sy, sx])
        else:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = (lab == idx).astype(np.uint8)
        self._cons = (x0, y0, x1, y1, comp)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
        ring = cv2.dilate(comp, np.ones((5, 5), np.uint8)) & (1 - comp)
        is_open = bool(np.any((ring > 0) & (obs == 0)))
        box = cv2.boxPoints(((cx, cy), (w, h), ang)) + np.float32([x0, y0])
        poly = cv2.approxPolyDP(cnt, 0.006 * cv2.arcLength(cnt, True), True).reshape(-1, 2) + np.int32([x0, y0])
        return {
            "length_mm": max(w, h) / self.ppm, "width_mm": min(w, h) / self.ppm,
            "area_mm2": float(comp.sum()) / (self.ppm ** 2), "open": is_open,
            "outline_mm": self.to_mm(poly), "box_mm": self.to_mm(box),
            "center_mm": self.to_mm(np.array([[cx + x0, cy + y0]]))[0],
            "_crop": (x0, y0, x1, y1, comp),
        }

    def seed_candidates(self, n: int = 24) -> list[tuple[float, float]]:
        """Deep-interior points of the object found so far (mm), spread along it, for re-seeding when the
        original seed has left the camera's view."""
        m = self.measure(None)
        if m is None:
            return []
        x0, y0, x1, y1, comp = m["_crop"]
        d = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        if d.max() < 3:
            return []
        ys, xs = np.nonzero(d >= 0.6 * d.max())
        if len(xs) == 0:
            return []
        idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
        order = np.lexsort((ys, xs))
        pts = np.column_stack([xs[order][idx] + x0, ys[order][idx] + y0])
        return [tuple(p) for p in self.to_mm(pts)]

    def preview_png_b64(self, m: dict, max_px: int = 240) -> str:
        """Small map: grey = seen, green = the object, black = never seen. For the sidebar."""
        x0, y0, x1, y1, comp = m["_crop"]
        pad = 40
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(self.G, x1 + pad), min(self.G, y1 + pad)
        seen = self.obs[Y0:Y1, X0:X1] > 0
        img = np.zeros((Y1 - Y0, X1 - X0, 3), np.uint8)
        img[seen] = (70, 70, 70)
        sub = img[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        sub[comp > 0] = (80, 200, 80)
        s = min(1.0, max_px / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf.tobytes()).decode() if ok else ""
