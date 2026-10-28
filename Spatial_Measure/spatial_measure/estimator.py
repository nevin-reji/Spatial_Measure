"""Core pipeline from the paper (Section 3):

    frame -> lens undistortion -> ArUco reference detection -> planar homography
          -> top-down rectification -> edge extraction -> contours
          -> minAreaRect -> pixel-to-mm conversion -> overlay

All measurements assume the objects lie (approximately) on the same plane as the
reference marker. Tall objects violate that assumption (see paper Section 7,
"Non-planar depth variances").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Measurement:
    length_mm: float  # longer side of the rotated bounding box
    width_mm: float  # shorter side
    angle_deg: float
    center_mm: tuple[float, float]  # on the marker plane, relative to marker top-left corner
    box_image: np.ndarray  # 4x2 box corners in (undistorted) image pixels
    center_image: tuple[float, float]
    outline_image: np.ndarray | None = None  # object outline (simplified polygon), undistorted image pixels
    area_mm2: float | None = None  # area enclosed by the outline


@dataclass
class FrameResult:
    frame: np.ndarray  # undistorted input frame
    marker_found: bool
    marker_id: int | None = None
    marker_corners: np.ndarray | None = None  # 4x2, image pixels (TL, TR, BR, BL)
    warped: np.ndarray | None = None  # top-down rectified view
    edges: np.ndarray | None = None
    measurements: list[Measurement] = field(default_factory=list)
    distance_similar_mm: float | None = None  # D = W * f / P  (paper Section 5.2)
    distance_pnp_mm: float | None = None  # full pose estimate, robust to tilt
    tilt_deg: float | None = None  # angle between optical axis and marker normal
    warnings: list[str] = field(default_factory=list)
    # Tap-to-measure: the selected object, its seed on the marker plane (mm, for tracking it
    # across frames) and a state: "ok" | "nothing" | "marker" | "lost" | None.
    selected: Measurement | None = None
    seed_mm: tuple[float, float] | None = None
    selection_state: str | None = None
    # Scan mode: how the plane was located ("marker" | "tracked"), the image->canvas homography with the canvas
    # scale/origin (to map mm back to pixels), and what ScanMap needs to accumulate the object.
    plane_source: str | None = None
    plane_map: tuple | None = None  # (H image->canvas, ppm, origin canvas px of mm (0,0))
    scan_data: dict | None = None
    seed_px: tuple[float, float] | None = None  # follow mode: where to look for the object next frame (raw pixels)


def valid_small(valid: np.ndarray, w: int, h: int) -> np.ndarray:
    return valid if valid.shape[:2] == (h, w) else cv2.resize(valid, (w, h), interpolation=cv2.INTER_NEAREST)


class SpatialDimensionEstimator:
    def __init__(
        self,
        marker_length_mm: float = 50.0,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        px_per_mm: float = 4.0,
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
        marker_id: int | None = None,
        min_area_mm2: float = 150.0,
        use_clahe: bool = True,
        canny_thresholds: tuple[int, int] = (50, 150),
        max_canvas_px: int = 2400,
        min_px_per_mm: float = 2.0,
        refine_edges: bool = True,
    ):
        self.marker_length_mm = marker_length_mm
        self.K = camera_matrix
        self.dist = dist_coeffs if dist_coeffs is not None else (np.zeros(5) if camera_matrix is not None else None)
        self.px_per_mm = px_per_mm
        self.marker_id = marker_id
        self.min_area_mm2 = min_area_mm2
        self.use_clahe = use_clahe
        self.canny_thresholds = canny_thresholds
        self.max_canvas_px = max_canvas_px
        self.min_px_per_mm = min_px_per_mm
        self.refine_edges = refine_edges

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco_dict), params)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.fill_tol = (9, 7, 7)  # flood-fill tolerance in Lab (L, a, b)
        self.hint: np.ndarray | None = None  # last mask from the segmenter: anchors the next frame on the same object
        self.segmenter = None  # optional learned promptable segmenter (spatial_measure.segmenter), see _ai_mask

        # Undistortion maps are expensive to build, so cache them per frame size.
        self._maps = None
        self._maps_size = None
        self._new_K = camera_matrix

    # ------------------------------------------------------------------ stages

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Removes radial and tangential lens distortion (paper Section 2.2)."""
        if self.K is None:
            return frame
        h, w = frame.shape[:2]
        if self._maps_size != (w, h):
            # alpha=0 keeps only valid pixels at full frame size, so no ROI crop is needed
            # and the principal point stays consistent with new_K.
            self._new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 0, (w, h))
            self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self._new_K, (w, h), cv2.CV_16SC2)
            self._maps_size = (w, h)
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def to_raw_pixels(self, pts: np.ndarray) -> np.ndarray:
        """Maps Nx2 points from the undistorted frame back onto the raw (distorted) camera frame,
        e.g. to draw results over a live preview that shows the raw feed."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.K is None or self._maps_size is None:
            return pts
        rays = np.column_stack([pts, np.ones(len(pts))]) @ np.linalg.inv(self._new_K).T
        raw, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.dist)
        return raw.reshape(-1, 2)

    def detect_marker(self, gray: np.ndarray) -> tuple[np.ndarray | None, int | None]:
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None, None
        ids = ids.flatten()
        if self.marker_id is not None:
            matches = np.where(ids == self.marker_id)[0]
            if len(matches) == 0:
                return None, None
            i = int(matches[0])
        else:
            i = int(np.argmax([cv2.contourArea(c.reshape(-1, 2)) for c in corners]))
        return corners[i].reshape(4, 2).astype(np.float32), int(ids[i])

    def compute_homography(self, marker_corners: np.ndarray, frame_shape) -> tuple[np.ndarray, tuple[int, int], np.ndarray, float, list[str]]:
        """Maps the marker plane to a top-down canvas where 1 mm == ppm pixels.

        The marker's 4 corners give exactly the N >= 4 correspondences the DLT needs
        (paper Section 2.3). The canvas is translated/sized to cover the whole frame; if that
        would exceed max_canvas_px, the resolution is lowered (down to min_px_per_mm) and only
        then is the canvas clipped around the projected image centre.

        Returns (H, canvas size, marker polygon in canvas coords, effective ppm, warnings).
        """
        warnings = []
        h, w = frame_shape[:2]
        frame_pts = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1], [w / 2, h / 2, 1]], np.float64).T

        ppm = self.px_per_mm
        for _ in range(2):
            s = self.marker_length_mm * ppm
            dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
            H = cv2.getPerspectiveTransform(marker_corners, dst)
            proj = H @ frame_pts
            horizon_visible = np.any(proj[2, :4] <= 1e-9)
            if horizon_visible:
                break
            pts = proj[:2] / proj[2]
            x_min, y_min = pts[:, :4].min(axis=1)
            x_max, y_max = pts[:, :4].max(axis=1)
            extent = max(x_max - x_min, y_max - y_min)
            if extent <= self.max_canvas_px or ppm <= self.min_px_per_mm:
                break
            ppm = max(self.min_px_per_mm, ppm * self.max_canvas_px / extent)

        half = self.max_canvas_px / 2
        if horizon_visible:
            # Part of the image maps to infinity; keep a window around the marker.
            warnings.append("Extreme tilt: horizon visible, canvas clipped around marker")
            x_min, y_min, x_max, y_max = s / 2 - half, s / 2 - half, s / 2 + half, s / 2 + half
        elif extent > self.max_canvas_px:
            cx, cy = pts[:, 4]
            x_min, x_max = max(x_min, cx - half), min(x_max, cx + half)
            y_min, y_max = max(y_min, cy - half), min(y_max, cy + half)

        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float64)
        size = (int(math.ceil(x_max - x_min)), int(math.ceil(y_max - y_min)))
        marker_poly = dst - np.float32([x_min, y_min])
        return T @ H, size, marker_poly, ppm, warnings

    def preprocess(self, warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(gray, edge map) of the rectified view.

        CLAHE on the LAB L-channel is the shadow mitigation from paper Section 6.1.
        """
        if self.use_clahe:
            gray = self._clahe.apply(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 0])
        else:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, *self.canny_thresholds)
        # Morphological closing consolidates broken edges (contour fragmentation).
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return gray, cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    def refine_contour(self, gray: np.ndarray, cnt: np.ndarray) -> np.ndarray:
        """Re-locates the object boundary at the local Otsu threshold.

        Canny + closing places the outer contour 1-2 px outside the true edge (more when the
        rectified view is upsampled), which systematically inflates sizes. Thresholding the
        object's neighbourhood at the midpoint between object and background intensity removes
        that bias. Falls back to the Canny contour when the object is not a clean blob.
        """
        x, y, w, h = cv2.boundingRect(cnt)
        pad = max(10, int(0.15 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        roi = gray[y0:y1, x0:x1]
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
        if np.mean(border) > 127:  # background is the bright class -> object is dark
            binary = cv2.bitwise_not(binary)

        filled = np.zeros_like(binary)
        cv2.drawContours(filled, [cnt - np.int32([x0, y0])], -1, 255, cv2.FILLED)
        candidates, _ = cv2.findContours(cv2.bitwise_and(binary, filled), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not candidates:
            return cnt
        best = max(candidates, key=cv2.contourArea)
        ratio = cv2.contourArea(best) / max(cv2.contourArea(cnt), 1.0)
        if not 0.8 <= ratio <= 1.0:
            return cnt
        return best + np.int32([x0, y0])

    @staticmethod
    def _object_contours(edges: np.ndarray, marker_center: np.ndarray, min_area: float) -> list[np.ndarray]:
        """Outermost contours that are objects rather than the surface the marker sits on.

        A contour enclosing the marker (a sheet of paper, a mat, the table edge) is treated as
        background and we look inside it instead, so objects on a sheet are still found.
        """
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        hierarchy = hierarchy[0]
        mc = (float(marker_center[0]), float(marker_center[1]))
        found = []
        stack = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
        while stack:
            i = stack.pop()
            if cv2.pointPolygonTest(contours[i], mc, False) >= 0:
                child = hierarchy[i][2]
                while child != -1:
                    stack.append(child)
                    child = hierarchy[child][0]
            elif cv2.contourArea(contours[i]) >= min_area:
                found.append(contours[i])
        return found

    def measure_contours(self, gray: np.ndarray, edges: np.ndarray, H: np.ndarray, marker_poly: np.ndarray, ppm: float) -> list[Measurement]:
        H_inv = np.linalg.inv(H)
        origin = marker_poly[0]
        # Grow the marker square 20% so the marker itself (and its paper border) is skipped.
        mc = marker_poly.mean(axis=0)
        marker_zone = ((marker_poly - mc) * 1.2 + mc).astype(np.float32)

        results = []
        for cnt in self._object_contours(edges, mc, self.min_area_mm2 * ppm * ppm):
            if self.refine_edges:
                cnt = self.refine_contour(gray, cnt)
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_px, h_px), angle = rect
            if cv2.pointPolygonTest(marker_zone, (float(cx), float(cy)), False) >= 0:
                continue
            if min(w_px, h_px) < 2 * ppm:  # slivers: shadows, table seams, stray edge fragments
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            box_img = cv2.perspectiveTransform(box.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
            c_img = cv2.perspectiveTransform(np.array([[[cx, cy]]], np.float32), H_inv).reshape(2)
            results.append(
                Measurement(
                    length_mm=max(w_px, h_px) / ppm,
                    width_mm=min(w_px, h_px) / ppm,
                    angle_deg=angle,
                    center_mm=((cx - origin[0]) / ppm, (cy - origin[1]) / ppm),
                    box_image=box_img,
                    center_image=(float(c_img[0]), float(c_img[1])),
                )
            )
        results.sort(key=lambda m: m.length_mm * m.width_mm, reverse=True)
        return results

    def estimate_distance(self, marker_corners: np.ndarray) -> tuple[float | None, float | None, float | None]:
        """Camera-to-marker distance.

        Returns (similar-triangles D = W*f/P, solvePnP distance, tilt in degrees).
        Similar triangles is only valid when the marker faces the camera; PnP is not.
        """
        if self._new_K is None:
            return None, None, None
        K = self._new_K
        f = (K[0, 0] + K[1, 1]) / 2
        sides = np.linalg.norm(marker_corners - np.roll(marker_corners, -1, axis=0), axis=1)
        d_similar = self.marker_length_mm * f / sides.mean()

        L = self.marker_length_mm / 2
        obj = np.array([[-L, L, 0], [L, L, 0], [L, -L, 0], [-L, -L, 0]], np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, marker_corners, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return d_similar, None, None
        R, _ = cv2.Rodrigues(rvec)
        tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
        return d_similar, float(np.linalg.norm(tvec)), tilt


    # ----------------------------------------------------------- tap to measure

    def _region_at(self, warped: np.ndarray, edges: np.ndarray, valid: np.ndarray, seed: tuple[float, float], max_frac: float = 0.5, skin: bool = False) -> np.ndarray | None:
        """Region of similar colour around the seed, walled in by the Canny edges.

        Works on a downscaled copy (cheap enough to run every frame) and returns the outline in
        canvas pixels, or None when the fill leaks across the scene or is too small to be an object.
        """
        h, w = warped.shape[:2]
        k = min(1.0, 700.0 / max(h, w))
        small = cv2.resize(warped, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else warped
        sh, sw = small.shape[:2]
        lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 0)
        wall = cv2.resize(edges, (sw, sh), interpolation=cv2.INTER_NEAREST) if k < 1 else edges.copy()
        wall = cv2.dilate(wall, np.ones((3, 3), np.uint8))
        wall = np.where(wall > 0, 1, 0).astype(np.uint8)
        wall[valid_small(valid, sw, sh) == 0] = 1
        if skin:
            # Fingers holding the object are walled off so the fill does not run into the hand.
            ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
            hand = cv2.inRange(ycc, (0, 135, 85), (255, 175, 125))
            hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            hand[max(0, int(seed[1] * k) - 1):int(seed[1] * k) + 2, max(0, int(seed[0] * k) - 1):int(seed[0] * k) + 2] = 0
            wall[hand > 0] = 1
        mask = np.zeros((sh + 2, sw + 2), np.uint8)
        mask[1:-1, 1:-1] = wall
        sx, sy = int(round(seed[0] * k)), int(round(seed[1] * k))
        if not (0 <= sx < sw and 0 <= sy < sh) or wall[sy, sx]:
            # seed sits exactly on an edge pixel: nudge to a free neighbour
            ok = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 2), (-2, -2), (2, -2), (-2, 2)):
                x2, y2 = sx + dx, sy + dy
                if 0 <= x2 < sw and 0 <= y2 < sh and not wall[y2, x2]:
                    sx, sy, ok = x2, y2, True
                    break
            if not ok:
                return None
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(lab, mask, (sx, sy), 0, self.fill_tol, self.fill_tol, flags)
        region = (mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # The walls keep the fill ~1.5 px short of the true edge; grow back, refine_contour trims the excess.
        region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.pointPolygonTest(c, (float(sx), float(sy)), False) >= 0]
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        frac = cv2.contourArea(cnt) / float(max(1, np.count_nonzero(valid_small(valid, sw, sh))))
        if frac > max_frac:  # leaked into the background
            return None
        return (cnt.reshape(-1, 1, 2).astype(np.float32) / k).astype(np.int32)

    def _contour_at(self, edges: np.ndarray, seed: tuple[float, float], min_area: float) -> np.ndarray | None:
        """Fallback: the smallest edge contour that contains the seed and is big enough."""
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best, best_a = None, None
        for c in contours:
            a = cv2.contourArea(c)
            if a >= min_area and cv2.pointPolygonTest(c, seed, False) >= 0 and (best_a is None or a < best_a):
                best, best_a = c, a
        return best


    def _ai_mask(self, frame: np.ndarray, seed: tuple[float, float], skin: bool) -> np.ndarray | None:
        """Mask (frame pixels) of the object under `seed` from the learned segmenter, with the hand taken out."""
        m = self.segmenter.segment(frame, seed, self._hint_points(frame.shape[:2]))
        if m is None:
            self.hint = None
            return None
        m = m.astype(np.uint8)
        if skin:
            hand = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb), (0, 135, 85), (255, 175, 125))
            hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            m[hand > 0] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n < 2:
            self.hint = None
            return None
        sx, sy = int(round(seed[0])), int(round(seed[1]))
        inside = 0 <= sx < m.shape[1] and 0 <= sy < m.shape[0] and lab[sy, sx] > 0
        idx = int(lab[sy, sx]) if inside else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        self.hint = (lab == idx).astype(np.uint8)
        return self.hint

    def _hint_points(self, shape, n: int = 4):
        """A few well-spread points deep inside the previous mask (farthest-point sampling), as extra clicks."""
        if self.hint is None or self.hint.shape != tuple(shape):
            return []
        d = cv2.distanceTransform(self.hint, cv2.DIST_L2, 3)
        if d.max() < 4:
            return []
        ys, xs = np.nonzero(d >= 0.5 * d.max())
        if len(xs) == 0:
            return []
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) > 4000:
            pts = pts[np.linspace(0, len(pts) - 1, 4000).astype(int)]
        chosen = [pts[int(np.argmax(d[pts[:, 1].astype(int), pts[:, 0].astype(int)]))]]
        dist = np.linalg.norm(pts - chosen[0], axis=1)
        for _ in range(n - 1):
            chosen.append(pts[int(np.argmax(dist))])
            dist = np.minimum(dist, np.linalg.norm(pts - chosen[-1], axis=1))
        return [(float(x), float(y)) for x, y in chosen[1:]]

    def _largest_contour(self, mask: np.ndarray, seed=None) -> np.ndarray | None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        if seed is not None:
            hit = [c for c in cnts if cv2.pointPolygonTest(c, (float(seed[0]), float(seed[1])), False) >= 0]
            cnts = hit or cnts
        return max(cnts, key=cv2.contourArea)

    def segment_image(self, frame: np.ndarray, seed: tuple[float, float], ppm: float = 1.0, skin: bool = True):
        """Outline of the object under `seed` in a plain image (no plane, no marker). Returns (contour, gray) with
        contour None when nothing sensible is found. Uses the learned segmenter when one is attached."""
        h, w = frame.shape[:2]
        gray, edges = self.preprocess(frame)
        cnt = None
        if self.segmenter is not None:
            m = self._ai_mask(frame, seed, skin)
            cnt = None if m is None else self._largest_contour(m, seed)
            if cnt is not None and cv2.contourArea(cnt) < 20.0 * ppm * ppm:
                cnt = None
        if cnt is None:
            valid = np.full((h, w), 255, np.uint8)
            valid[:3], valid[-3:], valid[:, :3], valid[:, -3:] = 0, 0, 0, 0
            min_area = 20.0 * ppm * ppm
            cnt = self._region_at(frame, edges, valid, seed, 0.97, skin)
            if cnt is None or cv2.contourArea(cnt) < min_area:
                cnt = self._contour_at(edges, seed, min_area)
            if cnt is not None and cv2.contourArea(cnt) < 150.0 * ppm * ppm:
                outer = self._contour_at(edges, seed, 3.0 * cv2.contourArea(cnt))
                if outer is not None and cv2.contourArea(outer) <= 60.0 * cv2.contourArea(cnt):
                    cnt = outer
            if cnt is not None and self.refine_edges:
                cnt = self.refine_contour(gray, cnt)
        return cnt, gray

    def select_object(self, warped, gray, edges, valid, H, marker_poly, ppm, seed, max_frac=0.5, skin=False, ai=None):
        """Outlines and measures the object under `seed` (canvas pixels). Returns Measurement or None."""
        mc = marker_poly.mean(axis=0)
        marker_zone = ((marker_poly - mc) * 1.2 + mc).astype(np.float32)
        if cv2.pointPolygonTest(marker_zone, (float(seed[0]), float(seed[1])), False) >= 0:
            return "marker"
        min_area = 20.0 * ppm * ppm  # 20 mm2
        cnt = ai(seed) if ai is not None else None  # learned segmentation first, classical as the fallback
        if cnt is not None and cv2.contourArea(cnt) < min_area:
            cnt = None
        ai_used = cnt is not None
        if cnt is None:
            cnt = self._region_at(warped, edges, valid, seed, max_frac, skin)
        if not ai_used and (cnt is None or cv2.contourArea(cnt) < min_area):
            cnt = self._contour_at(edges, seed, min_area)
        if not ai_used and cnt is not None and cv2.contourArea(cnt) < 150.0 * ppm * ppm:
            # A small patch is probably print or a mark on something bigger (a letter on a card):
            # prefer the enclosing outline when it is of comparable scale.
            outer = self._contour_at(edges, seed, 3.0 * cv2.contourArea(cnt))
            if outer is not None and cv2.contourArea(outer) <= 60.0 * cv2.contourArea(cnt):
                cnt = outer
        if cnt is None:
            return None
        if self.refine_edges and not ai_used:
            cnt = self.refine_contour(gray, cnt)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_px, h_px), angle = rect
        if min(w_px, h_px) < 2 * ppm:
            return None
        H_inv = np.linalg.inv(H)
        origin = marker_poly[0]
        box = cv2.boxPoints(rect).astype(np.float32)
        box_img = cv2.perspectiveTransform(box.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        c_img = cv2.perspectiveTransform(np.array([[[cx, cy]]], np.float32), H_inv).reshape(2)
        poly = cv2.approxPolyDP(cnt, 0.008 * cv2.arcLength(cnt, True), True).astype(np.float32)
        outline = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        return Measurement(
            length_mm=max(w_px, h_px) / ppm, width_mm=min(w_px, h_px) / ppm, angle_deg=angle,
            center_mm=((cx - origin[0]) / ppm, (cy - origin[1]) / ppm), box_image=box_img,
            center_image=(float(c_img[0]), float(c_img[1])), outline_image=outline,
            area_mm2=float(cv2.contourArea(cnt)) / (ppm * ppm),
        ), cnt

    def _inner_point(self, cnt: np.ndarray) -> tuple[float, float]:
        """A point well inside the outline (peak of the distance transform), used to follow the object."""
        x, y, w, h = cv2.boundingRect(cnt)
        m = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(m, [cnt - np.int32([x - 1, y - 1])], -1, 255, cv2.FILLED)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        _, _, _, loc = cv2.minMaxLoc(d)
        return float(loc[0] + x - 1), float(loc[1] + y - 1)

    def process_nomarker(self, frame: np.ndarray, seed_px: tuple[float, float], distance_mm: float,
                         hfov_deg: float = 65.0) -> FrameResult:
        """No marker: scale comes from an assumed camera-to-object distance and the lens' focal length
        (from the calibration if there is one, else from the horizontal field of view). The object is taken to
        face the camera, so this is an estimate: size error is about the distance error, in proportion."""
        frame = self.undistort(frame)
        h, w = frame.shape[:2]
        f = float(self._new_K[0, 0]) if self._new_K is not None else (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        ppm = f / max(distance_mm, 1.0)  # image pixels per mm at the object's distance
        p = np.array([[seed_px]], np.float32)
        if self.K is not None:
            p = cv2.undistortPoints(p, self.K, self.dist, P=self._new_K)
        seed = (float(p.reshape(2)[0]), float(p.reshape(2)[1]))
        warnings = [f"Estimate: assumes the object is {distance_mm / 10:.0f} cm from the camera"]
        cnt, _ = self.segment_image(frame, seed, ppm, skin=True)
        if cnt is None:
            return FrameResult(frame=frame, marker_found=False, selection_state="nothing", warnings=warnings)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_px, h_px), angle = rect
        if min(w_px, h_px) < 2 * ppm:
            return FrameResult(frame=frame, marker_found=False, selection_state="nothing", warnings=warnings)
        poly = cv2.approxPolyDP(cnt, 0.008 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(np.float32)
        m = Measurement(
            length_mm=max(w_px, h_px) / ppm, width_mm=min(w_px, h_px) / ppm, angle_deg=angle,
            center_mm=(cx / ppm, cy / ppm), box_image=cv2.boxPoints(rect).astype(np.float32),
            center_image=(float(cx), float(cy)), outline_image=poly, area_mm2=float(cv2.contourArea(cnt)) / (ppm * ppm),
        )
        ix, iy = self._inner_point(cnt)
        nxt = tuple(float(v) for v in self.to_raw_pixels(np.array([[ix, iy]]))[0])
        return FrameResult(frame=frame, marker_found=False, selected=m, selection_state="ok",
                           seed_px=nxt, plane_source="assumed", warnings=warnings)

    # --------------------------------------------------------------- pipeline

    def process(self, frame: np.ndarray, seed_px: tuple[float, float] | None = None,
                seed_mm: tuple[float, float] | None = None, tracker=None,
                extra_seeds_mm=None, validate=None, follow: bool = False, skin: bool = False) -> FrameResult:
        """seed_px: a tap in raw camera pixels. seed_mm: a previously selected object, as a point on the
        marker plane. With either, only that object is outlined and measured."""
        frame = self.undistort(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, mid = self.detect_marker(gray)
        source = "marker"
        if tracker is not None:
            # Scan mode: keep the plane located even when the marker is out of view.
            if corners is not None:
                tracker.anchor(gray, corners)
            else:
                M = tracker.track(gray)
                if M is not None:
                    corners, mid, source = tracker.virtual_corners(M), None, "tracked"
        if corners is None:
            return FrameResult(frame=frame, marker_found=False, warnings=["Reference marker not found"])

        H, size, marker_poly, ppm, warnings = self.compute_homography(corners, frame.shape)
        warped = cv2.warpPerspective(frame, H, size)
        # Ignore the artificial edge between real pixels and the black fill outside the frame.
        valid = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), H, size)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        gray_w, edges = self.preprocess(warped)
        edges = cv2.bitwise_and(edges, valid)

        d_similar, d_pnp, tilt = self.estimate_distance(corners) if source == "marker" else (None, None, None)
        if tilt is not None and tilt > 65:
            warnings.append(f"Pitch {tilt:.0f} deg > 65 deg: perspective compression, accuracy degraded")

        selected, new_seed, state, scan_data, next_px = None, None, None, None, None
        if seed_px is not None or seed_mm is not None:
            origin = marker_poly[0]
            if seed_px is not None:
                p = np.array([[seed_px]], np.float32)
                if self.K is not None:
                    p = cv2.undistortPoints(p, self.K, self.dist, P=self._new_K)
                seed_c = cv2.perspectiveTransform(p.reshape(1, 1, 2), H).reshape(2)
            else:
                seed_c = np.array([seed_mm[0] * ppm + origin[0], seed_mm[1] * ppm + origin[1]], np.float32)
            # The tracked seed first; in scan mode, when it has left the view, other known points of the same
            # object that are in view (extra_seeds_mm) take over.
            ai = None
            if self.segmenter is not None:
                H_inv_frame = np.linalg.inv(H)

                def ai(sc, _frame=frame, _H=H, _Hi=H_inv_frame, _size=size, _valid=valid):
                    si = cv2.perspectiveTransform(np.array([[sc]], np.float32), _Hi).reshape(2)
                    m = self._ai_mask(_frame, (float(si[0]), float(si[1])), skin)
                    if m is None:
                        return None
                    mc = cv2.warpPerspective(m, _H, _size, flags=cv2.INTER_NEAREST) & (_valid > 0).astype(np.uint8)
                    return self._largest_contour(mc, sc)

            tries = [seed_c]
            for sm in (extra_seeds_mm or []):
                tries.append(np.array([sm[0] * ppm + origin[0], sm[1] * ppm + origin[1]], np.float32))
            res, used = None, None
            for k, sc in enumerate(tries):
                x, y = int(sc[0]), int(sc[1])
                if not (8 <= x < size[0] - 8 and 8 <= y < size[1] - 8) or valid[y, x] == 0:
                    continue
                res = self.select_object(warped, gray_w, edges, valid, H, marker_poly, ppm,
                                         (float(sc[0]), float(sc[1])), 0.97 if tracker is not None else 0.5, skin, ai)
                if res is not None and res != "marker":
                    if validate is not None and not validate(res[1], valid, ppm, marker_poly[0]):
                        res = None
                        continue
                    used = sc
                    break
                if k > 0:
                    res = None
            if res == "marker":
                state = "marker"
            elif res is None:
                state = "nothing" if seed_px is not None else "lost"
                if seed_mm is not None:
                    new_seed = seed_mm
            else:
                selected, cnt = res
                if tracker is not None:
                    scan_data = {"cnt": cnt, "valid": valid, "ppm": ppm, "origin": marker_poly[0]}
                if not follow and cv2.pointPolygonTest(cnt, (float(used[0]), float(used[1])), False) >= 0:
                    ix, iy = float(used[0]), float(used[1])  # keep a seed that is still inside: steadier than re-picking
                else:
                    ix, iy = self._inner_point(cnt)
                new_seed = ((ix - origin[0]) / ppm, (iy - origin[1]) / ppm)
                state = "ok"
                if follow:
                    p_img = cv2.perspectiveTransform(np.array([[[ix, iy]]], np.float32), np.linalg.inv(H)).reshape(1, 2)
                    next_px = tuple(float(v) for v in self.to_raw_pixels(p_img)[0])

        return FrameResult(
            frame=frame,
            marker_found=source == "marker",
            marker_id=mid,
            marker_corners=corners,
            warped=warped,
            edges=edges,
            measurements=[] if (seed_px is not None or seed_mm is not None) else self.measure_contours(gray_w, edges, H, marker_poly, ppm),
            selected=selected, seed_mm=new_seed, selection_state=state,
            plane_source=source, plane_map=(H, ppm, marker_poly[0]), scan_data=scan_data, seed_px=next_px,
            distance_similar_mm=d_similar,
            distance_pnp_mm=d_pnp,
            tilt_deg=tilt,
            warnings=warnings,
        )


def draw_overlay(result: FrameResult, show_pairwise: bool = True) -> np.ndarray:
    out = result.frame.copy()
    if result.marker_found:
        cv2.polylines(out, [result.marker_corners.astype(np.int32)], True, (255, 160, 0), 2)
        tl = result.marker_corners[0].astype(int)
        cv2.putText(out, f"REF id={result.marker_id}", (tl[0], tl[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 2)

    for i, m in enumerate(result.measurements):
        cv2.polylines(out, [m.box_image.astype(np.int32)], True, (0, 255, 0), 2)
        cx, cy = int(m.center_image[0]), int(m.center_image[1])
        label = f"#{i + 1} {m.length_mm:.1f} x {m.width_mm:.1f} mm"
        _put_label(out, label, (cx - 70, cy))

    if show_pairwise and len(result.measurements) >= 2:
        ms = sorted(result.measurements, key=lambda m: m.center_image[0])
        for a, b in zip(ms, ms[1:]):
            d = math.dist(a.center_mm, b.center_mm)
            pa = tuple(int(v) for v in a.center_image)
            pb = tuple(int(v) for v in b.center_image)
            cv2.line(out, pa, pb, (0, 200, 255), 1, cv2.LINE_AA)
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2 - 6)
            _put_label(out, f"{d:.0f} mm", mid, color=(0, 200, 255))

    y = 24
    lines = []
    if result.distance_pnp_mm is not None:
        lines.append(f"Marker distance: {result.distance_pnp_mm:.0f} mm (PnP) | {result.distance_similar_mm:.0f} mm (W*f/P)")
    if result.tilt_deg is not None:
        lines.append(f"Plane tilt: {result.tilt_deg:.0f} deg")
    lines += result.warnings
    for line in lines:
        _put_label(out, line, (10, y), color=(0, 0, 255) if line in result.warnings else (255, 255, 255))
        y += 22
    return out


def _put_label(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
