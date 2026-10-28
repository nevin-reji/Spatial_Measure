"""Measurement engine for the native macOS app (macapp/) and the cross-platform app (app.py).

app.py runs `Engine` in-process on a worker thread. The macOS app owns the camera and the on-screen preview; it streams frames to this process over
stdin and draws whatever comes back. Because the app only sends a new frame once the previous
result has arrived, the preview never waits on the measurement pipeline.

stdin  (little-endian):
    b"F" u32 width, u32 height, u32 bytes_per_row, BGRA pixels  -> one "result" line
    b"C" u32 length, JSON settings                                -> one "config" line
    b"L" u32 length, JSON {label, truth, truth_distance}          -> one "logged" line
    b"T" u32 length, JSON {x, y} (tap, raw frame pixels) or {clear: true}   -> no reply; the next
         result then carries only the tapped object, outlined and measured
stdout: one JSON object per line. Nothing else may be printed to stdout.
"""
import json
import math
import os
import struct
import sys
import time
import traceback
import types
from datetime import datetime

import cv2
import numpy as np

from measure import log_rows, parse_truth
from spatial_measure import SpatialDimensionEstimator, draw_overlay
from spatial_measure.estimator import FrameResult
from spatial_measure.scan import PlaneTracker, ScanMap

OUT = sys.stdout


def send(msg: dict):
    OUT.write(json.dumps(msg, separators=(",", ":")) + "\n")
    OUT.flush()


def read_exact(stream, n: int) -> memoryview:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = stream.readinto(view[got:])
        if not k:
            raise EOFError
        got += k
    return view


class SizeSmoother:
    """Steadies the displayed numbers: an object keeps its id across frames (matched by its
    position on the marker plane) and its size is an exponential moving average."""

    def __init__(self, alpha=0.25, match_mm=15.0, max_misses=8):
        self.alpha, self.match_mm, self.max_misses = alpha, match_mm, max_misses
        self.tracks, self.next_id = [], 1

    def update(self, measurements):
        out, used = [], set()
        for m in measurements:
            best, best_d = None, self.match_mm
            for t in self.tracks:
                d = math.dist(t["center"], m.center_mm)
                same_size = abs(t["length"] - m.length_mm) < 0.15 * t["length"]
                if id(t) not in used and d < best_d and same_size:
                    best, best_d = t, d
            if best is None:
                best = {"id": self.next_id, "length": m.length_mm, "width": m.width_mm}
                self.next_id += 1
                self.tracks.append(best)
            else:
                a = self.alpha
                best["length"] += a * (m.length_mm - best["length"])
                best["width"] += a * (m.width_mm - best["width"])
            best["center"], best["misses"] = m.center_mm, 0
            used.add(id(best))
            out.append((best["id"], best["length"], best["width"]))
        for t in self.tracks:
            if id(t) not in used:
                t["misses"] = t.get("misses", 0) + 1
        self.tracks = [t for t in self.tracks if t["misses"] <= self.max_misses]
        return out

    def reset(self):
        self.tracks = []


class Engine:
    def __init__(self, emit=send):
        self.emit = emit  # receives every reply (a dict); stdout for the macOS app, a queue for app.py
        self.settings = {"marker_mm": 50.0, "calibration": None, "use_clahe": True, "refine": True, "smoothing": True, "scan": False, "mode": "table", "distance_mm": 400.0, "hfov_deg": 65.0, "ai": False}
        self.smoother = SizeSmoother()
        self.last = None
        self.seed_px = None  # fresh tap, raw frame pixels
        self.seed_mm = None  # selected object, tracked as a point on the marker plane
        self.sel_size = None  # smoothed (length, width, area) of the selected object
        self.sel_misses = 0
        self.following, self.follow_px, self.outliers = False, None, 0
        self.build()

    def build(self):
        K = dist = None
        calib = self.settings.get("calibration")
        if calib:
            data = np.load(calib)
            K, dist = data["camera_matrix"], data["dist_coeffs"]
        self.sam = getattr(self, "sam", None)
        if self.settings.get("ai") and self.sam is None:
            from spatial_measure.segmenter import SamSegmenter  # loads MobileSAM (downloads the weights once)
            self.sam = SamSegmenter()
        self.est = SpatialDimensionEstimator(
            marker_length_mm=float(self.settings["marker_mm"]), camera_matrix=K, dist_coeffs=dist,
            use_clahe=bool(self.settings["use_clahe"]), refine_edges=bool(self.settings["refine"]),
        )
        self.est.segmenter = self.sam if self.settings.get("ai") else None
        self.smoother.reset()
        self.tracker = PlaneTracker(float(self.settings["marker_mm"]))
        self.scanmap = ScanMap()
        self.scan_size, self.scan_frames, self.scan_map_b64 = None, 0, ""
        self.closed_run = 0

    def configure(self, cfg: dict):
        self.settings.update(cfg)
        try:
            self.build()
            self.emit({"type": "config", "ok": True, "calibrated": self.est.K is not None})
        except Exception as e:
            traceback.print_exc()
            failed_ai = bool(self.settings.get("ai")) and self.sam is None
            self.settings["calibration"] = None if not failed_ai else self.settings["calibration"]
            self.settings["ai"] = False
            self.build()
            what = "Smart outline unavailable" if failed_ai else "Could not load calibration"
            self.emit({"type": "config", "ok": False, "message": f"{what}: {e}", "ai_failed": failed_ai})

    def _open(self, is_open: bool) -> bool:
        """'Fully scanned' only after 6 consecutive frames agree, so one odd frame cannot flip the label."""
        self.closed_run = 0 if is_open else self.closed_run + 1
        return self.closed_run < 6

    def tap(self, req: dict):
        self.seed_mm, self.sel_size, self.sel_misses = None, None, 0
        self.following, self.follow_px, self.outliers = False, None, 0
        self.tracker.reset()
        self.scanmap.reset()
        self.closed_run = 0
        self.est.hint = None
        self.scan_size, self.scan_frames, self.scan_map_b64 = None, 0, ""
        self.seed_px = None if req.get("clear") else (float(req["x"]), float(req["y"]))

    def frame(self, w: int, h: int, bpr: int, data: memoryview):
        bgra = np.frombuffer(data, np.uint8).reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
        self.process(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))

    def process(self, bgr: np.ndarray):
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        fresh = self.seed_px
        nomarker = self.settings.get("mode") == "nomarker"  # scale from an assumed distance, no marker
        hand = nomarker or self.settings.get("mode") == "hand"  # object held in the hand: follow it in the image
        following = hand and self.following
        seed_arg = fresh if fresh is not None else (self.follow_px if following else None)
        scan = (not hand) and bool(self.settings.get("scan")) and (fresh is not None or self.seed_mm is not None)
        if nomarker and seed_arg is not None:
            r = self.est.process_nomarker(bgr, seed_arg, float(self.settings["distance_mm"]), float(self.settings["hfov_deg"]))
        elif nomarker:
            r = FrameResult(frame=bgr, marker_found=False, warnings=[])
        else:
            r = self.est.process(
                bgr, seed_px=seed_arg, seed_mm=None if hand else self.seed_mm,
                tracker=self.tracker if scan else None,
                extra_seeds_mm=self.scanmap.seed_candidates() if scan and self.seed_mm is not None else None,
                validate=self.scanmap.consistent if scan else None, follow=hand, skin=hand)
        self.last = r
        selecting = fresh is not None or following or (not hand and self.seed_mm is not None)
        state = r.selection_state
        if fresh is not None:
            self.seed_px = None
            if not r.marker_found and not nomarker:
                state = "no_marker"
        if selecting:
            if r.selected is not None:
                self.sel_misses = 0
                if hand:
                    self.following, self.follow_px = True, r.seed_px
                else:
                    self.seed_mm = r.seed_mm
            elif (state in ("nothing", "no_marker", "marker") and (fresh is not None or not hand)) \
                    or self.sel_misses > (400 if scan else 40):
                self.seed_mm, self.sel_size, self.following = None, None, False  # give up: back to showing every object
                selecting = False
            else:
                if r.seed_mm is not None and not hand:
                    self.seed_mm = r.seed_mm
                self.sel_misses += 1
                state = "lost" if state in (None, "nothing", "marker") else state

        raw = self.est.to_raw_pixels
        if self.settings["smoothing"]:
            sizes = self.smoother.update(r.measurements)
        else:
            sizes = [(i + 1, m.length_mm, m.width_mm) for i, m in enumerate(r.measurements)]
        objects = [
            {"id": oid, "length_mm": round(float(L), 2), "width_mm": round(float(W), 2),
             "box": raw(m.box_image).round(1).tolist(), "center": raw(np.array(m.center_image)).round(1)[0].tolist()}
            for m, (oid, L, W) in zip(r.measurements, sizes)
        ]
        if r.selected is not None:
            m, a = r.selected, 0.25
            cur = (m.length_mm, m.width_mm, m.area_mm2 or 0.0)
            if hand and self.sel_size is not None and self.outliers < 4 and \
                    abs(cur[0] - self.sel_size[0]) > 0.35 * self.sel_size[0]:
                self.outliers += 1  # one odd frame (a finger over the edge, motion blur): keep the last good size
            else:
                self.outliers = 0
                self.sel_size = cur if self.sel_size is None or not self.settings["smoothing"] else tuple(
                    o + a * (n - o) for o, n in zip(self.sel_size, cur))
            L, W, A = self.sel_size
            objects = [{"id": 1, "length_mm": round(float(L), 2), "width_mm": round(float(W), 2),
                        "area_mm2": round(float(A), 1), "box": raw(m.box_image).round(1).tolist(),
                        "center": raw(np.array(m.center_image)).round(1)[0].tolist(),
                        "outline": raw(m.outline_image).round(1).tolist()}]
        elif selecting:
            objects = []
        scan_msg = None
        if scan and selecting:
            if r.selected is not None and r.scan_data is not None:
                self.scanmap.add(r.scan_data["cnt"], r.scan_data["valid"], r.scan_data["ppm"], r.scan_data["origin"])
            sm = self.scanmap.measure(self.seed_mm)
            if sm is not None and r.plane_map is not None:
                H, ppm, origin = r.plane_map
                Hinv = np.linalg.inv(H)

                def to_raw(mm):
                    c = (np.asarray(mm, np.float32).reshape(-1, 2) * ppm + origin).astype(np.float32)
                    img = cv2.perspectiveTransform(c.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
                    img = np.nan_to_num(np.clip(img, -1e5, 1e5))
                    return raw(img).round(1).tolist()

                cur = (sm["length_mm"], sm["width_mm"], sm["area_mm2"])
                self.scan_size = cur if self.scan_size is None or not self.settings["smoothing"] else tuple(
                    o + 0.3 * (n - o) for o, n in zip(self.scan_size, cur))
                L, W, A = self.scan_size
                objects = [{"id": 1, "length_mm": round(float(L), 1), "width_mm": round(float(W), 1),
                            "area_mm2": round(float(A), 0), "box": to_raw(sm["box_mm"]),
                            "center": to_raw(sm["center_mm"])[0], "outline": to_raw(sm["outline_mm"]),
                            "open": self._open(sm["open"])}]
                self.scan_frames += 1
                if self.scan_frames % 4 == 1:
                    self.scan_map_b64 = self.scanmap.preview_png_b64(sm)
            scan_msg = {"source": r.plane_source, "frames_since_marker": self.tracker.frames_since_marker,
                        "map": self.scan_map_b64}
        pairs = []
        ordered = [] if selecting else sorted(zip(r.measurements, objects), key=lambda p: p[0].center_image[0])
        for (ma, oa), (mb, ob) in zip(ordered, ordered[1:]):
            pairs.append({"a": oa["center"], "b": ob["center"], "mm": round(math.dist(ma.center_mm, mb.center_mm), 1)})

        self.emit({
            "type": "result",
            "frame_size": [w, h],
            "marker_found": r.marker_found,
            "marker_id": r.marker_id,
            "marker": raw(r.marker_corners).round(1).tolist() if r.marker_found else None,
            "plane_source": r.plane_source,
            "objects": objects,
            "selecting": selecting,
            "scan": scan_msg,
            "selection_state": state if selecting or state else None,
            "pairs": pairs,
            "distance_pnp_mm": _r(r.distance_pnp_mm),
            "distance_similar_mm": _r(r.distance_similar_mm),
            "tilt_deg": _r(r.tilt_deg),
            "warnings": r.warnings,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    def log(self, req: dict):
        if self.last is None:
            self.emit({"type": "logged", "ok": False, "message": "No frame processed yet"})
            return
        truth = req.get("truth") or None
        args = types.SimpleNamespace(
            label=req.get("label") or None,
            truth=parse_truth(truth) if truth else None,
            truth_distance=float(req["truth_distance"]) if req.get("truth_distance") else None,
        )
        log_rows("measurements.csv", self.last, args)
        os.makedirs("snapshots", exist_ok=True)
        snap = f"snapshots/{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(snap, draw_overlay(self.last))
        n = 1 if args.label else len(self.last.measurements)
        self.emit({"type": "logged", "ok": True, "message": f"Logged {n} object(s) to measurements.csv"})


def _r(v):
    return None if v is None else round(float(v), 1)


def main():
    global OUT
    OUT, sys.stdout = sys.stdout, sys.stderr  # keep stray prints (e.g. from log_rows) off the protocol channel
    stdin = sys.stdin.buffer
    engine = Engine()
    send({"type": "ready", "opencv": cv2.__version__})
    while True:
        try:
            kind = read_exact(stdin, 1).tobytes()
            if kind == b"F":
                w, h, bpr = struct.unpack("<III", read_exact(stdin, 12))
                data = read_exact(stdin, h * bpr)
                try:
                    engine.frame(w, h, bpr, data)
                except Exception as e:
                    traceback.print_exc()
                    send({"type": "result", "frame_size": [w, h], "marker_found": False, "error": str(e)})
            elif kind in (b"C", b"L", b"T"):
                (n,) = struct.unpack("<I", read_exact(stdin, 4))
                payload = json.loads(read_exact(stdin, n).tobytes())
                try:
                    {b"C": engine.configure, b"L": engine.log, b"T": engine.tap}[kind](payload)
                except Exception as e:
                    traceback.print_exc()
                    if kind != b"T":
                        send({"type": "logged" if kind == b"L" else "config", "ok": False, "message": str(e)})
            else:
                send({"type": "error", "message": f"unknown message {kind!r}"})
                return
        except EOFError:
            return


if __name__ == "__main__":
    main()
