"""Real-time object dimension & distance measurement.

    python measure.py --source 0 --marker-mm 50 --calib calibration.npz
    python measure.py --source photo.jpg --marker-mm 50 --output annotated.jpg

Keys:  Q quit | W toggle rectified view | E toggle edge view | SPACE pause
       S log the current measurements to --log (and save a snapshot)

For benchmarking (paper Section 5), name the object and its caliper size; each S press then
logs the largest object, and evaluate.py turns the log into the accuracy table:
    python measure.py --source 0 --calib calibration.npz --label "Credit card" --truth 85.60x53.98
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from spatial_measure import SpatialDimensionEstimator, draw_overlay
from spatial_measure import settings as saved_settings

LOG_FIELDS = ["timestamp", "label", "object", "length_mm", "width_mm", "truth_length_mm", "truth_width_mm",
              "distance_pnp_mm", "distance_similar_mm", "truth_distance_mm", "tilt_deg"]


def load_calibration(path):
    if not path:
        print("No --calib given: skipping undistortion and camera-distance estimation.")
        return None, None
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def open_camera(index: int, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    """Opens a webcam on any OS. On Windows DirectShow is used: the default Media Foundation backend can take
    many seconds to open and often ignores the requested resolution."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if sys.platform == "win32" and not cap.isOpened():
        cap = cv2.VideoCapture(index)
    elif sys.platform == "win32":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # raw YUY2 at 1080p is often only 5 fps
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def fit(img, max_w=1280, max_h=720):
    scale = min(1.0, max_w / img.shape[1], max_h / img.shape[0])
    return cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img


def log_rows(path, result, args):
    ms = result.measurements[:1] if args.label else result.measurements
    if not ms and args.truth_distance is None:
        print("Nothing to log.")
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        tl, tw = args.truth or (None, None)
        common = dict(timestamp=datetime.now().isoformat(timespec="seconds"), label=args.label or "",
                      truth_length_mm=tl, truth_width_mm=tw, truth_distance_mm=args.truth_distance,
                      distance_pnp_mm=_r(result.distance_pnp_mm), distance_similar_mm=_r(result.distance_similar_mm),
                      tilt_deg=_r(result.tilt_deg))
        for i, m in enumerate(ms or [None]):
            row = dict(common, object=i + 1)
            if m is not None:
                row.update(length_mm=_r(m.length_mm), width_mm=_r(m.width_mm))
            w.writerow(row)
    print(f"Logged {len(ms) or 1} row(s) -> {path}")


def _r(v):
    return None if v is None else round(float(v), 2)


class LatestFrameReader:
    """Reads the camera on a background thread and keeps only the newest frame.

    Without this, frames queue up inside VideoCapture whenever processing is slower than the
    camera, and the display drifts further and further behind reality (the "lag").
    """

    def __init__(self, cap):
        self.cap, self.frame, self.ok = cap, None, True
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok, self.frame = ok, frame
                self.cond.notify()

    def read(self):
        with self.cond:
            self.cond.wait_for(lambda: self.frame is not None or not self.ok)
            frame, self.frame = self.frame, None
            return frame is not None, frame


def parse_truth(s):
    a, b = (float(x) for x in s.lower().split("x"))
    return max(a, b), min(a, b)


def main():
    saved = saved_settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index, video file or image file")
    ap.add_argument("--marker-mm", type=float, default=float(saved["marker_mm"]),
                    help="printed marker side length, measured with a ruler (default: the size saved in settings.json)")
    ap.add_argument("--marker-id", type=int, help="only use this ArUco id (default: largest marker in view)")
    ap.add_argument("--calib", default=saved["calibration"],
                    help="calibration.npz from calibrate.py (default: the file saved in settings.json, or ./calibration.npz)")
    ap.add_argument("--px-per-mm", type=float, default=4.0, help="resolution of the rectified view")
    ap.add_argument("--min-area-mm2", type=float, default=150.0)
    ap.add_argument("--no-clahe", action="store_true", help="disable LAB/CLAHE shadow suppression")
    ap.add_argument("--no-refine", action="store_true", help="disable Otsu edge refinement")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--output", help="image mode: save annotated result here")
    ap.add_argument("--log", default="measurements.csv")
    ap.add_argument("--label", help="benchmark: name of the object under test")
    ap.add_argument("--truth", type=parse_truth, help="benchmark: caliper size LxW in mm, e.g. 85.60x53.98")
    ap.add_argument("--truth-distance", type=float, help="benchmark: true camera-to-marker distance in mm")
    args = ap.parse_args()

    print(f"Marker side: {args.marker_mm:g} mm")
    K, dist = load_calibration(args.calib)
    est = SpatialDimensionEstimator(
        marker_length_mm=args.marker_mm, camera_matrix=K, dist_coeffs=dist, px_per_mm=args.px_per_mm,
        marker_id=args.marker_id, min_area_mm2=args.min_area_mm2, use_clahe=not args.no_clahe,
        refine_edges=not args.no_refine,
    )

    image = cv2.imread(args.source) if os.path.isfile(args.source) else None
    if image is not None:
        result = est.process(image)
        out = draw_overlay(result)
        for i, m in enumerate(result.measurements, 1):
            print(f"#{i}: {m.length_mm:.1f} x {m.width_mm:.1f} mm")
        for w in result.warnings:
            print("WARNING:", w)
        if args.output:
            cv2.imwrite(args.output, out)
            print(f"Saved -> {args.output}")
        cv2.imshow("measure", fit(out))
        cv2.waitKey(0)
        return

    cap = open_camera(int(args.source), args.width, args.height) if args.source.isdigit() else cv2.VideoCapture(args.source)
    if K is not None:
        print("Make sure the capture resolution matches the one used for calibration.")
    reader = LatestFrameReader(cap) if args.source.isdigit() else cap  # files: process every frame

    show_warped = show_edges = paused = False
    fps, last, result, out = 0.0, time.perf_counter(), None, None
    while True:
        if not paused:
            ok, frame = reader.read()
            if not ok:
                break
            result = est.process(frame)
            out = draw_overlay(result)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(out, f"{fps:.0f} FPS", (out.shape[1] - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("measure", fit(out))
        if show_warped and result.warped is not None:
            cv2.imshow("rectified (top-down)", fit(result.warped, 900, 900))
        if show_edges and result.edges is not None:
            cv2.imshow("edges", fit(result.edges, 900, 900))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("w"):
            show_warped = not show_warped
            if not show_warped:
                cv2.destroyWindow("rectified (top-down)")
        elif key == ord("e"):
            show_edges = not show_edges
            if not show_edges:
                cv2.destroyWindow("edges")
        elif key == ord("s") and result is not None:
            log_rows(args.log, result, args)
            os.makedirs("snapshots", exist_ok=True)
            cv2.imwrite(f"snapshots/{datetime.now():%Y%m%d_%H%M%S}.png", out)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
