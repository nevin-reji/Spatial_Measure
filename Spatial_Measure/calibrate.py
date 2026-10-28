"""Camera calibration with a checkerboard (paper Section 4.1, Zhang's method).

From saved photos:
    python calibrate.py --images "calib/*.jpg"
Live from a webcam (SPACE = capture view, C = calibrate & save, Q = quit):
    python calibrate.py --camera 0

Capture 15-30 views with the board tilted (pitch/roll/yaw), at different distances,
and covering the corners of the frame -- corner coverage is what pins down distortion.
"""
import argparse
import glob

import cv2
import numpy as np

from measure import open_camera

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def find_corners(gray, pattern):
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return ok, corners


def calibrate(image_points, image_size, pattern, square_mm, out):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_mm
    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(image_points), image_points, image_size, None, None)
    dist = dist.ravel()
    np.savez(out, camera_matrix=K, dist_coeffs=dist, image_size=np.array(image_size), rms=rms)
    print(f"\nViews used: {len(image_points)}   image size: {image_size[0]}x{image_size[1]}")
    print(f"RMS reprojection error: {rms:.3f} px   (aim for < 0.5 px)")
    print(f"fx = {K[0, 0]:.2f} px, fy = {K[1, 1]:.2f} px")
    print(f"cx = {K[0, 2]:.2f} px, cy = {K[1, 2]:.2f} px")
    print("Distortion: k1={:.4f}, k2={:.4f}, p1={:.4f}, p2={:.4f}, k3={:.4f}".format(*dist[:5]))
    print(f"Saved -> {out}")


def from_images(pattern_glob, pattern):
    points, size = [], None
    for path in sorted(glob.glob(pattern_glob)):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = gray.shape[::-1]
        ok, corners = find_corners(gray, pattern)
        print(f"{'ok  ' if ok else 'MISS'} {path}")
        if ok:
            points.append(corners)
    return points, size


def from_camera(index, pattern, width, height):
    cap = open_camera(index, width, height)
    points, size = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"captured: {len(points)}  SPACE=capture  C=calibrate  Q=quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            ok2, refined = find_corners(gray, pattern)
            if ok2:
                points.append(refined)
                cv2.imwrite(f"calib_{len(points):02d}.png", frame)
        elif key == ord("c") and len(points) >= 10:
            break
        elif key == ord("q"):
            points = []
            break
    cap.release()
    cv2.destroyAllWindows()
    return points, size


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help='glob, e.g. "calib/*.jpg"')
    src.add_argument("--camera", type=int, help="webcam index")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", default="calibration.npz")
    a = ap.parse_args()

    pattern = (a.cols, a.rows)
    if a.images:
        pts, size = from_images(a.images, pattern)
    else:
        pts, size = from_camera(a.camera, pattern, a.width, a.height)
    if len(pts) < 10:
        raise SystemExit(f"Need at least 10 good views, got {len(pts)}.")
    calibrate(pts, size, pattern, a.square_mm, a.out)
