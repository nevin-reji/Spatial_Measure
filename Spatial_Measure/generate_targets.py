"""Creates print-ready targets: an ArUco reference marker and a calibration checkerboard.

    python generate_targets.py --marker-mm 50 --marker-id 0 --cols 9 --rows 6 --square-mm 25

PDFs are written at true physical size. Print at 100% / "actual size" (never "fit to page")
and check the printed size with a ruler; pass the *measured* size to measure.py.
"""
import argparse

import cv2
import numpy as np
from PIL import Image

DPI = 300


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def save_pdf(img: np.ndarray, path: str):
    Image.fromarray(img).save(path, resolution=DPI)
    cv2.imwrite(path.replace(".pdf", ".png"), img)
    print(f"wrote {path} (+ .png)")


def make_marker(marker_mm: float, marker_id: int) -> np.ndarray:
    side = mm_to_px(marker_mm)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, side)
    margin = mm_to_px(15)  # white quiet zone the detector needs
    page = np.full((side + 2 * margin, side + 2 * margin), 255, np.uint8)
    page[margin : margin + side, margin : margin + side] = marker
    cv2.putText(page, f"DICT_4X4_50 id={marker_id}  side={marker_mm:g} mm", (margin, page.shape[0] - margin // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    return page


def make_checkerboard(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """cols x rows are INNER corners (OpenCV convention), so the board has (cols+1) x (rows+1) squares."""
    sq = mm_to_px(square_mm)
    board = np.full(((rows + 1) * sq, (cols + 1) * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 0
    margin = mm_to_px(10)
    return cv2.copyMakeBorder(board, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-mm", type=float, default=50.0)
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out-dir", default="targets")
    a = ap.parse_args()

    import os

    os.makedirs(a.out_dir, exist_ok=True)
    save_pdf(make_marker(a.marker_mm, a.marker_id), f"{a.out_dir}/aruco_id{a.marker_id}_{a.marker_mm:g}mm.pdf")
    save_pdf(make_checkerboard(a.cols, a.rows, a.square_mm), f"{a.out_dir}/checkerboard_{a.cols}x{a.rows}_{a.square_mm:g}mm.pdf")
