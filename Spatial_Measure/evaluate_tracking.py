"""Does the tracker follow *only* the tapped object? Scored on DAVIS 2017 (video object segmentation).

For each sequence the first annotated object is "clicked" in frame 0 (the deepest point of its ground-truth mask),
then followed exactly like the app's hand mode: the seed for the next frame is the inner point of the mask just found.
Every frame's mask is compared with the ground truth (IoU, the DAVIS "J" score).

    python evaluate_tracking.py datasets/davis --backend classical --sequences 20 --frames 30
    python evaluate_tracking.py datasets/davis --backend ai
    python evaluate_tracking.py datasets/davis --tune          # grid-search the classical segmenter's settings

Reported: mean J, "on-object" (share of the predicted mask that lies on the true object: 100 % = it never leaks onto
the background or the hand, whatever part of the object it covers),  % of frames with J >= 0.5 ("tracked") and J >= 0.8 ("clean"), and the frame at which each
sequence is first lost (J < 0.2 for 3 frames in a row).
"""
import argparse
import glob
import itertools
import os
import sys
import time

import cv2
import numpy as np

from PIL import Image

from spatial_measure import SpatialDimensionEstimator


def load_sequence(root, name, n_frames):
    imgs = sorted(glob.glob(os.path.join(root, "JPEGImages", "480p", name, "*.jpg")))[:n_frames]
    anns = sorted(glob.glob(os.path.join(root, "Annotations", "480p", name, "*.png")))[:n_frames]
    frames = [cv2.imread(p) for p in imgs]
    masks = [np.array(Image.open(p)) for p in anns]  # palette PNG: pixel value = object id (cv2 would expand the palette)
    return frames, masks


def inner_point(mask):
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    _, _, _, loc = cv2.minMaxLoc(d)
    return float(loc[0]), float(loc[1])


def run_sequence(est, frames, masks):
    gt0 = (masks[0] == 1).astype(np.uint8)
    if gt0.sum() < 500:
        return None
    seed, scores, precs = inner_point(gt0), [], []
    est.hint = None
    for frame, m in zip(frames, masks):
        gt = m == 1
        cnt, _ = est.segment_image(frame, seed, ppm=1.0, skin=False)
        pred = np.zeros(gt.shape, np.uint8)
        if cnt is not None:
            cv2.drawContours(pred, [cnt], -1, 1, cv2.FILLED)
            seed = inner_point(pred)
        inter, union = np.logical_and(pred > 0, gt).sum(), np.logical_or(pred > 0, gt).sum()
        scores.append(inter / union if union else 1.0)
        precs.append(inter / (pred > 0).sum() if (pred > 0).any() else 0.0)  # share of the mask that is on the object
    return scores, precs


def lost_at(scores):
    for i in range(len(scores) - 2):
        if max(scores[i:i + 3]) < 0.2:
            return i
    return None


def evaluate(root, est, n_seq, n_frames, verbose=True):
    names = sorted(os.listdir(os.path.join(root, "JPEGImages", "480p")))[:n_seq]
    allj, allp, per = [], [], []
    t0 = time.perf_counter()
    for name in names:
        frames, masks = load_sequence(root, name, n_frames)
        out = run_sequence(est, frames, masks)
        if out is None:
            continue
        sc, pr = out
        allj += sc
        allp += pr
        per.append((name, float(np.mean(sc)), lost_at(sc)))
        if verbose:
            print(f"  {name:<18} J={np.mean(sc):.2f}  on-object={np.mean(pr):.2f}  lost at {per[-1][2]}", flush=True)
    j = np.array(allj)
    return {"J": j.mean(), "precision": float(np.mean(allp)), "tracked": (j >= 0.5).mean(), "clean": (j >= 0.8).mean(),
            "lost_seqs": sum(1 for _, _, l in per if l is not None), "seqs": len(per),
            "ms_per_frame": (time.perf_counter() - t0) / max(len(j), 1) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="extracted DAVIS-2017-trainval-480p folder (contains JPEGImages/ and Annotations/)")
    ap.add_argument("--backend", choices=["classical", "ai"], default="classical")
    ap.add_argument("--sequences", type=int, default=20)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--tune", action="store_true", help="grid-search the classical segmenter's settings")
    a = ap.parse_args()

    est = SpatialDimensionEstimator()
    if a.tune:
        best = None
        for tol, canny in itertools.product([(5, 4, 4), (9, 7, 7), (14, 10, 10), (20, 14, 14)],
                                            [(30, 90), (50, 150), (80, 200)]):
            est.fill_tol, est.canny_thresholds = tol, canny
            r = evaluate(a.root, est, a.sequences, a.frames, verbose=False)
            print(f"fill_tol={tol} canny={canny}  J={r['J']:.3f} tracked={r['tracked']:.2f} lost={r['lost_seqs']}/{r['seqs']}", flush=True)
            if best is None or r["J"] > best[0]:
                best = (r["J"], tol, canny)
        print(f"\nBest: fill_tol={best[1]} canny_thresholds={best[2]} (J={best[0]:.3f})")
        return
    if a.backend == "ai":
        from spatial_measure.segmenter import SamSegmenter
        est.segmenter = SamSegmenter()
    r = evaluate(a.root, est, a.sequences, a.frames)
    print(f"\n{a.backend}: J={r['J']:.3f}  on-object={r['precision']:.1%}  tracked(J>=0.5)={r['tracked']:.1%}  clean(J>=0.8)={r['clean']:.1%}  "
          f"sequences lost={r['lost_seqs']}/{r['seqs']}  {r['ms_per_frame']:.0f} ms/frame")


if __name__ == "__main__":
    sys.exit(main())
