"""Turns a measure.py log into the paper's accuracy tables (Sections 5.1 and 5.2).

    python evaluate.py measurements.csv            # prints Markdown tables
"""
import argparse
import csv
import sys
from collections import defaultdict
from statistics import mean, pstdev


def rows_with(rows, *cols):
    return [r for r in rows if all(r.get(c) not in (None, "") for c in cols)]


def dimension_table(rows):
    groups = defaultdict(list)
    for r in rows_with(rows, "label", "truth_length_mm", "length_mm"):
        groups[(r["label"], "Length", float(r["truth_length_mm"]))].append(float(r["length_mm"]))
        if r.get("truth_width_mm") and r.get("width_mm"):
            groups[(r["label"], "Width", float(r["truth_width_mm"]))].append(float(r["width_mm"]))
    if not groups:
        return None
    lines = ["| Test Object | Ground Truth (mm) | Estimated Mean (mm) | Absolute Error (mm) | Error (%) | Std Dev (mm) | n |",
             "|---|---|---|---|---|---|---|"]
    abs_errors = []
    for (label, side, truth), vals in groups.items():
        m = mean(vals)
        abs_errors.append(abs(m - truth))
        lines.append(f"| {label} ({side}) | {truth:.2f} | {m:.2f} | {abs(m - truth):.2f} | {(m - truth) / truth * 100:+.2f}% "
                     f"| {pstdev(vals):.2f} | {len(vals)} |")
    lines.append(f"\nMean absolute error: {mean(abs_errors):.2f} mm, worst: {max(abs_errors):.2f} mm")
    return "\n".join(lines)


def distance_table(rows):
    groups = defaultdict(lambda: ([], []))
    for r in rows_with(rows, "truth_distance_mm", "distance_pnp_mm"):
        pnp, sim = groups[float(r["truth_distance_mm"])]
        pnp.append(float(r["distance_pnp_mm"]))
        if r.get("distance_similar_mm"):
            sim.append(float(r["distance_similar_mm"]))
    if not groups:
        return None
    lines = ["| Nominal Distance (mm) | PnP Mean (mm) | PnP Error (mm) | PnP Variance (σ²) | W·f/P Mean (mm) | W·f/P Error (mm) | n |",
             "|---|---|---|---|---|---|---|"]
    for truth in sorted(groups):
        pnp, sim = groups[truth]
        sim_cols = f"{mean(sim):.1f} | {abs(mean(sim) - truth):.1f}" if sim else "– | –"
        lines.append(f"| {truth:.1f} | {mean(pnp):.1f} | {abs(mean(pnp) - truth):.1f} | {pstdev(pnp) ** 2:.2f} | {sim_cols} | {len(pnp)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="measurements.csv")
    sys.stdout.reconfigure(encoding="utf-8")  # the tables use σ² and ·, which a redirected Windows console can't encode
    rows = list(csv.DictReader(open(ap.parse_args().log, newline="", encoding="utf-8")))
    dims, dist = dimension_table(rows), distance_table(rows)
    if dims:
        print("## Dimensional accuracy\n\n" + dims + "\n")
    if dist:
        print("## Distance estimation accuracy\n\n" + dist)
    if not dims and not dist:
        print("No benchmark rows found. Log with --label/--truth or --truth-distance in measure.py.")
