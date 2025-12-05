#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TIMINGS_CSV = "timings.csv"
OUTPUT_PDF = "latency_boxplot.pdf"

if __name__ == "__main__":
    times_opencv_real = []
    times_skimage_real = []

    # Read per-run timings from CSV
    with open(TIMINGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] != "real":
                continue

            t = float(row["time_ms"])

            if row["pipeline"] == "opencv":
                times_opencv_real.append(t)
            elif row["pipeline"] == "skimage":
                times_skimage_real.append(t)

    times_opencv_real = np.array(times_opencv_real, dtype=np.float64)
    times_skimage_real = np.array(times_skimage_real, dtype=np.float64)

    if len(times_opencv_real) == 0 or len(times_skimage_real) == 0:
        raise RuntimeError("Missing OpenCV or scikit-image real timing data in timings.csv")

    # ---------- Compute means ----------
    mean_opencv = np.mean(times_opencv_real)
    mean_skimage = np.mean(times_skimage_real)

    # ---------- Boxplot ----------
    plt.figure(figsize=(6.5, 3.0))

    positions = [1, 2]
    plt.boxplot(
        [times_opencv_real, times_skimage_real],
        widths=0.6,
        positions=positions,
        showmeans=True,
        meanprops=dict(
            marker="x",
            markersize=6,
            markeredgecolor="black",
            markerfacecolor="black"
        )
    )

    plt.xticks(positions, ["OpenCV real", "scikit-image real"])
    plt.ylabel("Latency (ms per frame)")

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(50))

    # ✅ Horizontal grid only (no vertical grid)
    plt.grid(axis="y")

    # ✅ Remove inner border (all spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # ---------- Mean value labels ----------
    plt.text(
        1.35, mean_opencv,
        f"avg = {mean_opencv:.1f} ms",
        ha="left",
        va="center",
        fontsize=9
    )

    plt.text(
        1.65, mean_skimage,
        f"avg = {mean_skimage:.1f} ms",
        ha="right",
        va="center",
        fontsize=9
    )

    # ---------- Save ----------
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()

    print(f"Saved boxplot to {OUTPUT_PDF}")
