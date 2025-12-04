#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt

# Input CSVs from separated experiments
TIMINGS_OPENCV = "timings_opencv.csv"
TIMINGS_SKIMAGE = "timings_skimage.csv"

# Output figure
OUTPUT_PDF = "latency_boxplot_real.pdf"


def load_real_latencies(csv_path, expected_pipeline):
    """
    Load only:
      - category == "real"
      - pipeline == expected_pipeline
    Returns np.array of time_ms
    """
    times = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] != "real":
                continue
            if row["pipeline"] != expected_pipeline:
                continue
            times.append(float(row["time_ms"]))
    return np.array(times, dtype=np.float64)


if __name__ == "__main__":
    # Load data
    times_opencv_real = load_real_latencies(
        TIMINGS_OPENCV, expected_pipeline="opencv"
    )
    times_skimage_real = load_real_latencies(
        TIMINGS_SKIMAGE, expected_pipeline="skimage"
    )

    # Safety checks
    if len(times_opencv_real) == 0:
        raise RuntimeError("No OpenCV real timings found in timings_opencv.csv")

    if len(times_skimage_real) == 0:
        raise RuntimeError("No scikit-image real timings found in timings_skimage.csv")

    print(f"Loaded {len(times_opencv_real)} OpenCV real samples")
    print(f"Loaded {len(times_skimage_real)} scikit-image real samples")

    # ---- Boxplot (single figure, no custom colors) ----
    plt.figure()
    plt.boxplot([times_opencv_real, times_skimage_real])
    plt.xticks([1, 2], ["OpenCV (real)", "scikit-image (real)"])
    plt.ylabel("Latency (ms per frame)")
    plt.title("Latency Distribution: 4K → 640×360 + RGB→YUV (Real pipelines)")
    plt.grid(True)

    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()

    print(f"Saved boxplot to {OUTPUT_PDF}")
