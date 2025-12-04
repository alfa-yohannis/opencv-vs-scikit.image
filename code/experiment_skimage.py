#!/usr/bin/env python
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from skimage import transform as sktf
from skimage import color as skcol

# ================== Benchmark knobs (defaults) ==================
DEFAULT_N_WARMUP = 20
DEFAULT_N_RUNS   = 100

# ================== Global config ==================
# You can switch this to np.float64 if you want to test 64-bit everywhere.
DTYPE = np.float32

H, W = 2160, 3840          # 4K input resolution
target_size = (360, 640)   # (height, width) -> 640×360

# Synthetic but reproducible 4K frame
rng = np.random.default_rng(42)
img_rgb_u8 = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

# Normalize once; we will cast to DTYPE later where needed
img_rgb_f = img_rgb_u8.astype(np.float32) / 255.0


# ================== Common YUV (for controlled / fair-mode) ==================
YUV_MATRIX = np.array([
    [ 0.29900,  0.58700,  0.11400],   # Y
    [-0.14713, -0.28886,  0.43600],   # U
    [ 0.61500, -0.51499, -0.10001],   # V
], dtype=np.float64)  # keep matrix in high precision; cast later.


def rgb2yuv_common(img: np.ndarray) -> np.ndarray:
    """
    img: float32/float64 RGB in [0,1], shape (H, W, 3)
    Returns: YUV with same dtype as img.
    """
    img64 = img.astype(np.float64, copy=False)
    yuv64 = img64 @ YUV_MATRIX.T  # (H,W,3) @ (3,3)^T -> (H,W,3)
    return yuv64.astype(img.dtype, copy=False)


# ================== Building blocks: resize only ==================
def skimage_resize(img: np.ndarray, dtype=DTYPE) -> np.ndarray:
    """
    img: float32/float64 RGB in [0,1]
    Returns resized RGB in dtype.
    """
    resized = sktf.resize(
        img,
        (target_size[0], target_size[1], 3),
        order=1,
        anti_aliasing=False,   # closer to INTER_LINEAR-style
        preserve_range=True,
    )
    return resized.astype(dtype, copy=False)


# ================== Building blocks: YUV conversion only ==================
def skimage_yuv(img: np.ndarray, dtype=DTYPE) -> np.ndarray:
    """
    img: RGB in [0,1], dtype float32 or float64
    Uses skimage's internal rgb2yuv.
    Returns cast to dtype for fair comparison.
    """
    yuv = skcol.rgb2yuv(img.astype(np.float32, copy=False))
    return yuv.astype(dtype, copy=False)


# ================== End-to-end pipelines ==================
def skimage_pipeline_real(img: np.ndarray) -> np.ndarray:
    """
    Realistic scikit-image: resize + skimage YUV (internal mechanism).
    """
    rgb_resized = skimage_resize(img, dtype=DTYPE)
    yuv = skimage_yuv(rgb_resized, dtype=DTYPE)
    return yuv


def skimage_pipeline_controlled(img: np.ndarray) -> np.ndarray:
    """
    Controlled: skimage resize + COMMON RGB->YUV.
    """
    rgb_resized = skimage_resize(img, dtype=DTYPE)
    yuv = rgb2yuv_common(rgb_resized)
    return yuv


# ================== Utilities ==================
def time_distribution(
    f,
    *args,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_runs: int = DEFAULT_N_RUNS,
) -> np.ndarray:
    """
    Returns an array of per-run times (seconds) to compute
    mean, std.dev, and distributions.
    Timing does NOT include any file I/O.
    """
    # Warm-up
    for _ in range(n_warmup):
        _ = f(*args)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = f(*args)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def mean_abs_diff(a: np.ndarray, b: np.ndarray):
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    global_mad = float(np.mean(diff))
    per_channel = np.mean(diff, axis=(0, 1))  # (3,)
    return global_mad, per_channel


def log_print(msg: str, fh):
    """Print to console AND write to results.txt."""
    print(msg)
    fh.write(msg + "\n")


if __name__ == "__main__":
    # ---------- CLI args ----------
    parser = argparse.ArgumentParser(
        description="CE-style scikit-image benchmark (4K -> 640x360 + RGB->YUV)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_N_WARMUP,
        help=f"Number of warm-up runs (default: {DEFAULT_N_WARMUP})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Number of timed runs (default: {DEFAULT_N_RUNS})",
    )
    args = parser.parse_args()

    N_WARMUP = args.warmup
    N_RUNS = args.runs

    # ---------- Paths ----------
    results_txt = Path("results_skimage.txt")
    timings_csv = Path("timings_skimage.csv")
    summary_csv = Path("summary_skimage.csv")

    # Init/refresh output files (truncate or create empty)
    for p in (results_txt, timings_csv, summary_csv):
        p.write_text("", encoding="utf-8")

    img_in = img_rgb_f.astype(DTYPE)

    with results_txt.open("w", encoding="utf-8") as f_out:
        log_print(f"Using DTYPE   = {DTYPE.__name__}", f_out)
        log_print(f"Warm-up runs  = {N_WARMUP}", f_out)
        log_print(f"Timed runs    = {N_RUNS}", f_out)

        # ---------- 1) Resize-only timing (ms/frame) ----------
        times_ski_resize = time_distribution(
            skimage_resize, img_in, n_warmup=N_WARMUP, n_runs=N_RUNS
        )
        times_ski_resize_ms = times_ski_resize * 1e3

        mean_ski_r = float(np.mean(times_ski_resize_ms))
        std_ski_r  = float(np.std(times_ski_resize_ms))

        log_print("\n=== Resize-only (4K -> 640x360 RGB, scikit-image) ===", f_out)
        log_print(f"{'Lib':<12} {'mean [ms]':>12} {'std [ms]':>12}", f_out)
        log_print("-" * 40, f_out)
        log_print(f"{'skimage':<12} {mean_ski_r:12.3f} {std_ski_r:12.3f}", f_out)

        # ---------- 2) YUV-only timing ----------
        rgb_common = skimage_resize(img_in)  # reference resized RGB

        times_ski_yuv = time_distribution(
            skimage_yuv, rgb_common, n_warmup=N_WARMUP, n_runs=N_RUNS
        )
        times_ski_yuv_ms = times_ski_yuv * 1e3

        mean_ski_y = float(np.mean(times_ski_yuv_ms))
        std_ski_y  = float(np.std(times_ski_yuv_ms))

        log_print("\n=== YUV-only (RGB 640x360 -> YUV, scikit-image) ===", f_out)
        log_print(f"{'Lib':<12} {'mean [ms]':>12} {'std [ms]':>12}", f_out)
        log_print("-" * 40, f_out)
        log_print(f"{'skimage':<12} {mean_ski_y:12.3f} {std_ski_y:12.3f}", f_out)

        # ---------- 3) End-to-end (Realistic) pipeline ----------
        times_ski_real = time_distribution(
            skimage_pipeline_real, img_in, n_warmup=N_WARMUP, n_runs=N_RUNS
        )
        times_ski_real_ms = times_ski_real * 1e3

        mean_ski_real = float(np.mean(times_ski_real_ms))
        std_ski_real  = float(np.std(times_ski_real_ms))

        log_print("\n=== End-to-end (Realistic) resize + internal YUV (scikit-image) ===", f_out)
        log_print(f"{'Pipeline':<15} {'mean [ms]':>12} {'std [ms]':>12}", f_out)
        log_print("-" * 42, f_out)
        log_print(f"{'skimage real':<15} {mean_ski_real:12.3f} {std_ski_real:12.3f}", f_out)

        # ---------- 4) End-to-end (Controlled) pipeline ----------
        times_ski_ctrl = time_distribution(
            skimage_pipeline_controlled, img_in, n_warmup=N_WARMUP, n_runs=N_RUNS
        )
        times_ski_ctrl_ms = times_ski_ctrl * 1e3

        mean_ski_ctrl = float(np.mean(times_ski_ctrl_ms))
        std_ski_ctrl  = float(np.std(times_ski_ctrl_ms))

        # Overhead: internal YUV vs common YUV
        mean_ski_real_s = float(np.mean(times_ski_real))
        mean_ski_ctrl_s = float(np.mean(times_ski_ctrl))
        overhead_ratio = (
            mean_ski_real_s / mean_ski_ctrl_s if mean_ski_ctrl_s > 0 else float("inf")
        )

        log_print("\n=== End-to-end (Controlled) resize + COMMON YUV (scikit-image) ===", f_out)
        log_print(f"{'Pipeline':<15} {'mean [ms]':>12} {'std [ms]':>12}", f_out)
        log_print("-" * 42, f_out)
        log_print(f"{'skimage ctrl':<15} {mean_ski_ctrl:12.3f} {std_ski_ctrl:12.3f}", f_out)
        log_print(
            f"\nOverhead (realistic / controlled, scikit-only): {overhead_ratio:.2f}× slower",
            f_out,
        )

        # ---------- 5) Numerical differences: real vs controlled (same resize) ----------
        yuv_ski_ctrl = skimage_pipeline_controlled(img_in)
        yuv_ski_real = skimage_pipeline_real(img_in)
        g_mad, ch_mad = mean_abs_diff(yuv_ski_real, yuv_ski_ctrl)

        log_print("\n=== Numerical differences (YUV, scikit: internal vs common) ===", f_out)
        log_print("Difference comes from internal rgb2yuv vs fixed BT.601 matrix:", f_out)
        log_print(f"  Global MAD: {g_mad:.6f}", f_out)
        log_print(
            f"  Y: {ch_mad[0]:.6f}, U: {ch_mad[1]:.6f}, V: {ch_mad[2]:.6f}",
            f_out,
        )

        log_print(
            "\nOutput shapes (real vs controlled pipelines): "
            f"{yuv_ski_real.shape} {yuv_ski_ctrl.shape}",
            f_out,
        )

    # ---------- 6) Export CSVs (after all timing!) ----------
    # a) Per-run timings (for plots, including boxplots)
    with timings_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["category", "pipeline", "run_index", "time_ms"])

        for i, t in enumerate(times_ski_resize_ms):
            writer.writerow(["resize", "skimage", i, f"{t:.6f}"])

        for i, t in enumerate(times_ski_yuv_ms):
            writer.writerow(["yuv", "skimage", i, f"{t:.6f}"])

        for i, t in enumerate(times_ski_real_ms):
            writer.writerow(["real", "skimage", i, f"{t:.6f}"])

        for i, t in enumerate(times_ski_ctrl_ms):
            writer.writerow(["ctrl", "skimage", i, f"{t:.6f}"])

    # b) Summary stats (means, std, MADs)
    with summary_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "category", "pipeline", "value"])

        # Resize
        writer.writerow(["mean_ms", "resize", "skimage", f"{mean_ski_r:.6f}"])
        writer.writerow(["std_ms",  "resize", "skimage", f"{std_ski_r:.6f}"])

        # YUV
        writer.writerow(["mean_ms", "yuv", "skimage", f"{mean_ski_y:.6f}"])
        writer.writerow(["std_ms",  "yuv", "skimage", f"{std_ski_y:.6f}"])

        # Realistic
        writer.writerow(["mean_ms", "real", "skimage", f"{mean_ski_real:.6f}"])
        writer.writerow(["std_ms",  "real", "skimage", f"{std_ski_real:.6f}"])

        # Controlled
        writer.writerow(["mean_ms", "ctrl", "skimage", f"{mean_ski_ctrl:.6f}"])
        writer.writerow(["std_ms",  "ctrl", "skimage", f"{std_ski_ctrl:.6f}"])
        writer.writerow(
            ["overhead_real_over_ctrl", "ctrl_vs_real", "skimage", f"{overhead_ratio:.6f}"]
        )

        # MADs (real vs controlled)
        writer.writerow(
            ["mad_global", "real_vs_ctrl", "skimage", f"{g_mad:.6f}"]
        )
        writer.writerow(
            ["mad_Y", "real_vs_ctrl", "skimage", f"{ch_mad[0]:.6f}"]
        )
        writer.writerow(
            ["mad_U", "real_vs_ctrl", "skimage", f"{ch_mad[1]:.6f}"]
        )
        writer.writerow(
            ["mad_V", "real_vs_ctrl", "skimage", f"{ch_mad[2]:.6f}"]
        )
