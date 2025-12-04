import time
import numpy as np
import cv2
from skimage import transform as sktf
from skimage import color as skcol
import matplotlib.pyplot as plt  # for boxplot

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
def opencv_resize(img: np.ndarray, dtype=DTYPE) -> np.ndarray:
    """
    img: float32/float64 RGB in [0,1]
    Returns resized RGB in dtype.
    """
    resized = cv2.resize(
        img,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(dtype, copy=False)


def skimage_resize(img: np.ndarray, dtype=DTYPE) -> np.ndarray:
    """
    img: float32/float64 RGB in [0,1]
    Returns resized RGB in dtype.
    """
    resized = sktf.resize(
        img,
        (target_size[0], target_size[1], 3),
        order=1,
        anti_aliasing=False,   # match INTER_LINEAR style
        preserve_range=True,
    )
    return resized.astype(dtype, copy=False)


# ================== Building blocks: YUV conversion only ==================
def opencv_yuv(img: np.ndarray) -> np.ndarray:
    """
    img: RGB in [0,1], dtype float32 or float64
    Uses OpenCV's cvtColor.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def skimage_yuv(img: np.ndarray, dtype=DTYPE) -> np.ndarray:
    """
    img: RGB in [0,1], dtype float32 or float64
    Uses skimage's internal rgb2yuv.
    Returns cast to dtype for fair comparison.
    """
    yuv64 = skcol.rgb2yuv(img.astype(np.float32, copy=False))
    return yuv64.astype(dtype, copy=False)


# ================== End-to-end pipelines ==================
def opencv_pipeline_real(img: np.ndarray) -> np.ndarray:
    """
    Realistic OpenCV: resize + OpenCV YUV.
    """
    rgb_resized = opencv_resize(img, dtype=DTYPE)
    yuv = opencv_yuv(rgb_resized)
    return yuv.astype(DTYPE, copy=False)


def skimage_pipeline_real(img: np.ndarray) -> np.ndarray:
    """
    Realistic scikit-image: resize + skimage YUV (internal mechanism).
    """
    rgb_resized = skimage_resize(img, dtype=DTYPE)
    yuv = skimage_yuv(rgb_resized, dtype=DTYPE)
    return yuv


def opencv_pipeline_controlled(img: np.ndarray) -> np.ndarray:
    """
    Controlled: OpenCV resize + COMMON RGB->YUV.
    """
    rgb_resized = opencv_resize(img, dtype=DTYPE)
    yuv = rgb2yuv_common(rgb_resized)
    return yuv


def skimage_pipeline_controlled(img: np.ndarray) -> np.ndarray:
    """
    Controlled: skimage resize + COMMON RGB->YUV.
    """
    rgb_resized = skimage_resize(img, dtype=DTYPE)
    yuv = rgb2yuv_common(rgb_resized)
    return yuv


# ================== Utilities ==================
def time_function(f, *args, n_warmup: int = 10, n_runs: int = 50) -> float:
    """
    Returns average seconds per run.
    """
    for _ in range(n_warmup):
        _ = f(*args)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = f(*args)
    end = time.perf_counter()

    return (end - start) / n_runs


def time_distribution(f, *args, n_warmup: int = 10, n_runs: int = 50) -> np.ndarray:
    """
    Returns an array of per-run times (seconds) to compute
    mean, std.dev, and boxplots.
    """
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


if __name__ == "__main__":
    print(f"Using DTYPE = {DTYPE.__name__}")
    img_in = img_rgb_f.astype(DTYPE)

    N_RUNS = 50  # number of samples for all distributions

    # ---------- 1) Resize-only timing (mean + std, ms/frame) ----------
    times_cv_resize = time_distribution(opencv_resize, img_in, n_runs=N_RUNS)
    times_ski_resize = time_distribution(skimage_resize, img_in, n_runs=N_RUNS)

    times_cv_resize_ms = times_cv_resize * 1e3
    times_ski_resize_ms = times_ski_resize * 1e3

    mean_cv_r = np.mean(times_cv_resize_ms)
    std_cv_r  = np.std(times_cv_resize_ms)

    mean_ski_r = np.mean(times_ski_resize_ms)
    std_ski_r  = np.std(times_ski_resize_ms)

    print("\n=== Resize-only (4K -> 640x360 RGB) ===")
    print(f"{'Lib':<12} {'mean [ms]':>12} {'std [ms]':>12}")
    print("-" * 40)
    print(f"{'OpenCV':<12} {mean_cv_r:12.3f} {std_cv_r:12.3f}")
    print(f"{'skimage':<12} {mean_ski_r:12.3f} {std_ski_r:12.3f}")

    # ---------- 2) YUV-only timing (same resized input) ----------
    rgb_common = opencv_resize(img_in)  # reference resized RGB

    times_cv_yuv = time_distribution(opencv_yuv, rgb_common, n_runs=N_RUNS)
    times_ski_yuv = time_distribution(skimage_yuv, rgb_common, n_runs=N_RUNS)

    times_cv_yuv_ms = times_cv_yuv * 1e3
    times_ski_yuv_ms = times_ski_yuv * 1e3

    mean_cv_y = np.mean(times_cv_yuv_ms)
    std_cv_y  = np.std(times_cv_yuv_ms)

    mean_ski_y = np.mean(times_ski_yuv_ms)
    std_ski_y  = np.std(times_ski_yuv_ms)

    print("\n=== YUV-only (RGB 640x360 -> YUV) ===")
    print(f"{'Lib':<12} {'mean [ms]':>12} {'std [ms]':>12}")
    print("-" * 40)
    print(f"{'OpenCV':<12} {mean_cv_y:12.3f} {std_cv_y:12.3f}")
    print(f"{'skimage':<12} {mean_ski_y:12.3f} {std_ski_y:12.3f}")

    # ---------- 3) End-to-end (Realistic) pipelines ----------
    times_cv_real = time_distribution(opencv_pipeline_real, img_in, n_runs=N_RUNS)
    times_ski_real = time_distribution(skimage_pipeline_real, img_in, n_runs=N_RUNS)

    times_cv_real_ms = times_cv_real * 1e3
    times_ski_real_ms = times_ski_real * 1e3

    mean_cv_real = np.mean(times_cv_real_ms)
    std_cv_real  = np.std(times_cv_real_ms)

    mean_ski_real = np.mean(times_ski_real_ms)
    std_ski_real  = np.std(times_ski_real_ms)

    # speedup based on mean seconds
    mean_cv_real_s = np.mean(times_cv_real)
    mean_ski_real_s = np.mean(times_ski_real)
    speedup_real = mean_ski_real_s / mean_cv_real_s if mean_cv_real_s > 0 else float("inf")

    print("\n=== End-to-end (Realistic) resize + internal YUV ===")
    print(f"{'Pipeline':<15} {'mean [ms]':>12} {'std [ms]':>12}")
    print("-" * 42)
    print(f"{'OpenCV real':<15} {mean_cv_real:12.3f} {std_cv_real:12.3f}")
    print(f"{'skimage real':<15} {mean_ski_real:12.3f} {std_ski_real:12.3f}")
    print(f"\nSpeedup (skimage / OpenCV, real): {speedup_real:.1f}× slower")

    # ---------- 4) End-to-end (Controlled) pipelines ----------
    times_cv_ctrl = time_distribution(opencv_pipeline_controlled, img_in, n_runs=N_RUNS)
    times_ski_ctrl = time_distribution(skimage_pipeline_controlled, img_in, n_runs=N_RUNS)

    times_cv_ctrl_ms = times_cv_ctrl * 1e3
    times_ski_ctrl_ms = times_ski_ctrl * 1e3

    mean_cv_ctrl = np.mean(times_cv_ctrl_ms)
    std_cv_ctrl  = np.std(times_cv_ctrl_ms)

    mean_ski_ctrl = np.mean(times_ski_ctrl_ms)
    std_ski_ctrl  = np.std(times_ski_ctrl_ms)

    mean_cv_ctrl_s = np.mean(times_cv_ctrl)
    mean_ski_ctrl_s = np.mean(times_ski_ctrl)
    speedup_ctrl = mean_ski_ctrl_s / mean_cv_ctrl_s if mean_cv_ctrl_s > 0 else float("inf")

    print("\n=== End-to-end (Controlled) resize + COMMON YUV ===")
    print(f"{'Pipeline':<15} {'mean [ms]':>12} {'std [ms]':>12}")
    print("-" * 42)
    print(f"{'OpenCV ctrl':<15} {mean_cv_ctrl:12.3f} {std_cv_ctrl:12.3f}")
    print(f"{'skimage ctrl':<15} {mean_ski_ctrl:12.3f} {std_ski_ctrl:12.3f}")
    print(f"\nSpeedup (skimage / OpenCV, ctrl): {speedup_ctrl:.1f}× slower")

    # ---------- 5) Numerical differences ----------
    # (a) Controlled: error due to resize only (same YUV)
    yuv_cv_ctrl = opencv_pipeline_controlled(img_in)
    yuv_ski_ctrl = skimage_pipeline_controlled(img_in)
    g_mad_ctrl, ch_mad_ctrl = mean_abs_diff(yuv_cv_ctrl, yuv_ski_ctrl)

    # (b) Realistic: error includes different YUV implementations
    yuv_cv_real = opencv_pipeline_real(img_in)
    yuv_ski_real = skimage_pipeline_real(img_in)
    g_mad_real, ch_mad_real = mean_abs_diff(yuv_cv_real, yuv_ski_real)

    print("\n=== Numerical differences (YUV) ===")
    print("Controlled (same YUV math, difference from resize/interpolation only):")
    print(f"  Global MAD: {g_mad_ctrl:.6f}")
    print(f"  Y: {ch_mad_ctrl[0]:.6f}, U: {ch_mad_ctrl[1]:.6f}, V: {ch_mad_ctrl[2]:.6f}")

    print("\nRealistic (library YUV: OpenCV vs skimage.rgb2yuv, + dtype effects):")
    print(f"  Global MAD: {g_mad_real:.6f}")
    print(f"  Y: {ch_mad_real[0]:.6f}, U: {ch_mad_real[1]:.6f}, V: {ch_mad_real[2]:.6f}")

    print("\nOutput shapes (real pipelines):",
          yuv_cv_real.shape, yuv_ski_real.shape)

    # ---------- 6) Boxplot for Realistic pipelines ----------
    plt.figure()
    plt.boxplot([times_cv_real_ms, times_ski_real_ms])
    plt.xticks([1, 2], ["OpenCV real", "scikit-image real"])
    plt.ylabel("Latency (ms per frame)")
    plt.title("Latency Distribution: 4K → 640×360 + RGB→YUV (Realistic pipelines)")
    plt.grid(True)
    plt.show()
