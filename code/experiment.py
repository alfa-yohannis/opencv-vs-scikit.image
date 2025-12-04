import time
import numpy as np
import cv2
from skimage import transform as sktf
from skimage import color as skcol

# ---------- Setup: load or create test data ----------
# Option A: load a real image
# img_bgr = cv2.imread("test.jpg")
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Option B: synthetic image (for easy reproducibility)
H, W = 2160, 3840  # 4K
img_rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

# Work in float32 [0,1] for BOTH pipelines
img_f = img_rgb.astype(np.float32) / 255.0

target_size = (360, 640)  # (height, width)


# ---------- Pipelines ----------
def opencv_pipeline(img: np.ndarray) -> np.ndarray:
    """
    img: float32 RGB in [0,1]
    OpenCV resize + OpenCV RGB->YUV
    """
    resized = cv2.resize(
        img,
        (target_size[1], target_size[0]),  # (width, height)
        interpolation=cv2.INTER_LINEAR,    # linear interpolation
    )
    # RGB -> YUV (OpenCV)
    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    return yuv


def skimage_pipeline(img: np.ndarray) -> np.ndarray:
    """
    img: float32 RGB in [0,1]
    skimage resize + skimage RGB->YUV
    """
    resized = sktf.resize(
        img,
        (target_size[0], target_size[1], 3),  # (height, width, channels)
        order=1,               # linear interpolation
        anti_aliasing=False,   # closer to OpenCV's INTER_LINEAR
        preserve_range=True,   # keep [0,1] range
    ).astype(np.float32)

    # RGB -> YUV using skimage's color conversion
    yuv = skcol.rgb2yuv(resized)
    return yuv


# ---------- Simple timing utility ----------
def time_function(f, img, n_warmup=10, n_runs=100):
    # Warm-up (avoid first-call overhead)
    for _ in range(n_warmup):
        _ = f(img)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = f(img)
    end = time.perf_counter()

    total = end - start
    return total / n_runs  # average seconds per run


if __name__ == "__main__":
    n_runs = 10

    t_cv2 = time_function(opencv_pipeline, img_f, n_runs=n_runs)
    t_ski = time_function(skimage_pipeline, img_f, n_runs=n_runs)

    print(f"OpenCV pipeline:        {t_cv2 * 1e3:.3f} ms per image")
    print(f"scikit-image pipeline:  {t_ski * 1e3:.3f} ms per image")

    # Optional: check outputs (they won't be identical, but closer than before)
    yuv_cv2 = opencv_pipeline(img_f)
    yuv_ski = skimage_pipeline(img_f)

    print("Output shapes:", yuv_cv2.shape, yuv_ski.shape)
    print("Mean abs diff (rough sanity check):",
          np.mean(np.abs(yuv_cv2 - yuv_ski)))
