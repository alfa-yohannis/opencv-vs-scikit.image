sudo taskset -c 0 \
systemd-run --scope \
  -p CPUQuota=25% \
  -p MemoryMax=512M \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 experiment.py

# sudo taskset -c 0 \
# systemd-run --scope \
#   -p CPUQuota=25% \
#   -p MemoryMax=512M \
#   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
#   python3 experiment_skimage.py

# sudo taskset -c 0 \
# systemd-run --scope \
#   -p CPUQuota=25% \
#   -p MemoryMax=512M \
#   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
#   python3 experiment_opencv.py

sudo python3 boxplot.py
