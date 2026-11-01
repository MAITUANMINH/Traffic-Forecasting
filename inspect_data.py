import os
import numpy as np

ROOT = "data"
for dirpath, _, files in os.walk(ROOT):
    for f in sorted(files):
        path = os.path.join(dirpath, f)
        size_mb = os.path.getsize(path) / 1024**2
        if f.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            print(f"{path:60s} shape={arr.shape} dtype={arr.dtype} ({size_mb:.0f} MB)")
        else:
            print(f"{path:60s} ({size_mb:.0f} MB)")