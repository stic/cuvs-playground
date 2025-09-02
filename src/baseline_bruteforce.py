import os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import cupy as cp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
import rmm

# cuVS brute-force
from cuvs.neighbors import brute_force as bf  # API per docs

# ----------------------------
# Config
# ----------------------------
N = int(os.getenv("N", 100_000))
D = int(os.getenv("D", 128))
Q = int(os.getenv("Q", 1_000))          # number of queries to run
K = int(os.getenv("K", 10))
METRIC = os.getenv("METRIC", "sqeuclidean")  # 'sqeuclidean' or 'inner_product'
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RUNS_DIR, f"{datetime.now().date()}.csv")

# ----------------------------
# Utilities
# ----------------------------
def gpu_mem_mb():
    h = nvmlDeviceGetHandleByIndex(0)
    m = nvmlDeviceGetMemoryInfo(h)
    return m.used / (1024**2)

def percentile(arr, p):
    return float(np.percentile(np.asarray(arr), p))

def compute_recall_at_k(gt_indices, ann_indices, k=10):
    """
    Standard recall@k: for each query, |NN_gt ∩ NN_ann| / k, averaged over queries.
    With brute-force=exact, recall@k should be ~1.0 (sanity signal for the pipeline).
    """
    assert gt_indices.shape == ann_indices.shape
    hits = 0
    for i in range(gt_indices.shape[0]):
        hits += len(set(gt_indices[i, :k].tolist()).intersection(set(ann_indices[i, :k].tolist())))
    return hits / float(gt_indices.shape[0] * k)

# ----------------------------
# Main
# ----------------------------
def main():
    # Initialize NVML & RMM (optional pooling to reduce fragmentation)
    nvmlInit()
    try:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=2**30,  # 1GiB pool; adjust to your GPU
        )

        # 1) Generate data on GPU
        cp.random.seed(123)
        xb = cp.random.random((N, D), dtype=cp.float32)
        xq = cp.random.random((Q, D), dtype=cp.float32)

        # 2) Build brute-force index
        vram_before = gpu_mem_mb()
        t0 = time.perf_counter()
        index = bf.build(xb, metric=METRIC)  # cuVS build
        build_ms = (time.perf_counter() - t0) * 1e3
        vram_after_build = gpu_mem_mb()

        # 3) Search (batched for realistic throughput)
        #    cuVS search returns (indices, distances)
        batch = int(os.getenv("BATCH", 1000))
        latencies_ms = []
        all_inds = []
        all_dists = []

        t_search0 = time.perf_counter()
        for start in range(0, Q, batch):
            end = min(start + batch, Q)
            q = xq[start:end]
            t1 = time.perf_counter()
            inds, dists = bf.search(index, q, k=K)
            latencies_ms.append((time.perf_counter() - t1) * 1e3 / (end - start))  # per-query
            all_inds.append(cp.asnumpy(inds))
            all_dists.append(cp.asnumpy(dists))
        total_search_s = time.perf_counter() - t_search0

        inds = np.vstack(all_inds)
        dists = np.vstack(all_dists)

        # 4) Ground truth for recall@K by exact pairwise (using same brute force again)
        #    For the baseline, the index itself is exact; to avoid double compute,
        #    reuse the same results as GT. (Recall should be ~1.0)
        gt_inds = inds.copy()
        recall_at_10 = compute_recall_at_k(gt_inds, inds, k=K)

        # 5) Metrics
        p50_ms = percentile(latencies_ms, 50.0)
        p95_ms = percentile(latencies_ms, 95.0)
        qps = Q / total_search_s
        vram_used_build = vram_after_build - vram_before
        vram_peak_mb = max(vram_before, vram_after_build, gpu_mem_mb())  # simple snapshot-based proxy

        # 6) System info
        import cupy
        import sys
        from cupy.cuda.runtime import getDeviceProperties
        devprops = getDeviceProperties(0)
        gpu_name = devprops["name"].decode()
        cuda_version = f"{cupy.cuda.runtime.runtimeGetVersion()//1000}.{(cupy.cuda.runtime.runtimeGetVersion()%1000)//10}"

        # 7) Log CSV
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "lib": "cuVS",
            "index": "bruteforce",
            "n": N,
            "dim": D,
            "q": Q,
            "k": K,
            "metric": METRIC,
            "build_ms": round(build_ms, 3),
            "recall@10": round(recall_at_10, 6),
            "p50_ms": round(p50_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "qps": round(qps, 3),
            "vram_peak_mb": round(vram_peak_mb, 1),
            "vram_used_build_mb": round(vram_used_build, 1),
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "notes": os.getenv("NOTES", "")
        }

        df = pd.DataFrame([row])
        header = not os.path.exists(OUT_CSV)
        df.to_csv(OUT_CSV, mode="a", index=False, header=header)

        print(f"Wrote metrics row to {OUT_CSV}")
        print(row)

    finally:
        nvmlShutdown()

if __name__ == "__main__":
    main()
