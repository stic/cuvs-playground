# ivf_pq.py
# Requirements: cupy, cuvs (Python), a CUDA-capable GPU.

import os, csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cupy as cp

from cuvs.neighbors import brute_force as bf
from cuvs.neighbors import ivf_pq

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
import rmm


# ---------------------------
# Config (tweak as needed)
# ---------------------------
N = int(os.getenv("N", 3000_000))     # database vectors
D = int(os.getenv("D", 256))         # dimensions
Q = int(os.getenv("Q", 1_000))       # queries
K = int(os.getenv("K", 10))          # top-k
METRIC = os.getenv("METRIC", "sqeuclidean")  # or "cosine"
BATCH = int(os.getenv("BATCH", 256))  # micro-batch size for timing

def recommend_nlists(N):
    raw = int(4 * np.sqrt(N))
    # snap to nearest power of two between 256 and 16384
    p2 = 2**int(np.round(np.log2(np.clip(raw, 256, 16384))))
    return int(p2)

N_LISTS = int(os.getenv("N_LISTS", recommend_nlists(N)))
ratios = [0.005, 0.01, 0.02, 0.05]
N_PROBES_LIST = [max(1, int(np.ceil(r * N_LISTS))) for r in ratios]

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_PATH = RUNS_DIR / (datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".csv")

# ---------------------------
# Helpers
# ---------------------------
def normalize_search_output(out):
    """Return (I, D) regardless of whether search() returns (I,D) or (D,I)."""
    a, b = out
    is_int = lambda arr: arr.dtype.kind in "iu"
    is_float = lambda arr: arr.dtype.kind == "f"
    if is_int(a) and is_float(b):
        return a, b
    if is_int(b) and is_float(a):
        return b, a
    raise ValueError("Cannot disambiguate search() outputs; got dtypes:", a.dtype, b.dtype)

def cuda_version_str():
    v = cp.cuda.runtime.runtimeGetVersion()
    major = v // 1000
    minor = (v - major * 1000) // 10
    return f"{major}.{minor}"

def gpu_name_str():
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    return name.decode() if isinstance(name, bytes) else str(name)

def gpu_mem_mb():
    h = nvmlDeviceGetHandleByIndex(0)
    m = nvmlDeviceGetMemoryInfo(h)
    return m.used / (1024**2)

def time_gpu_ms(fn, *args, **kwargs):
    """Time a single GPU operation with CUDA events; returns (result, elapsed_ms)."""
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    res = fn(*args, **kwargs)
    end.record()
    end.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    return res, float(elapsed_ms)

def batched_search(search_callable, queries, k, batch_size=BATCH, track_vram_peak=False):
    """
    Runs search in micro-batches for robust latency stats.
    search_callable: function(q_chunk, k) -> (I_chunk, D_chunk) (any return order ok)
    Returns: I, D, p50_ms, p95_ms, qps, peak_used_mb (if track_vram_peak)
    """
    Q = queries.shape[0]
    I_chunks, D_chunks, lat_ms = [], [], []
    peak_used = gpu_mem_mb() if track_vram_peak else None

    for i in range(0, Q, batch_size):
        q_chunk = queries[i:i+batch_size]

        # Time just the search; normalize outputs
        (out, dt_ms) = time_gpu_ms(search_callable, q_chunk, k)
        I_chunk, D_chunk = normalize_search_output(out)
        I_chunks.append(I_chunk)
        D_chunks.append(D_chunk)
        lat_ms.append(dt_ms)

        if track_vram_peak:
            peak_used = max(peak_used, gpu_mem_mb())

    I = cp.concatenate(I_chunks, axis=0)
    D = cp.concatenate(D_chunks, axis=0)

    total_ms = float(sum(lat_ms))
    p50_ms = float(np.percentile(lat_ms, 50))
    p95_ms = float(np.percentile(lat_ms, 95))
    qps = float(Q / (total_ms / 1000.0))

    if track_vram_peak:
        return I, D, p50_ms, p95_ms, qps, float(peak_used)
    else:
        return I, D, p50_ms, p95_ms, qps, None

def recall_at_k(I_pred, I_true, k):
    """Set-based recall@k averaged over queries. I_* are (Q,k) int arrays (CuPy)."""
    I_pred = I_pred[:, :k]
    I_true = I_true[:, :k]
    # (Q,k,1) vs (Q,1,k) -> (Q,k,k) matches; any over last axis -> (Q,k)
    matches = (I_pred[..., None] == I_true[:, None, :]).any(-1)
    hits_per_q = matches.sum(axis=1)  # (Q,)
    return float(hits_per_q.mean() / k)

def write_rows(rows):
    fieldnames = [
        "timestamp",
        "lib",
        "index",
        "n", "dim", "q", "k",
        "metric",
        "build_ms",
        "n_probes",
        "scanned_frac",
        "cands_per_query",
        "recall@10",
        "p50_ms", "p95_ms", "qps",
        "vram_peak_mb",
        "vram_used_build_mb",
        "gpu_name",
        "cuda_version",
        "notes",
    ]
    file_exists = RUNS_PATH.exists()
    with RUNS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------------------------
# Data
# ---------------------------
cp.random.seed(17)
xb = cp.random.random((N, D), dtype=cp.float32)
xq = cp.random.random((Q, D), dtype=cp.float32)

gpu_name = gpu_name_str()
cuda_version = cuda_version_str()
notes = os.getenv("NOTES", "")

rows = []
nvmlInit()
rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=6 * 1024**3,  # 6GiB pool for a 6GB GPU
)

# ---------------------------
# Brute-force baseline (GT)
# ---------------------------
# Build BF index
bf_build_start_used = gpu_mem_mb()
(gt_index, bf_build_ms) = time_gpu_ms(
    lambda: bf.build(xb, metric=METRIC)
)
bf_used_after_build = gpu_mem_mb()
bf_vram_used_build = max(0.0, bf_used_after_build - bf_build_start_used)
bf_peak_used = bf_used_after_build  # will update while searching

# Batched BF search (exact), also collect p50/p95/qps
def bf_search_callable(q_chunk, k):
    return bf.search(gt_index, q_chunk, k=k)

I_gt, D_gt, bf_p50, bf_p95, bf_qps, bf_peak_used = batched_search(
    bf_search_callable, xq, K, batch_size=BATCH, track_vram_peak=True
)

# Log brute-force row
rows.append({
    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    "lib": "cuVS",
    "index": "Brute-Force",
    "n": N, "dim": D, "q": Q, "k": K,
    "metric": METRIC,
    "build_ms": round(bf_build_ms, 3),
    "n_probes": 0,
    "scanned_frac": 1.0,  # always 1.0 for BF; same as
    "cands_per_query": N,  # always N for BF; same as
    "recall@10": round(1.0, 6),              # exact
    "p50_ms": round(bf_p50, 3),
    "p95_ms": round(bf_p95, 3),
    "qps": round(bf_qps, 3),
    "vram_peak_mb": round(bf_peak_used, 1),
    "vram_used_build_mb": round(bf_vram_used_build, 1),
    "gpu_name": gpu_name,
    "cuda_version": cuda_version,
    "notes": notes,
})

rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=6 * 1024**3,  # 6GiB pool for a 6GB GPU
)
# ---------------------------
# IVF-PQ build
# ---------------------------
ivfpq_build_start_used = gpu_mem_mb()
(ivfpq_index, ivfpq_build_ms) = time_gpu_ms(
    lambda: ivf_pq.build(ivf_pq.IndexParams(n_lists=N_LISTS, metric=METRIC, pq_bits=8, pq_dim=32), xb)
)
ivfpq_used_after_build = gpu_mem_mb()
ivfpq_vram_used_build = max(0.0, ivfpq_used_after_build - ivfpq_build_start_used)

# ---------------------------
# IVF-PQ searches (sweep n_probes)
# ---------------------------
peak_used_all = ivfpq_used_after_build

def make_ivf_pq_search_callable(nprobes):
    sp = ivf_pq.SearchParams(n_probes=nprobes)
    def _call(q_chunk, k):
        return ivf_pq.search(sp, ivfpq_index, q_chunk, k)
    return _call

for nprobes in N_PROBES_LIST:
    ivf_search_callable = make_ivf_pq_search_callable(nprobes)
    I_ann, D_ann, p50_ms, p95_ms, qps, peak_used = batched_search(
        ivf_search_callable, xq, K, batch_size=BATCH, track_vram_peak=True
    )
    peak_used_all = max(peak_used_all, peak_used)

    r10 = recall_at_k(I_ann, I_gt, K)

    rows.append({
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "lib": "cuVS",
        "index": "IVF-PQ",
        "n": N, "dim": D, "q": Q, "k": K,
        "metric": METRIC,
        "build_ms": round(ivfpq_build_ms, 3),
        "n_probes": nprobes,
        "scanned_frac": round(nprobes / N_LISTS, 4),
        "cands_per_query": int((N / N_LISTS) * nprobes),
        "recall@10": round(r10, 6),
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "qps": round(qps, 3),
        "vram_peak_mb": round(peak_used_all, 1),
        "vram_used_build_mb": round(ivfpq_vram_used_build, 1),
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "notes": notes,
    })

nvmlShutdown()

# ---------------------------
# Write results
# ---------------------------
write_rows(rows)
print(f"Wrote {len(rows)} rows to {RUNS_PATH}")
