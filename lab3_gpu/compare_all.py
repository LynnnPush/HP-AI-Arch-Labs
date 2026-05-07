"""Run all TVB implementations and print a side-by-side timing comparison table.

Node-loop (tvb_vec) is very slow for TVB998 (~998 x 3000 iterations).
Set SKIP_SLOW_THRESHOLD to limit which datasets it runs on, or set to None
to run everything.
"""
import importlib
import time
import sys
import numpy as np
from lib import data

# ── Config ────────────────────────────────────────────────────────────────────
M = 2       # state variables per node
dt = 0.05   # timestep in ms
tf = 150.0  # simulation duration in ms
speed = 4.0 # propagation speed in mm/ms

# Node-loop impl is skipped for datasets with more nodes than this
SKIP_SLOW_THRESHOLD = None   # set to None to run all datasets on all impls

VERIFY_CORRECTNESS = True   # compare outputs of each impl against node-loop
RTOL = 1e-5                 # relative tolerance for np.allclose
# ──────────────────────────────────────────────────────────────────────────────

datasets = [
    ("TVB76",  data.tvb76_weights_lengths),
    ("TVB192", data.tvb192_weights_lengths),
    ("TVB998", data.tvb998_weights_lengths),
]

# ── Load implementations via importlib (names start with digits) ──────────────
impls = []

# Baseline: per-node loop (tvb_vec.py)
mod = importlib.import_module("tvb_vec")
impls.append(("node_loop", mod.simulate, False))

# Step 2: fully vectorized NumPy
mod = importlib.import_module("3_1_2_tvb_vec_numpyNode")
impls.append(("numpy_vec", mod.simulate, False))

# Step 3: CuPy GPU — skip gracefully if CUDA not available
try:
    mod = importlib.import_module("3_1_3_tvb_vec_cupy")
    impls.append(("cupy_gpu", mod.simulate, True))
    print("[info] CuPy found — GPU implementation will run.")
except (ImportError, ModuleNotFoundError):
    print("[warn] CuPy not available — skipping GPU implementation.")

# ── Run ───────────────────────────────────────────────────────────────────────
# results[impl_label][dataset_name] = (wall_time, Xs_cpu)
results = {label: {} for label, _, _ in impls}
reference = {}   # dataset_name -> Xs from node_loop (used for correctness check)

for ds_name, loader in datasets:
    W, D = loader()
    N = len(W)
    print(f"\n{'='*56}")
    print(f"Dataset: {ds_name}  (N={N}, tf={tf}ms, dt={dt}ms)")
    print(f"{'='*56}")

    for label, simulate_fn, is_gpu in impls:
        if (SKIP_SLOW_THRESHOLD is not None
                and label == "node_loop"
                and N > SKIP_SLOW_THRESHOLD):
            results[label][ds_name] = (None, None)
            print(f"  [{label:12s}] skipped (N={N} > threshold {SKIP_SLOW_THRESHOLD})")
            continue

        t0 = time.time()
        T, Xs = simulate_fn(W, D, N, M, dt, tf, speed)
        wall = time.time() - t0
        results[label][ds_name] = (wall, Xs)
        suffix = "(incl. GPU transfers)" if is_gpu else ""
        print(f"  [{label:12s}] wall={wall:.4f}s  {suffix}")

        # Store node_loop output as reference for correctness checking
        if label == "node_loop" and VERIFY_CORRECTNESS:
            reference[ds_name] = Xs

    # Correctness check: compare every non-reference impl against node_loop
    if VERIFY_CORRECTNESS and ds_name in reference:
        ref_Xs = reference[ds_name]
        for label, _, _ in impls:
            if label == "node_loop":
                continue
            _, cand_Xs = results[label].get(ds_name, (None, None))
            if cand_Xs is None:
                continue
            match = np.allclose(ref_Xs, cand_Xs, rtol=RTOL, equal_nan=False)
            status = "OK" if match else "MISMATCH"
            print(f"  [correctness] {label:12s} vs node_loop: {status}")

# ── Summary table ─────────────────────────────────────────────────────────────
ds_names = [n for n, _ in datasets]
col_w = 14

print(f"\n{'='*56}")
print("TIMING SUMMARY (wall-clock seconds)")
print(f"{'='*56}")
header = f"{'Implementation':<14}" + "".join(f"{n:>{col_w}}" for n in ds_names)
print(header)
print("-" * len(header))

for label, _, _ in impls:
    row = f"{label:<14}"
    for ds_name in ds_names:
        t, _ = results[label].get(ds_name, (None, None))
        cell = f"{t:.4f}s" if t is not None else "skipped"
        row += f"{cell:>{col_w}}"
    print(row)

# Speedup rows relative to node_loop
print()
print("SPEEDUP vs node_loop")
print("-" * len(header))
for label, _, _ in impls:
    if label == "node_loop":
        continue
    row = f"{label:<14}"
    for ds_name in ds_names:
        t_impl, _ = results[label].get(ds_name, (None, None))
        t_base, _ = results["node_loop"].get(ds_name, (None, None))
        if t_impl and t_base:
            cell = f"{t_base / t_impl:.1f}x"
        else:
            cell = "n/a"
        row += f"{cell:>{col_w}}"
    print(row)
