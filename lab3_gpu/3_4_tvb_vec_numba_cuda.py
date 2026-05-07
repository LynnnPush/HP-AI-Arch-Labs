"""Single TVB simulation on GPU using Numba's @cuda.jit.

This is a low-level CUDA-style implementation: we write explicit kernels
that are launched on a 1-D grid of N threads (one thread per brain region).
Each timestep launches two kernels — one to compute delayed coupling, one
to advance the state via the MLP + Forward Euler.

NOTE on multi-simulation throughput (see __main__ comment block at bottom):
the kernels here are written for a single simulation. To batch S sims we
would extend the grid to 2-D (S x N), add a leading "sim" axis to Xs, and
index Xs[s, n, m, t] inside the kernels. W and D can stay 2-D and be
shared across the batch.
"""
import math
import time
import numpy as np
from numba import cuda, float64, int64
from typing import List, Tuple

from lib import data
from lib.mlp_params import (
    layer_1_b_np, layer_1_w_np, layer_2_b_np, layer_2_w_np, MLP_L, MLP_M,
)

# Hidden layer width baked in as a compile-time constant so we can declare a
# fixed-size local array inside the kernel (CUDA local arrays must be sized
# from a literal or constant).
L_HIDDEN = MLP_L   # 64
M_VARS   = MLP_M   # 2

THREADS_PER_BLOCK = 128


@cuda.jit
def coupling_kernel(Xs, W, D_ts, c_ins, t, N):
    """Compute delayed coupling c_in for every destination node.

    Grid:  N threads, one per destination node.
    Each thread loops over all sources, gathers the delayed sample from
    Xs[src, 0, t - D_ts[dst, src]], applies pre/post functions, and writes
    the resulting scalar to c_ins[dst].
    """
    dst = cuda.grid(1)
    if dst >= N:
        return

    # Destination state at t-1 (used by pre()).
    x_dst = Xs[dst, 0, t - 1]

    acc = 0.0
    for src in range(N):
        d = D_ts[dst, src]
        # Self-loops and zero-delay edges read the previous timestep.
        if dst == src or d == 0:
            ti = t - 1
            valid = True
        else:
            ti = t - d
            valid = (t >= d)   # outside simulated history -> contribute 0
        if valid:
            x_src = Xs[src, 0, ti]
            # pre(x_src, x_dst) = x_src - 1.0
            acc += W[dst, src] * (x_src - 1.0)
        # else: contribute nothing

    # post(gx) = 1e-3 * gx
    c_ins[dst] = 1e-3 * acc


@cuda.jit
def step_kernel(Xs, c_ins, t, dt, N,
                W1, B1, W2, B2):
    """Advance one node by one Forward Euler step.

    Grid: N threads, one per node.
    Each thread:
      1) reads its M-dim state X at t-1,
      2) runs the 2-layer MLP (matmul + ReLU + matmul) entirely in registers
         / per-thread local memory,
      3) injects the coupling into the 2nd state variable,
      4) writes Xs[:, :, t] = X + fx * dt.
    """
    n = cuda.grid(1)
    if n >= N:
        return

    # Per-thread local arrays — sizes are compile-time constants.
    x = cuda.local.array(M_VARS, dtype=float64)
    h = cuda.local.array(L_HIDDEN, dtype=float64)
    fx = cuda.local.array(M_VARS, dtype=float64)

    # Load current state x = Xs[n, :, t-1]
    for m in range(M_VARS):
        x[m] = Xs[n, m, t - 1]

    # Layer 1: hidden[l] = ReLU(sum_m x[m] * W1[m, l] + B1[l])
    for l in range(L_HIDDEN):
        s = B1[l]
        for m in range(M_VARS):
            s += x[m] * W1[m, l]
        h[l] = s if s > 0.0 else 0.0

    # Layer 2: fx[m] = sum_l h[l] * W2[l, m] + B2[m]
    for m in range(M_VARS):
        s = B2[m]
        for l in range(L_HIDDEN):
            s += h[l] * W2[l, m]
        fx[m] = s

    # Inject coupling into the second state variable, then Forward Euler.
    fx[1] += c_ins[n]
    for m in range(M_VARS):
        Xs[n, m, t] = x[m] + fx[m] * dt


@cuda.jit
def init_kernel(Xs, N, value):
    """Set Xs[:, :, 0] = value as the initial condition."""
    n = cuda.grid(1)
    if n >= N:
        return
    for m in range(M_VARS):
        Xs[n, m, 0] = value


def simulate(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
) -> Tuple[List[float], np.ndarray, float]:
    """Run a single TVB simulation on GPU using @cuda.jit kernels.

    Returns (T, Xs_cpu, gpu_time_seconds). gpu_time_seconds excludes the
    initial host->device upload and final device->host transfer to mirror the
    timing convention used in 3_1_3_tvb_vec_cupy.simulate.
    """
    total_timesteps = int(tf / dt)

    # Convert distances to integer timestep delays (kept as int64 for safe indexing).
    D_ts_np = ((D / speed) / dt).astype(np.int64)

    # Host->device transfers (per-dataset matrices + MLP weights).
    W_d   = cuda.to_device(W.astype(np.float64))
    D_d   = cuda.to_device(D_ts_np)
    W1_d  = cuda.to_device(layer_1_w_np.astype(np.float64))   # [M, L]
    B1_d  = cuda.to_device(layer_1_b_np.astype(np.float64))   # [L]
    W2_d  = cuda.to_device(layer_2_w_np.astype(np.float64))   # [L, M]
    B2_d  = cuda.to_device(layer_2_b_np.astype(np.float64))   # [M]

    # Allocate full state history and coupling buffer on device.
    Xs_d    = cuda.device_array((N, M, total_timesteps), dtype=np.float64)
    c_ins_d = cuda.device_array(N, dtype=np.float64)

    # 1-D launch config — one thread per node.
    # // is floor division, so adding (divisor − 1) to the numerator before flooring rounds up instead of down.
    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Initial condition Xs[:, :, 0] = -1.0
    init_kernel[blocks, THREADS_PER_BLOCK](Xs_d, N, -1.0)

    # Synchronize so the timer captures only the simulation loop on the device.
    cuda.synchronize()
    start = time.time()

    for t in range(1, total_timesteps):
        # Two-kernel timestep: coupling first (reads Xs[:, :, < t]), then step
        # (writes Xs[:, :, t]). The implicit per-launch ordering on the default
        # stream guarantees the step sees the freshly-written c_ins values.
        coupling_kernel[blocks, THREADS_PER_BLOCK](Xs_d, W_d, D_d, c_ins_d, t, N)
        step_kernel[blocks, THREADS_PER_BLOCK](
            Xs_d, c_ins_d, t, dt, N, W1_d, B1_d, W2_d, B2_d,
        )

    cuda.synchronize()
    end = time.time()
    gpu_time = end - start

    T = [t * dt for t in range(total_timesteps)]
    print(f"[simulate @cuda.jit] GPU Total Time: {gpu_time:.6f}s")

    Xs_cpu = Xs_d.copy_to_host()
    return T, Xs_cpu, gpu_time


if __name__ == "__main__":
    datasets = [
        ("TVB76",  data.tvb76_weights_lengths),
        ("TVB192", data.tvb192_weights_lengths),
        ("TVB998", data.tvb998_weights_lengths),
    ]

    M = 2
    dt = 0.05
    tf = 150.0
    speed = 4.0

    # ── Warmup ────────────────────────────────────────────────────────────────
    # First @cuda.jit launch triggers PTX compilation. Run a tiny simulation
    # so compile time does not pollute the reported per-dataset timings.
    print("[warmup] JIT-compiling CUDA kernels...")
    W_w, D_w = data.tvb76_weights_lengths()
    _ = simulate(W_w, D_w, len(W_w), M, dt, tf=1.0, speed=speed)
    print("[warmup] done.\n")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    for name, loader in datasets:
        print(f"\n{'='*40}")
        print(f"Running {name}  (tf={tf}ms, dt={dt}ms)  [@cuda.jit]")
        print(f"{'='*40}")
        W, D = loader()
        N = len(W)
        wall_start = time.time()
        T, Xs, gpu_time = simulate(W, D, N, M, dt, tf, speed)
        wall_end = time.time()
        print(f"[{name}] Wall-clock total (incl. transfers): {wall_end - wall_start:.6f}s")

    # ── Notes on multi-simulation extension (NOT implemented) ─────────────────
    #
    # To run S simulations concurrently on one GPU with @cuda.jit, the changes
    # would be:
    #
    #   1) Add a leading "sim" axis to the state history: Xs shape becomes
    #      [S, N, M, T]. W and D can stay [N, N] and be shared across sims
    #      (same as 3_2_tvb_vec_cupy_throughput.py).
    #
    #   2) Launch on a 2-D grid of size (S, N) — e.g. block dim (1, 128),
    #      grid dim (S, ceil(N/128)). Each thread handles one (sim, node)
    #      pair. Inside the kernel use `s, n = cuda.grid(2)` and index
    #      Xs[s, n, m, t].
    #
    #   3) c_ins becomes [S, N] and is indexed c_ins[s, n] inside both
    #      kernels.
    #
    #   4) MLP weights stay shared (read-only) across all sims; placing them
    #      in __constant__ memory via `cuda.const.array_like` would let every
    #      thread broadcast-read the same parameters efficiently.
    #
    #   5) For very large S, swap per-thread local arrays for shared-memory
    #      tiles cooperatively loaded by the block, so threads in the same
    #      block can reuse the hidden activations and reduce register pressure.
    #
    #   6) If sims have different W/D (not the case in this assignment),
    #      promote W and D to [S, N, N] and index W[s, dst, src] in the
    #      coupling kernel.
