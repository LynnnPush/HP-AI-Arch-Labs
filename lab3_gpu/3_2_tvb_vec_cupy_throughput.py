"""High-throughput TVB on GPU with CuPy.

Runs S independent TVB simulations in parallel on a single GPU by adding a
leading "simulation" axis to all state tensors. All simulations share the
same W and D matrices (per the assignment), so connectivity tensors stay 2-D
and broadcast across the batch.
"""
import time
import numpy as np
import cupy as cp
from typing import List, Tuple

from lib import data
from lib.mlp_params import layer_1_b_np, layer_1_w_np, layer_2_b_np, layer_2_w_np

# MLP weights live on GPU for the entire run (shared across all sims).
layer_1_w_gpu = cp.array(layer_1_w_np)   # [M, L]
layer_1_b_gpu = cp.array(layer_1_b_np)   # [L]
layer_2_w_gpu = cp.array(layer_2_w_np)   # [L, M]
layer_2_b_gpu = cp.array(layer_2_b_np)   # [M]


def pre(x_src: cp.ndarray, x_dst: cp.ndarray) -> cp.ndarray:
    return (x_src - 1.0)


def post(gx: cp.ndarray) -> cp.ndarray:
    return (1e-3 * gx)


def f_batched(X: cp.ndarray) -> cp.ndarray:
    """Two-layer MLP applied to a batch of state tensors.

    X shape: [S, N, M]. matmul broadcasts over the leading S axis, so the
    same MLP weights are applied to every simulation without copying.
    """
    hidden = cp.matmul(X, layer_1_w_gpu) + layer_1_b_gpu   # [S, N, L]
    hidden = cp.where(hidden <= 0, 0.0, hidden)            # ReLU
    out = cp.matmul(hidden, layer_2_w_gpu) + layer_2_b_gpu # [S, N, M]
    return out


def calculate_coupling_batched(
    Xs: cp.ndarray,
    W: cp.ndarray,
    D_timestep: cp.ndarray,
    t: int,
    sim_idx: cp.ndarray,
    src_idx_b: cp.ndarray,
) -> cp.ndarray:
    """Delayed coupling for every (sim, dst) pair on GPU.

    Xs:           [S, N, M, T] full state history (sim, node, var, time).
    W:            [N, N] connectivity weights, shared across simulations.
    D_timestep:   [N, N] integer delays in timesteps, shared across simulations.
    sim_idx:      [S, 1, 1] precomputed broadcasting helper for fancy indexing.
    src_idx_b:    [1, 1, N] precomputed broadcasting helper for fancy indexing.

    Returns c_ins of shape [S, N].
    """
    N = W.shape[0]

    # Pairs (dst, src) whose delayed sample exists in the simulated history.
    valid_time = (t >= D_timestep)                                     # [N, N]

    # Self-loops and zero-delay edges always read the previous step.
    use_prev = cp.eye(N, dtype=bool) | (D_timestep == 0)              # [N, N]
    timesteps_indices = cp.where(use_prev, t - 1, t - D_timestep)     # [N, N]

    # Clamp invalid (out-of-history) entries so fancy indexing stays in-bounds.
    safe_indices = cp.where(valid_time, timesteps_indices, 0)         # [N, N]

    # Gather Xs[s, src, 0, safe_indices[dst, src]] -> x_src[s, dst, src]
    ti = safe_indices[cp.newaxis, :, :]                               # [1, N, N]
    x_src = Xs[sim_idx, src_idx_b, 0, ti]                             # [S, N, N]
    x_src = cp.where(valid_time, x_src, 0.0)                          # mask invalids

    # Destination state at t-1, shape [S, N, 1] to broadcast across src.
    x_dst = Xs[:, :, 0, t - 1][:, :, cp.newaxis]                      # [S, N, 1]

    # Weighted sum over sources; W[N,N] broadcasts over the S axis.
    c_in = cp.sum(W * pre(x_src, x_dst), axis=2)                      # [S, N]
    return post(c_in)                                                  # [S, N]


def step_batched(Xs: cp.ndarray, t: int, c_ins: cp.ndarray, dt: float) -> cp.ndarray:
    """Forward Euler update for the entire batch."""
    X_all = Xs[:, :, :, t - 1]      # [S, N, M]
    fx = f_batched(X_all)           # [S, N, M]
    fx[:, :, 1] += c_ins            # inject coupling into 2nd state variable
    return X_all + fx * dt          # [S, N, M]


def simulate_batched(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
    S: int,
) -> Tuple[List[float], np.ndarray, float]:
    """Run S TVB simulations concurrently on the GPU.

    Returns (T, Xs_cpu, gpu_time_seconds). gpu_time_seconds excludes the
    initial host->device upload and final device->host transfer to mirror the
    reporting style of 3_1_3_tvb_vec_cupy.simulate.
    """
    total_timesteps = int(tf / dt)

    # Shared connectivity tensors uploaded once, reused by every simulation.
    W_gpu = cp.array(W)
    D_timestep_gpu = cp.array(((D / speed) / dt).astype(int))

    # State history with a leading batch axis. Memory ~ S*N*M*T*8 bytes.
    Xs = cp.zeros((S, N, M, total_timesteps), dtype=cp.float64)

    # Precomputed index broadcasting helpers (allocated once, reused every step).
    sim_idx   = cp.arange(S)[:, cp.newaxis, cp.newaxis]   # [S, 1, 1]
    src_idx_b = cp.arange(N)[cp.newaxis, cp.newaxis, :]   # [1, 1, N]

    cp.cuda.Stream.null.synchronize()
    start = time.time()

    for t in range(total_timesteps):
        if t == 0:
            Xs[:, :, :, t] = -1.0   # identical initial condition for every sim
        else:
            c_ins = calculate_coupling_batched(Xs, W_gpu, D_timestep_gpu, t, sim_idx, src_idx_b)
            Xs[:, :, :, t] = step_batched(Xs, t, c_ins, dt)

    cp.cuda.Stream.null.synchronize()
    end = time.time()
    gpu_time = end - start

    T = [t * dt for t in range(total_timesteps)]
    print(f"[simulate_batched S={S}] GPU Time: {gpu_time:.6f}s")

    Xs_cpu = cp.asnumpy(Xs)
    return T, Xs_cpu, gpu_time


if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    M = 2          # state variables per node
    dt = 0.05      # timestep size in ms
    tf = 150.0     # simulation duration in ms
    speed = 4.0    # signal propagation speed in mm/ms
    S_LIST = [100, 250, 500]

    # Load TVB192 once; same W, D used by every simulation in every batch.
    W, D = data.tvb192_weights_lengths()
    N = len(W)
    total_timesteps = int(tf / dt)

    # ── GPU warmup ────────────────────────────────────────────────────────────
    # First CuPy calls trigger NVRTC kernel compilation. Run a tiny batched
    # simulation before any timed measurements so kernel-compile time does
    # not pollute throughput numbers.
    print("[warmup] compiling kernels...")
    _ = simulate_batched(W, D, N, M, dt, tf, speed, S=4)
    print("[warmup] done.\n")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"TVB192  (N={N}, tf={tf}ms, dt={dt}ms, total_timesteps={total_timesteps})")
    print(f"{'='*72}\n")

    results = []   # list of (S, gpu_time, wall_time)
    for S in S_LIST:
        print(f"--- S = {S} ---")
        wall_start = time.time()
        _, _, gpu_time = simulate_batched(W, D, N, M, dt, tf, speed, S)
        wall = time.time() - wall_start
        print(f"[batched] wall (incl. transfers): {wall:.4f}s\n")
        results.append((S, gpu_time, wall))

        # Free the large state buffer before the next batch to avoid OOM.
        cp.get_default_memory_pool().free_all_blocks()

    # ── Summary: throughput in simulation-iterations per second ───────────────
    # One "simulation iteration" = advancing one simulation by one timestep.
    # Total iterations for a batch of S sims = S * total_timesteps.
    print(f"{'='*72}")
    print("THROUGHPUT  (simulation-iterations / second on TVB192, 150ms)")
    print(f"{'='*72}")
    hdr = f"{'S':>5} | {'GPU time':>10} | {'wall time':>10} | {'iters/s (GPU)':>16} | {'iters/s (wall)':>16}"
    print(hdr)
    print("-" * len(hdr))
    for S, t_gpu, t_wall in results:
        iters = S * total_timesteps
        print(f"{S:>5} | {t_gpu:>8.4f}s | {t_wall:>8.4f}s | {iters/t_gpu:>14.0f}   | {iters/t_wall:>14.0f}")
