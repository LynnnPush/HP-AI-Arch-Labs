import time
import numpy as np
import cupy as cp
from typing import List

from lib import data, plot
from lib.mlp_params import layer_1_b_np, layer_1_w_np, layer_2_b_np, layer_2_w_np

# Upload MLP weights to GPU once at import time to avoid repeated transfers
layer_1_w_gpu = cp.array(layer_1_w_np)   # [M, L]
layer_1_b_gpu = cp.array(layer_1_b_np)   # [L]
layer_2_w_gpu = cp.array(layer_2_w_np)   # [L, M]
layer_2_b_gpu = cp.array(layer_2_b_np)   # [M]


def pre(x_src: cp.ndarray, x_dst: cp.ndarray) -> cp.ndarray:
    return (x_src - 1.0)


def post(gx: cp.ndarray) -> cp.ndarray:
    return (1e-3 * gx)


def f(X: cp.ndarray) -> cp.ndarray:
    """Two-layer MLP dynamics executed entirely on GPU.

    Args:
        X: State array shaped as [N, M], already on GPU.
    """
    # Layer 1: linear + ReLU
    hidden = cp.matmul(X, layer_1_w_gpu) + layer_1_b_gpu   # [N, L]
    hidden = cp.where(hidden <= 0, 0.0, hidden)             # ReLU activation
    # Layer 2: linear output
    out = cp.matmul(hidden, layer_2_w_gpu) + layer_2_b_gpu  # [N, M]
    return out


def calculate_coupling(
    Xs: cp.ndarray,
    W: cp.ndarray,
    D_timestep: cp.ndarray,
    t: int,
) -> cp.ndarray:
    """Compute delayed coupling for all destination nodes on GPU.

    Args:
        Xs: State history shaped as [N, M, T], on GPU.
        W: Connectivity weight matrix [N, N], on GPU.
        D_timestep: Integer delay matrix [N, N] in timesteps, on GPU.
        t: Current timestep index (t > 0).

    Returns:
        c_ins: Post-synaptic coupling contributions shaped as [N], on GPU.
    """
    N = Xs.shape[0]

    # Mask for pairs whose delay fits within the simulated history
    valid_time = (t >= D_timestep)                                     # [N, N]

    # Self-connections and zero-delay links always read the immediately prior step
    use_prev = cp.eye(N, dtype=bool) | (D_timestep == 0)              # [N, N]
    timesteps_indices = cp.where(use_prev, t - 1, t - D_timestep)     # [N, N]

    # Clamp out-of-range indices to 0 so fancy indexing never goes OOB
    safe_indices = cp.where(valid_time, timesteps_indices, 0)          # [N, N]

    # Gather delayed source states: x_src[dst, src] = Xs[src, 0, delay_idx]
    src_idx = cp.arange(N)[cp.newaxis, :]                              # [1, N]
    x_src = Xs[src_idx, 0, safe_indices]                               # [N, N]
    x_src = cp.where(valid_time, x_src, 0.0)                          # zero invalid pairs

    # Destination state at t-1, broadcast across all source columns
    x_dst = Xs[:, 0, t - 1][:, cp.newaxis]                            # [N, 1]

    # Weighted sum of pre-synaptic signals, then post-synaptic scaling
    c_in = cp.sum(W * pre(x_src, x_dst), axis=1)                      # [N]
    return post(c_in)                                                   # [N]


def step(Xs: cp.ndarray, t: int, c_ins: cp.ndarray, dt: float) -> cp.ndarray:
    """Advance all node states by one Forward Euler step on GPU.

    Args:
        Xs: State history shaped as [N, M, T], on GPU.
        t: Current timestep index.
        c_ins: Coupling inputs shaped as [N], on GPU.
        dt: Simulation timestep size.

    Returns:
        X_new: Updated states shaped as [N, M], on GPU.
    """
    X_all = Xs[:, :, t - 1]     # [N, M] — current state
    fx = f(X_all)                # [N, M] — MLP derivatives
    fx[:, 1] += c_ins            # inject coupling into second state variable
    return X_all + fx * dt       # [N, M] — Euler update


def simulate(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
) -> tuple[List[float], np.ndarray]:
    """Run the GPU-accelerated TVB simulation loop.

    Connectivity and state history live entirely on the GPU throughout the
    simulation; only the final result is transferred back to CPU.

    Args:
        W: Connectivity weight matrix [N, N], on CPU.
        D: Distance matrix [N, N] in mm, on CPU.
        N: Number of brain regions.
        M: Number of state variables per node.
        dt: Simulation timestep in ms.
        tf: Total simulation duration in ms.
        speed: Signal propagation speed in mm/ms.

    Returns:
        T: Time axis list (length = total_timesteps).
        Xs_cpu: State history [N, M, T] transferred back to CPU.
    """
    total_timesteps = int(tf / dt)

    # Upload per-dataset matrices to GPU
    W_gpu = cp.array(W)
    D_timestep_gpu = cp.array(((D / speed) / dt).astype(int))  # convert distances to integer timestep delays

    # Allocate full state history buffer on GPU
    Xs = cp.zeros((N, M, total_timesteps), dtype=cp.float64)

    # Warmup: run two timesteps and discard to trigger CUDA kernel compilation
    # (CuPy compiles kernels via NVRTC on first call and caches the .cubin).
    # Without this, the first dataset absorbs compilation cost (~seconds) that
    # is not representative of steady-state GPU performance.
    Xs[:, :, 0] = -1.0
    _c = calculate_coupling(Xs, W_gpu, D_timestep_gpu, 1)
    Xs[:, :, 1] = step(Xs, 1, _c, dt)
    cp.cuda.Stream.null.synchronize()
    Xs[:] = 0.0   # reset state before the real timed run
    del _c

    # Synchronize so the timer starts only after all uploads and warmup are done
    cp.cuda.Stream.null.synchronize()
    start = time.time()

    for t in range(total_timesteps):
        if t == 0:
            Xs[:, :, t] = -1.0   # initial condition for all nodes
        else:
            c_ins = calculate_coupling(Xs, W_gpu, D_timestep_gpu, t)   # [N]
            Xs[:, :, t] = step(Xs, t, c_ins, dt)                       # [N, M]

    # Synchronize before stopping the timer so all queued GPU work is counted
    cp.cuda.Stream.null.synchronize()
    end = time.time()

    T = [t * dt for t in range(total_timesteps)]

    print(f"[simulate] GPU Total Time: {end - start:.6f}s")

    # Transfer result from GPU back to CPU
    Xs_cpu = cp.asnumpy(Xs)
    return T, Xs_cpu


if __name__ == "__main__":
    datasets = [
        ("TVB76",  data.tvb76_weights_lengths),
        ("TVB192", data.tvb192_weights_lengths),
        ("TVB998", data.tvb998_weights_lengths),
    ]

    M = 2       # state variables per node
    dt = 0.05   # timestep size in ms
    tf = 150.0  # simulation duration in ms
    speed = 4.0 # signal propagation speed in mm/ms

    for name, loader in datasets:
        print(f"\n{'='*40}")
        print(f"Running {name}  (tf={tf}ms, dt={dt}ms)")
        print(f"{'='*40}")
        W, D = loader()
        N = len(W)
        wall_start = time.time()
        T, Xs = simulate(W, D, N, M, dt, tf, speed)
        wall_end = time.time()
        print(f"[{name}] Wall-clock total (incl. transfers): {wall_end - wall_start:.6f}s")
