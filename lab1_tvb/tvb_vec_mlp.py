from lib import data, plot
from lib.mlp_params import layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np
import numpy as np
import time


def f_vec(x_all: np.ndarray, y_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MLP local dynamics for all N nodes at once.

    Replaces the scalar f(x, y) which looped over MLP_L and MLP_M.
    The two nested loops collapse into two matrix multiplications.
    """
    # Stack the two state variables into a (N, 2) input matrix
    sv = np.stack([x_all, y_all], axis=1)

    # Layer 1: linear transform then ReLU
    # layer_1_w_np is (2, 64), so sv @ layer_1_w_np gives (N, 64)
    hidden = sv @ layer_1_w_np + layer_1_b_np
    hidden = np.maximum(hidden, 0)  # ReLU: replaces the if hidden[l] <= 0 branch

    # Layer 2: linear transform only (no activation)
    # layer_2_w_np is (64, 2), so hidden @ layer_2_w_np gives (N, 2)
    out = hidden @ layer_2_w_np + layer_2_b_np

    return out[:, 0], out[:, 1]  # fx and fy, each (N,)


def calculate_coupling_vec(
    Xs: np.ndarray,
    W: np.ndarray,
    D_timestep: np.ndarray,
    i_idx: np.ndarray,
    same_or_zero: np.ndarray,
    t: int,
) -> np.ndarray:
    """Vectorized delayed coupling for all N destination nodes simultaneously.

    Replaces the scalar calculate_coupling() which looped over all N sources
    for a single destination node n.

    Args:
        Xs:           State history, shape (N, M, T).
        W:            Connectivity weight matrix, shape (N, N).
        D_timestep:   Delay matrix in timesteps, shape (N, N).
                      D_timestep[dst, src] = delay from src to dst.
        i_idx:        Precomputed source index grid, shape (N, N).
                      i_idx[n, i] = i, used for fancy gather indexing.
        same_or_zero: Precomputed boolean mask, shape (N, N).
                      True when the sequential code would use t-1 instead of t-delay.
        t:            Current timestep (>= 1).

    Returns:
        c_in_all: Coupling input for every node, shape (N,).
    """
    # Determine which past timestep to read for each (dst, src) pair.
    # same_or_zero covers: self-connection (i==n) or zero-delay edge.
    # np.where picks t-1 for those cases, t-D otherwise.
    t_idx = np.where(same_or_zero, t - 1, t - D_timestep)  # (N, N)

    # Clamp to 0 so negative indices (delay not yet elapsed) don't wrap around.
    # Those entries are masked out below anyway.
    t_idx = np.maximum(t_idx, 0)

    # Fancy gather: x_src_raw[n, i] = Xs[i, 0, t_idx[n, i]]
    # i_idx supplies the source-node axis; t_idx supplies the time axis.
    x_src_raw = Xs[i_idx, 0, t_idx]  # (N, N)

    # Zero out entries where the delay hasn't been reached yet (t < D),
    # mirroring the sequential x_src = 0.0 default before the t >= D_row[i] check.
    x_src = np.where(t >= D_timestep, x_src_raw, 0.0)  # (N, N)

    # pre(x_src, x_dst) = x_src - 1.0  (x_dst is not used by this pre function)
    pre_vals = x_src - 1.0  # (N, N)

    # Weighted sum over all sources for each destination, then post() scale
    c_in = np.sum(W * pre_vals, axis=1)  # (N,)
    return 1e-3 * c_in  # post() = 1e-3 * gx


def simulate(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the vectorized MLP TVB simulation (no-sparse, no-sparsity mode).

    Args:
        W:     Connectivity weight matrix, shape (N, N) — NumPy array.
        D:     Distance matrix, shape (N, N) — NumPy array.
        N:     Number of brain regions.
        M:     Number of state variables per node (= 2).
        dt:    Timestep size in ms.
        tf:    Total simulation time in ms.
        speed: Signal propagation speed in mm/ms.

    Returns:
        T:  Time axis, shape (total_timesteps,).
        Xs: State history, shape (N, M, total_timesteps).
    """
    total_timesteps = int(tf / dt)

    # (N, M, T) NumPy array replaces the [N][M][T] nested Python list.
    # Contiguous layout means slicing Xs[:, 0, t] is a cheap strided view.
    Xs = np.zeros((N, M, total_timesteps))

    # Delay matrix: D_timestep[dst, src] = int(D[src, dst] / speed / dt).
    # D[i, j] is the tract length from node i to node j, so transposing gives
    # us D[src, dst] indexed as [dst, src] — matching the sequential D_row[i].
    D_timestep = (D.T / speed / dt).astype(int)  # (N, N)

    # --- Precompute per-timestep constants (done once, reused every iteration) ---

    # i_idx[n, i] = i: the source-node index for fancy gather in coupling.
    # broadcast_to avoids allocating a full (N, N) copy.
    i_idx = np.broadcast_to(np.arange(N)[None, :], (N, N))

    # Mask for the "use t-1" branch in the sequential coupling code:
    # either the source is the destination itself, or the edge has zero delay.
    same_or_zero = np.eye(N, dtype=bool) | (D_timestep == 0)

    c_duration = 0.0
    f_duration = 0.0

    start = time.time()

    # t=0: initial condition — all state variables set to -1.0
    Xs[:, :, 0] = -1.0

    for t in range(1, total_timesteps):
        # --- Coupling: replaces the for n in range(N) loop + calculate_coupling ---
        c_start = time.time()
        c_in_all = calculate_coupling_vec(Xs, W, D_timestep, i_idx, same_or_zero, t)
        c_duration += time.time() - c_start

        # --- Local dynamics + Euler step: replaces step() + scalar f() ---
        f_start = time.time()
        x = Xs[:, 0, t - 1]   # (N,)
        y = Xs[:, 1, t - 1]   # (N,)
        fx, fy = f_vec(x, y)
        Xs[:, 0, t] = x + dt * fx
        Xs[:, 1, t] = y + dt * (fy + c_in_all)  # coupling enters the y equation
        f_duration += time.time() - f_start

    end = time.time()
    total_duration = end - start

    print(f"[simulate] Mode: vectorized (N={N})")
    print(f"[simulate] Coupling Time: {c_duration:.6f}s")
    print(f"[simulate] Step Time:     {f_duration:.6f}s")
    print(f"[simulate] Total Time:    {total_duration:.6f}s")

    T = np.arange(total_timesteps) * dt
    return T, Xs


if __name__ == "__main__":
    dt = 0.05
    tf = 15.0
    speed = 4.0
    M = 2

    for label, loader in [
        ("TVB-76",  data.tvb76_weights_lengths),
        ("TVB-192", data.tvb192_weights_lengths),
        ("TVB-998", data.tvb998_weights_lengths),
    ]:
        print(f"\n=== {label} ===")
        W, D = loader()
        N = len(W)
        T, Xs = simulate(W, D, N, M, dt, tf, speed)

    # Uncomment to visualise the last dataset run:
    # plot.plot_xs(T, Xs, speed)
