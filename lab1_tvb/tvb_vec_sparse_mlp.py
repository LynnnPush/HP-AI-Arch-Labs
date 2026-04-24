from lib import data, plot
from lib.mlp_params import layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np
import numpy as np
import time


def f_vec(x_all: np.ndarray, y_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MLP for all N nodes — identical to tvb_vec_mlp.py."""
    sv = np.stack([x_all, y_all], axis=1)
    hidden = sv @ layer_1_w_np + layer_1_b_np
    hidden = np.maximum(hidden, 0)
    out = hidden @ layer_2_w_np + layer_2_b_np
    return out[:, 0], out[:, 1]


def calculate_coupling_sparse_vec(
    Xs: np.ndarray,
    W_arr: np.ndarray,
    D_arr: np.ndarray,
    row_arr: np.ndarray,
    col_arr: np.ndarray,
    same_or_zero: np.ndarray,
    t: int,
    N: int,
) -> np.ndarray:
    """Vectorized sparse coupling for all N destinations simultaneously.

    Uses COO format (row_arr, col_arr, W_arr, D_arr) instead of CSR.
    Each array has length nnz (number of non-zero weights).

    The key difference from the dense vectorized version (tvb_vec_mlp.py):
    all arrays here have size nnz instead of N×N, so we skip zero-weight
    entries entirely. The scatter-add (np.bincount) replaces the N×N
    row-wise sum.

    Args:
        Xs:           State history, shape (N, M, T).
        W_arr:        Non-zero weights, shape (nnz,).
        D_arr:        Delays in timesteps for each non-zero entry, shape (nnz,).
        row_arr:      Destination node index for each entry, shape (nnz,).
        col_arr:      Source node index for each entry, shape (nnz,).
        same_or_zero: Precomputed mask: True where col==row or delay==0, shape (nnz,).
        t:            Current timestep (>= 1).
        N:            Total number of nodes (needed for bincount output length).

    Returns:
        c_in_all: Coupling input for every node, shape (N,).
    """
    # Compute delayed time index per non-zero entry.
    # same_or_zero mirrors the sequential: (col_index[i] == n) or (D_sparse[i] == 0)
    t_idx = np.where(same_or_zero, t - 1, t - D_arr)  # (nnz,)
    t_idx = np.maximum(t_idx, 0)  # clamp so negative indices don't wrap

    # Gather delayed source state for each non-zero (dst, src) pair
    x_src_raw = Xs[col_arr, 0, t_idx]  # (nnz,) fancy index on src and time
    # Zero out entries where the signal delay has not been reached yet
    x_src = np.where(t >= D_arr, x_src_raw, 0.0)  # (nnz,)

    # pre(x_src, x_dst) = x_src - 1.0
    contribs = W_arr * (x_src - 1.0)  # (nnz,) weighted pre-synaptic values

    # Scatter-add contributions to their destination nodes.
    # np.bincount sums contribs[k] into bucket row_arr[k] for all k.
    # This replaces both the CSR row loop and the dense axis=1 sum.
    c_in = np.bincount(row_arr, weights=contribs, minlength=N)  # (N,)
    return 1e-3 * c_in  # post() scaling


def simulate(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the vectorized sparse MLP TVB simulation."""
    total_timesteps = int(tf / dt)
    Xs = np.zeros((N, M, total_timesteps))

    # Delay matrix [dst, src]: D_timestep[n, i] = delay from src i to dst n
    D_timestep = (D.T / speed / dt).astype(int)  # (N, N)

    # --- Convert dense W to COO sparse format (done once before the time loop) ---
    # Only keep entries where weight != 0, mirroring the sequential CSR build.
    mask = W != 0  # (N, N) bool
    row_arr = np.where(mask)[0]          # destination indices, shape (nnz,)
    col_arr = np.where(mask)[1]          # source indices,      shape (nnz,)
    W_arr   = W[row_arr, col_arr]        # non-zero weights,    shape (nnz,)
    D_arr   = D_timestep[row_arr, col_arr]  # matching delays,  shape (nnz,)

    nnz = len(W_arr)
    sparsity = 1.0 - nnz / (N * N)
    print(f"  nnz={nnz}, N²={N*N}, sparsity={sparsity:.1%}")

    # Precompute the condition mask: same node or zero delay (constant every step)
    same_or_zero = (col_arr == row_arr) | (D_arr == 0)  # (nnz,)

    c_duration = 0.0
    f_duration = 0.0

    start = time.time()

    Xs[:, :, 0] = -1.0  # initial condition

    for t in range(1, total_timesteps):
        # --- Sparse vectorized coupling (nnz operations instead of N²) ---
        c_start = time.time()
        c_in_all = calculate_coupling_sparse_vec(
            Xs, W_arr, D_arr, row_arr, col_arr, same_or_zero, t, N
        )
        c_duration += time.time() - c_start

        # --- Vectorized MLP step (same as dense version) ---
        f_start = time.time()
        x = Xs[:, 0, t - 1]
        y = Xs[:, 1, t - 1]
        fx, fy = f_vec(x, y)
        Xs[:, 0, t] = x + dt * fx
        Xs[:, 1, t] = y + dt * (fy + c_in_all)
        f_duration += time.time() - f_start

    end = time.time()
    total_duration = end - start

    print(f"[simulate] Mode: vectorized-sparse (N={N})")
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
