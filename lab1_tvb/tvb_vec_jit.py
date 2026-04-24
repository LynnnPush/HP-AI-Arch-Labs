from lib import data, plot
from lib.mlp_params import layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np
import numpy as np
import time
# Numba JIT: compiles NumPy-heavy code to machine code on first call.
from numba import njit


@njit(cache=True)
def f_vec(
    x_all: np.ndarray,
    y_all: np.ndarray,
    l1_w: np.ndarray,
    l1_b: np.ndarray,
    l2_w: np.ndarray,
    l2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MLP local dynamics for all N nodes at once.

    MLP weights are passed as arguments rather than read from module globals
    so Numba gets a clean typed signature (same reason as D_timestep in the
    sequential JIT version).
    """
    N = x_all.shape[0]
    # Build the (N, 2) input matrix without np.stack (cheaper / simpler for Numba).
    sv = np.empty((N, 2))
    sv[:, 0] = x_all
    sv[:, 1] = y_all

    # Layer 1: linear + ReLU. `@` maps to np.dot which Numba compiles well.
    hidden = sv @ l1_w + l1_b
    hidden = np.maximum(hidden, 0.0)

    # Layer 2: linear only.
    out = hidden @ l2_w + l2_b
    return out[:, 0].copy(), out[:, 1].copy()


@njit(cache=True)
def calculate_coupling_vec(
    Xs: np.ndarray,
    W: np.ndarray,
    D_timestep: np.ndarray,
    i_idx: np.ndarray,
    same_or_zero: np.ndarray,
    t: int,
) -> np.ndarray:
    """Vectorized delayed coupling for all N destinations at once."""
    # Pick t-1 for self/zero-delay edges, t-D otherwise.
    t_idx = np.where(same_or_zero, t - 1, t - D_timestep)
    t_idx = np.maximum(t_idx, 0)  # clamp so negatives don't wrap; masked below.

    # Numba-friendly gather: explicit loop replaces Xs[i_idx, 0, t_idx]
    # fancy indexing, which has partial/fragile support in nopython mode.
    N = Xs.shape[0]
    x_src_raw = np.empty((N, N))
    for n in range(N):
        for i in range(N):
            x_src_raw[n, i] = Xs[i_idx[n, i], 0, t_idx[n, i]]

    # Zero out entries whose delay hasn't elapsed yet (t < D), matching the
    # x_src = 0.0 default in the scalar version.
    x_src = np.where(t >= D_timestep, x_src_raw, 0.0)

    pre_vals = x_src - 1.0          # pre(x_src, x_dst) = x_src - 1.0
    c_in = np.sum(W * pre_vals, axis=1)  # weighted sum per destination
    return 1e-3 * c_in              # post() = 1e-3 * gx


@njit(cache=True)
def simulate(
    W: np.ndarray,
    D_timestep: np.ndarray,
    i_idx: np.ndarray,
    same_or_zero: np.ndarray,
    N: int,
    M: int,
    dt: float,
    total_timesteps: int,
    l1_w: np.ndarray,
    l1_b: np.ndarray,
    l2_w: np.ndarray,
    l2_b: np.ndarray,
) -> np.ndarray:
    """JIT-compiled vectorized TVB loop.

    Precomputed D_timestep / i_idx / same_or_zero are passed in so they are
    not rebuilt inside the compiled hot loop, and so Numba doesn't need to
    reach for module globals.
    """
    Xs = np.zeros((N, M, total_timesteps))
    Xs[:, :, 0] = -1.0  # initial condition

    for t in range(1, total_timesteps):
        c_in_all = calculate_coupling_vec(Xs, W, D_timestep, i_idx, same_or_zero, t)
        x = Xs[:, 0, t - 1]
        y = Xs[:, 1, t - 1]
        fx, fy = f_vec(x, y, l1_w, l1_b, l2_w, l2_b)
        Xs[:, 0, t] = x + dt * fx
        Xs[:, 1, t] = y + dt * (fy + c_in_all)  # coupling enters the y equation

    return Xs


if __name__ == "__main__":
    dt = 0.05
    tf = 15.0
    speed = 4.0
    M = 2

    # Warm-up on a tiny problem: triggers JIT compilation outside the timed
    # region so measured times reflect steady-state execution only.
    W_warm, D_warm = data.tvb76_weights_lengths()
    N_warm = len(W_warm)
    D_ts_warm = (D_warm.T / speed / dt).astype(np.int64)
    i_idx_warm = np.broadcast_to(np.arange(N_warm)[None, :], (N_warm, N_warm)).copy()
    soz_warm = np.eye(N_warm, dtype=np.bool_) | (D_ts_warm == 0)
    simulate(W_warm, D_ts_warm, i_idx_warm, soz_warm, N_warm, M, dt, 2,
             layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np)

    for label, loader in [
        ("TVB-76",  data.tvb76_weights_lengths),
        ("TVB-192", data.tvb192_weights_lengths),
        ("TVB-998", data.tvb998_weights_lengths),
    ]:
        print(f"\n=== {label} ===")
        W, D = loader()
        N = len(W)
        total_timesteps = int(tf / dt)

        # Precompute once per dataset (these don't change across timesteps).
        D_timestep = (D.T / speed / dt).astype(np.int64)
        i_idx = np.broadcast_to(np.arange(N)[None, :], (N, N)).copy()
        same_or_zero = np.eye(N, dtype=np.bool_) | (D_timestep == 0)

        start = time.time()
        Xs = simulate(W, D_timestep, i_idx, same_or_zero, N, M, dt,
                      total_timesteps,
                      layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np)
        end = time.time()
        print(f"[simulate_vec_jit] Total Time: {end - start:.6f}s")
