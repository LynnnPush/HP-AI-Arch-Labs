import time
import numpy as np
from typing import List

from lib import data, plot
from lib.mlp_params import layer_1_b_np, layer_1_w_np, layer_2_b_np, layer_2_w_np


def pre(x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Pre-synaptic transform used before weighted summation."""
    return (x_src - 1.0)


def post(gx: np.ndarray) -> np.ndarray:
    """Post-synaptic scaling applied after aggregation."""
    return (1e-3 * gx)


def f(X: np.ndarray) -> np.ndarray:
    """Vectorized MLP dynamics for all node state vectors.

    Args:
        X: State array shaped as [N, M] for all nodes simultaneously.
    """
    hidden = np.matmul(X, layer_1_w_np) + layer_1_b_np
    hidden = np.where(hidden <= 0, 0, hidden)
    out = np.matmul(hidden, layer_2_w_np) + layer_2_b_np
    return out


def calculate_coupling(
    Xs: np.ndarray,
    W: np.ndarray,
    D_timestep: np.ndarray,
    t: int,
) -> np.ndarray:
    """Compute delayed coupling for all destination nodes simultaneously.

    Args:
        Xs: State history shaped as [N, M, T].
        W: Full connectivity weight matrix [N, N].
        D_timestep: Integer delay matrix [N, N] in timesteps.
        t: Current timestep index (t > 0 when this function is called).

    Returns:
        c_ins: Post-synaptic coupling contributions shaped as [N].
    """
    N = len(Xs)
    # valid_time[n, src]: delay is within simulation window
    valid_time = (t >= D_timestep)                                    # [N, N]
    # self-connections and zero-delay connections use t-1
    use_prev = np.eye(N, dtype=bool) | (D_timestep == 0)             # [N, N]
    timesteps_indices = np.where(use_prev, t - 1, t - D_timestep)    # [N, N]
    safe_indices = np.where(valid_time, timesteps_indices, 0)         # [N, N]

    # x_src[n, src] = Xs[src, 0, safe_indices[n, src]]
    src_idx = np.arange(N)[np.newaxis, :]                             # [1, N]
    x_src = Xs[src_idx, 0, safe_indices]                              # [N, N]
    x_src = np.where(valid_time, x_src, 0.0)                         # [N, N]

    # x_dst[n, :] = Xs[n, 0, t-1] broadcast over all sources
    x_dst = Xs[:, 0, t - 1][:, np.newaxis]                           # [N, 1]

    c_in = np.sum(W * pre(x_src, x_dst), axis=1)                     # [N]
    return post(c_in)                                                  # [N]


def step(Xs: np.ndarray, t: int, c_ins: np.ndarray, dt: float) -> np.ndarray:
    """Advance all node states by one Forward Euler update.

    Args:
        Xs: State history shaped as [N, M, T].
        t: Current timestep index.
        c_ins: Coupling inputs shaped as [N].
        dt: Simulation timestep size.

    Returns:
        X_new: Updated states shaped as [N, M].
    """
    X_all = Xs[:, :, t - 1]    # [N, M]
    fx = f(X_all)               # [N, M]
    fx[:, 1] += c_ins           # add coupling to second state variable only
    return X_all + fx * dt      # [N, M]


def simulate(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
) -> tuple[List[float], np.ndarray]:
    """Run the vectorized TVB simulation loop.

    Args:
        W: Connectivity weight matrix of shape [N, N].
        D: Distance matrix of shape [N, N] in physical distance units.
        N: Number of brain regions (nodes).
        M: Number of state variables per node.
        dt: Simulation timestep size.
        tf: Total simulation time.
        speed: Propagation speed used to convert distances to delays.

    Returns:
        A tuple (T, Xs):
        - T: Time axis list with length total_timesteps.
        - Xs: State history shaped as [N, M, T].
    """
    total_timesteps = int(tf/dt)
    Xs = np.zeros((N, M, total_timesteps))
    D_timestep = ((D / speed) / dt).astype(int)

    c_duration = 0
    f_duration = 0

    start = time.time()
    for t in range(total_timesteps):
        if t == 0:
            Xs[:, :, t] = -1.0
        else:
            c_start = time.time()
            c_ins = calculate_coupling(Xs, W, D_timestep, t)   # [N]
            c_duration += time.time() - c_start

            f_start = time.time()
            Xs[:, :, t] = step(Xs, t, c_ins, dt)               # [N, M]
            f_duration += time.time() - f_start
    end = time.time()

    T = [t * dt for t in range(total_timesteps)]

    total_duration = end - start
    print(f"[simulate] Coupling Time: {c_duration:.6f}s")
    print(f"[simulate] Step Time: {f_duration:.6f}s")
    print(f"[simulate] Total Time: {total_duration:.6f}s")

    return T, Xs

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
        print(f"[{name}] Wall-clock total: {wall_end - wall_start:.6f}s")
