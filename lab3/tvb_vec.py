import time
import numpy as np
from typing import List

from lib import data, plot
from lib.mlp_params import layer_1_b_np, layer_1_w_np, layer_2_b_np, layer_2_w_np


def pre(x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Pre-synaptic transform used before weighted summation."""
    return (x_src - 1.0)


def post(gx: float) -> float:
    """Post-synaptic scaling applied after aggregation."""
    return (1e-3 * gx)


def f(X: np.ndarray) -> np.ndarray:
    """Local vectorized MLP dynamics for one node state vector."""
    hidden = np.matmul(X, layer_1_w_np) + layer_1_b_np
    hidden = np.where(hidden <= 0, 0, hidden)
    out = np.matmul(hidden, layer_2_w_np) + layer_2_b_np
    return out


def calculate_coupling(
    Xs: np.ndarray,
    W_row: np.ndarray,
    D_row: np.ndarray,
    t: int,
    n: int,
) -> float:
    """Compute delayed coupling input for one destination node.

    Args:
        Xs: State history shaped as [N, M, T], where index 0 in M is x(t).
        W_row: Incoming connectivity weights for destination node n.
        D_row: Integer delay (in timesteps) for each source node into n.
        t: Current timestep index (t > 0 when this function is called).
        n: Destination node index.

    Returns:
        The post-synaptic coupling contribution c_in for node n at timestep t.
    """
    N = len(Xs)
    valid_time = (t >= D_row)
    timesteps_indices = np.where((np.arange(N) == n) | (D_row == 0), t - 1, t - D_row)
    safe_indices = np.where(valid_time, timesteps_indices, 0)

    x_src = Xs[np.arange(N), 0, safe_indices]
    x_src = np.where(valid_time, x_src, 0)
    x_dst = np.full((N), Xs[n, 0, t - 1])

    c_partial = W_row * pre(x_src, x_dst)
    c_in = np.sum(c_partial)

    return post(c_in)


def step(Xs: np.ndarray, t: int, n: int, c_in: float, dt: float) -> np.ndarray:
    """Advance one node state by one Forward Euler update."""
    X = Xs[n, :, t-1]
    fx = f(X)
    fx = fx + np.array([0, c_in])
    dx = fx * dt
    X_new = X + dx
    return X_new


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
    total_timesteps = int(tf/dt) # total number of timesteps the simulation must run
    Xs = np.zeros((N, M, total_timesteps))
    D_timestep = ((D / speed) / dt).astype(int)

    c_duration = 0
    f_duration = 0

    start = time.time()
    for t in range(total_timesteps):
        if t == 0:
            Xs[:, :, t] = -1.0
        else:
            for n in range(N):
                c_start = time.time()
                c_in = calculate_coupling(Xs, W[n], D_timestep[n], t, n)
                c_duration += (time.time() - c_start)
                f_start = time.time()
                X_new = step(Xs, t, n, c_in, dt)
                f_duration += (time.time() - f_start)
                Xs[n, :, t] = X_new
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
