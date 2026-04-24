from lib import data, plot
from lib.mlp_params import MLP_L, MLP_M, mlp_params
import time
import numpy as np
# Numba JIT: compiles Python+NumPy to machine code on first call.
from numba import njit


@njit(cache=True)
def pre(x_src: float, x_dst: float) -> float:
    """Pre-synaptic transform used before weighted summation."""
    return x_src - 1.0


@njit(cache=True)
def post(gx: float) -> float:
    """Post-synaptic scaling applied after aggregation."""
    return 1e-3 * gx


@njit(cache=True)
def f(x: float, y: float, params: np.ndarray) -> tuple[float, float]:
    """Local MLP dynamics returning derivatives for x and y."""
    # Replace Python lists with NumPy arrays so Numba can type them.
    sv = np.empty(2)
    sv[0] = x
    sv[1] = y
    hidden = np.zeros(MLP_L)
    out = np.zeros(MLP_M)

    l1_w = params[0:(MLP_L * MLP_M)]
    l1_b = params[(MLP_L * MLP_M):(MLP_L * MLP_M) + MLP_L]
    l2_w = params[(MLP_L * MLP_M) + MLP_L:(2 * MLP_L * MLP_M) + MLP_L]
    l2_b = params[(2 * MLP_L * MLP_M) + MLP_L:]

    for l in range(MLP_L):
        for m in range(MLP_M):
            hidden[l] += sv[m] * l1_w[MLP_L*m + l]
        hidden[l] += l1_b[l]
        if(hidden[l] <= 0):
            hidden[l] = 0

    for m in range(MLP_M):
        for l in range(MLP_L):
            out[m] += hidden[l] * l2_w[MLP_M*l + m]
        out[m] += l2_b[m]

    return out[0], out[1]
    # return 1.0 * (x - (x**3) / 3.0 + y) * 3.0, 1.0 * (1.01 - x) / 3.0


@njit(cache=True)
def calculate_coupling(
    Xs: np.ndarray,
    W_row: np.ndarray,
    D_row: np.ndarray,
    t: int,
    n: int,
) -> float:
    """Compute delayed coupling input for one destination node.

    Args:
        Xs: State history array shaped as [N, M, T], with x(t) at index 0 in M.
        W_row: Incoming connectivity weights for destination node n.
        D_row: Integer delay (in timesteps) for each source node into n.
        t: Current timestep index (t > 0 when this function is called).
        n: Destination node index.

    Returns:
        The post-synaptic coupling contribution c_in for node n at timestep t.
    """
    c_in = 0.0
    x_dst = Xs[n, 0, t - 1]
    N = len(Xs)

    for i in range(N):
        x_src = 0.0
        if t >= D_row[i]:
            if (i == n) or (D_row[i] == 0):
                x_src = Xs[i, 0, t - 1]
            else:
                x_src = Xs[i, 0, t - D_row[i]]

        c_in = c_in + W_row[i] * pre(x_src, x_dst)

    return post(c_in)


@njit(cache=True)
def step(
    Xs: np.ndarray,
    t: int,
    n: int,
    c_in: float,
    dt: float,
    params: np.ndarray,
) -> tuple[float, float]:
    """Advance one node state by one Forward Euler update.

    Args:
        Xs: State history array shaped as [N, M, T].
        t: Current timestep index (uses values from t-1).
        n: Node index to update.
        c_in: Coupling input computed for node n at timestep t.
        dt: Simulation timestep size.
        params: Flattened MLP parameters used by local dynamics f(x, y, params).

    Returns:
        Updated scalar state values (x_new, y_new) for node n at timestep t.
    """
    x = Xs[n, 0, t-1]
    y = Xs[n, 1, t-1]
    fx, fy = f(x, y, params)
    dx = dt * fx
    dy = dt * (fy + c_in)
    x_new = x + dx
    y_new = y + dy
    return x_new, y_new


@njit(cache=True)
def simulate(
    W: np.ndarray,
    D_timestep: np.ndarray,
    N: int,
    M: int,
    dt: float,
    total_timesteps: int,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the sequential JIT-compiled TVB simulation loop.

    Args:
        W: Connectivity weight matrix of shape [N, N].
        D_timestep: Integer delay matrix (in timesteps) of shape [N, N],
            passed explicitly so Numba gets a typed arg instead of a
            module-level global.
        N: Number of brain regions (nodes).
        M: Number of state variables per node.
        dt: Simulation timestep size.
        total_timesteps: Number of simulation timesteps to execute.
        params: Flattened MLP parameter vector used in local dynamics.

    Returns:
        A tuple (T, Xs):
        - T: Time axis as a NumPy array of length total_timesteps.
        - Xs: State history shaped as [N, M, T].
    """
    Xs = np.zeros((N, M, total_timesteps)) # state variable list (M list of total_timesteps values for each N centers)

    for t in range(total_timesteps):
        if t == 0:
            for n in range(N):
                for m in range(M):
                    Xs[n, m, t] = -1.0
        else:
            for n in range(N):
                c_in = calculate_coupling(Xs, W[n], D_timestep[n], t, n)
                Xs[n, 0, t], Xs[n, 1, t] = step(Xs, t, n, c_in, dt, params)

    # Typed array instead of a Python list — Numba-friendly and equivalent.
    T = np.arange(total_timesteps) * dt

    return T, Xs

if __name__ == "__main__":
    W, D = data.tvb192_weights_lengths()
    N = len(W)          # number of centers
    M = 2               # number of state variables per center

    dt = 0.05           # timestep size for the simulation in ms
    tf = 15.0           # final timestep of the simulation in ms
    speed = 4.0         # signal speed in mm/ms
    freq = 1.0          # frequency parameter for the local dynamics

    total_timesteps = int(tf/dt)
    D_timestep = ((D / speed) / dt).astype(np.int64) # delay list in terms of timesteps

    # Warm-up call: triggers JIT compilation so it is NOT counted in the
    # measured runtime below. Uses tiny total_timesteps to keep it cheap.
    simulate(W, D_timestep, N, M, dt, 2, mlp_params)

    start = time.time()
    T, Xs = simulate(W, D_timestep, N, M, dt, total_timesteps, mlp_params)
    end = time.time()
    total_duration = end - start
    print(f"[simulate_jit] Total Time: {total_duration:.6f}s")
    plot.plot_xs(T, Xs, speed)
    # plot.plot_delay_hist(D, W, speed)
