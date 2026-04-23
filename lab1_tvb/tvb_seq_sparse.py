from lib import data, plot
import time
from typing import List


def pre(x_src: float, x_dst: float) -> float:
    """Pre-synaptic transform used before weighted summation."""
    return (x_src - 1.0)                                                # pre-synapse function


def post(gx: float) -> float:
    """Post-synaptic scaling applied after aggregation."""
    return (1e-3 * gx)                                                           # post-synapse function


def f(x: float, y: float, freq: float) -> tuple[float, float]:
    """Local node dynamics returning derivatives for x and y."""
    return (freq * (x - ((x**3)/3.0) + y) * 3.0, freq * (1.01 - x) / 3.0)   # returning (fx, fy)


def calculate_coupling_sparse(
    Xs: List[List[List[float]]],
    W_sparse: List[float],
    D_sparse: List[int],
    col_index: List[int],
    row_pointer: List[int],
    t: int,
    n: int,
) -> float:
    """Compute delayed coupling input for one destination node using CSR data.

    Args:
        Xs: State history shaped as [N][M][T], where index 0 in M is x(t).
        W_sparse: Non-zero connectivity values in CSR order.
        D_sparse: Delay values (in timesteps) aligned with W_sparse entries.
        col_index: Source-node column indices aligned with sparse entries.
        row_pointer: CSR row offsets, length N+1.
        t: Current timestep index (t > 0 when this function is called).
        n: Destination node index.

    Returns:
        The post-synaptic coupling contribution c_in for node n at timestep t.
    """
    c_in = 0.0
    x_dst = Xs[n][0][t - 1]

    for i in range(row_pointer[n], row_pointer[n+1]):
        x_src = 0.0
        if t >= D_sparse[i]:
            if (col_index[i] == n) or (D_sparse[i] == 0):
                x_src = Xs[col_index[i]][0][t - 1]
            else:
                x_src = Xs[col_index[i]][0][t - D_sparse[i]]

        c_in = c_in + W_sparse[i] * pre(x_src, x_dst)

    return post(c_in)

def step(
    Xs: List[List[List[float]]],
    t: int,
    n: int,
    c_in: float,
    dt: float,
    freq: float,
) -> List[float]:
    """Advance one node state by one Forward Euler update.

    Args:
        Xs: State history shaped as [N][M][T].
        t: Current timestep index (uses values from t-1).
        n: Node index to update.
        c_in: Coupling input computed for node n at timestep t.
        dt: Simulation timestep size.
        freq: Frequency scaling applied in local dynamics f(x, y, freq).

    Returns:
        Updated state vector [x_new, y_new] for node n at timestep t.
    """
    x = Xs[n][0][t-1]
    y = Xs[n][1][t-1]
    fx, fy = f(x, y, freq)
    dx = dt * fx
    dy = dt * (fy + c_in)
    x_new = x + dx
    y_new = y + dy
    return [x_new, y_new]

def simulate_sparse(
    W: List[List[float]],
    D: List[List[float]],
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
    freq: float,
) -> tuple[List[float], List[List[List[float]]]]:
    """Run the sequential sparse TVB simulation loop.

    Args:
        W: Connectivity weight matrix of shape [N][N].
        D: Distance matrix of shape [N][N] in physical distance units.
        N: Number of brain regions (nodes).
        M: Number of state variables per node.
        dt: Simulation timestep size.
        tf: Total simulation time.
        speed: Propagation speed used to convert distances to delays.
        freq: Frequency scaling parameter for local dynamics.

    Returns:
        A tuple (T, Xs):
        - T: Time axis list with length total_timesteps.
        - Xs: State history shaped as [N][M][T].
    """
    total_timesteps = int(tf/dt) # total number of timesteps the simulation must run
    Xs = [[[0.0 for _ in range(total_timesteps)] for _ in range(M)] for _ in range(N)] # state variable list (M list of total_timesteps values for each N centers)
    D_timestep = [[int((D[i][j] / speed) / dt) for i in range(N)] for j in range(N)] # delay list in terms of timesteps

    # pre-processing for sparse calculation
    W_sparse = []
    D_sparse = []
    col_index = []
    row_pointer = [0]

    for i in range(N):
        new_row_pointer = row_pointer[-1]
        for j in range(N):
            if not W[i][j] == 0:
                W_sparse.append(W[i][j])
                D_sparse.append(D_timestep[i][j])
                col_index.append(j)
                new_row_pointer += 1
        row_pointer.append(new_row_pointer)



    c_duration = 0

    start = time.time()
    for t in range(total_timesteps):
        if t == 0:
            for n in range(N):
                for m in range(M):
                    Xs[n][m][t] = -1.0
        else:
            for n in range(N):
                c_start = time.time()
                c_in = calculate_coupling_sparse(Xs, W_sparse, D_sparse, col_index, row_pointer, t, n)
                c_duration += (time.time() - c_start)
                X_new = step(Xs, t, n, c_in, dt, freq)
                for m in range(M):
                    Xs[n][m][t] = X_new[m]
    end = time.time()

    total_duration = end - start
    print(f"[simulate] Coupling Time: {c_duration:.6f}s")
    print(f"[simulate] Total Time: {total_duration:.6f}s")

    T = [t * dt for t in range(total_timesteps)]

    return T, Xs

if __name__ == "__main__":
    W, D = data.tvb192_weights_lengths()
    W = W.tolist()      # weight matrix
    D = D.tolist()      # distance matrix
    N = len(W)          # number of centers
    M = 2               # number of state variables per center


    dt = 0.05           # timestep size for the simulation in ms
    tf = 150.0          # final timestep of the simulation in ms
    speed = 4.0         # signal speed in mm/ms
    freq = 1.0          # frequency parameter for the local dynamics

    T, Xs = simulate_sparse(W, D, N, M, dt, tf, speed, freq)
    plot.plot_xs(T, Xs, speed)
    # plot.plot_delay_hist(D, W, speed)
