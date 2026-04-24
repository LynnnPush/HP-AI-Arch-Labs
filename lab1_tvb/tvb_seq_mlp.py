from lib import data, plot
from lib.mlp_params import MLP_L, MLP_M, layer_1_w, layer_1_b, layer_2_w, layer_2_b
import time
from typing import List


def pre(x_src: float, x_dst: float) -> float:
    """Pre-synaptic transform used before weighted summation."""
    return (x_src - 1.0)    # pre-synapse function


def post(gx: float) -> float:
    """Post-synaptic scaling applied after aggregation."""
    return (1e-3 * gx)               # post-synapse function


def f(x: float, y: float) -> tuple[float, float]:
    """Local MLP dynamics returning derivatives for x and y."""
    sv = [x, y]
    hidden = [0] * MLP_L
    out = [0] * MLP_M

    for l in range(MLP_L):
        for m in range(MLP_M):
            hidden[l] += sv[m] * layer_1_w[MLP_L*m + l]
        hidden[l] += layer_1_b[l]
        if(hidden[l] <= 0):
            hidden[l] = 0

    for m in range(MLP_M):
        for l in range(MLP_L):
            out[m] += hidden[l] * layer_2_w[MLP_M*l + m]
        out[m] += layer_2_b[m]

    return tuple(out)


def calculate_coupling(
    Xs: List[List[List[float]]],
    W_row: List[float],
    D_row: List[int],
    t: int,
    n: int,
) -> float:
    """Compute delayed coupling input for one destination node.

    Args:
        Xs: State history shaped as [N][M][T], where index 0 in M is x(t).
        W_row: Incoming connectivity weights for destination node n.
        D_row: Integer delay (in timesteps) for each source node into n.
        t: Current timestep index (t > 0 when this function is called).
        n: Destination node index.

    Returns:
        The post-synaptic coupling contribution c_in for node n at timestep t.
    """
    c_in = 0.0
    x_dst = Xs[n][0][t - 1]
    N = len(Xs)

    for i in range(N):
        x_src = 0.0
        if t >= D_row[i]:
            if (i == n) or (D_row[i] == 0):
                x_src = Xs[i][0][t - 1]
            else:
                x_src = Xs[i][0][t - D_row[i]]

        c_in = c_in + W_row[i] * pre(x_src, x_dst)

    return post(c_in)


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
) -> List[float]:
    """Advance one node state by one Forward Euler update.

    Args:
        Xs: State history shaped as [N][M][T].
        t: Current timestep index (uses values from t-1).
        n: Node index to update.
        c_in: Coupling input computed for node n at timestep t.
        dt: Simulation timestep size.

    Returns:
        Updated state vector [x_new, y_new] for node n at timestep t.
    """
    x = Xs[n][0][t-1]
    y = Xs[n][1][t-1]
    fx, fy = f(x, y)
    dx = dt * fx
    dy = dt * (fy + c_in)
    x_new = x + dx
    y_new = y + dy
    return [x_new, y_new]


def simulate(
    W: List[List[float]],
    D: List[List[float]],
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
    sparse_flag: bool,
) -> tuple[List[float], List[List[List[float]]]]:
    """Run the sequential MLP TVB simulation loop.

    Args:
        W: Connectivity weight matrix of shape [N][N].
        D: Distance matrix of shape [N][N] in physical distance units.
        N: Number of brain regions (nodes).
        M: Number of state variables per node.
        dt: Simulation timestep size.
        tf: Total simulation time.
        speed: Propagation speed used to convert distances to delays.
        sparse_flag: Whether to use sparse (CSR-based) coupling computation.

    Returns:
        A tuple (T, Xs):
        - T: Time axis list with length total_timesteps.
        - Xs: State history shaped as [N][M][T].
    """
    total_timesteps = int(tf/dt) # total number of timesteps the simulation must run
    Xs = [[[0.0 for _ in range(total_timesteps)] for _ in range(M)] for _ in range(N)] # state variable list (M list of total_timesteps values for each N centers)
    D_timestep = [[int((D[i][j] / speed) / dt) for i in range(N)] for j in range(N)] # delay list in terms of timesteps

    c_duration = 0
    f_duration = 0

    if sparse_flag:
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
                    f_start = time.time()
                    X_new = step(Xs, t, n, c_in, dt)
                    f_duration += (time.time() - f_start)
                    for m in range(M):
                        Xs[n][m][t] = X_new[m]
        end = time.time()

    else:
        start = time.time()
        for t in range(total_timesteps):
            if t == 0:
                for n in range(N):
                    for m in range(M):
                        Xs[n][m][t] = -1.0
            else:
                for n in range(N):
                    c_start = time.time()
                    c_in = calculate_coupling(Xs, W[n], D_timestep[n], t, n)
                    c_duration += (time.time() - c_start)
                    f_start = time.time()
                    X_new = step(Xs, t, n, c_in, dt)
                    f_duration += (time.time() - f_start)
                    for m in range(M):
                        Xs[n][m][t] = X_new[m]
        end = time.time()

    T = [t * dt for t in range(total_timesteps)]

    total_duration = end - start
    mode = "sparse" if sparse_flag else "dense"
    print(f"[simulate] Mode: {mode}")
    print(f"[simulate] Coupling Time: {c_duration:.6f}s")
    print(f"[simulate] Step Time: {f_duration:.6f}s")
    print(f"[simulate] Total Time: {total_duration:.6f}s")

    return T, Xs

if __name__ == "__main__":
    dt = 0.05
    tf = 15.0
    speed = 4.0
    M = 2
    USE_SPARSE = False

    for label, loader in [
        ("TVB-76",  data.tvb76_weights_lengths),
        ("TVB-192", data.tvb192_weights_lengths),
        ("TVB-998", data.tvb998_weights_lengths),
    ]:
        print(f"\n=== {label} ===")
        W, D = loader()
        W = W.tolist()
        D = D.tolist()
        N = len(W)
        simulate(W, D, N, M, dt, tf, speed, USE_SPARSE)
