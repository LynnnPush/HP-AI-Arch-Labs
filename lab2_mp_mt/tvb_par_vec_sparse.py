from lib import data
from concurrent.futures import ProcessPoolExecutor
from lib.mlp_params import layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np
import numpy as np
import time
from multiprocessing import shared_memory

# Worker-local globals: attached once via init_worker, kept alive by shm handles
global Xs_g, W_g, D_g, col_g, soz_g, rptr_g
global _Xs_shm, _W_shm, _D_shm, _col_shm, _soz_shm, _rptr_shm

def init_worker(
    Xs_name,   Xs_shape,   Xs_dtype,
    W_name,    W_shape,    W_dtype,
    D_name,    D_shape,    D_dtype,
    col_name,  col_shape,  col_dtype,
    soz_name,  soz_shape,  soz_dtype,
    rptr_name, rptr_shape, rptr_dtype,
):
    """Open all shared memory blocks once per worker process."""
    global Xs_g, W_g, D_g, col_g, soz_g, rptr_g
    global _Xs_shm, _W_shm, _D_shm, _col_shm, _soz_shm, _rptr_shm

    _Xs_shm   = shared_memory.SharedMemory(name=Xs_name)
    _W_shm    = shared_memory.SharedMemory(name=W_name)
    _D_shm    = shared_memory.SharedMemory(name=D_name)
    _col_shm  = shared_memory.SharedMemory(name=col_name)
    _soz_shm  = shared_memory.SharedMemory(name=soz_name)
    _rptr_shm = shared_memory.SharedMemory(name=rptr_name)

    Xs_g   = np.ndarray(Xs_shape,   dtype=Xs_dtype,   buffer=_Xs_shm.buf)
    W_g    = np.ndarray(W_shape,    dtype=W_dtype,    buffer=_W_shm.buf)
    D_g    = np.ndarray(D_shape,    dtype=D_dtype,    buffer=_D_shm.buf)
    col_g  = np.ndarray(col_shape,  dtype=col_dtype,  buffer=_col_shm.buf)
    soz_g  = np.ndarray(soz_shape,  dtype=soz_dtype,  buffer=_soz_shm.buf)
    rptr_g = np.ndarray(rptr_shape, dtype=rptr_dtype, buffer=_rptr_shm.buf)


def center_task(t: int, n: int, dt: float) -> list[float]:
    """Forward-Euler step for node n at timestep t.

    Coupling is vectorized over the node's sparse neighbors (CSR slice).
    MLP dynamics use numpy matrix multiply instead of Python loops.
    Only t, n, dt cross the IPC boundary — all arrays live in shared memory.
    """
    global Xs_g, W_g, D_g, col_g, soz_g, rptr_g

    # CSR slice: rows rptr_g[n]..rptr_g[n+1] are node n's non-zero connections
    start = int(rptr_g[n])
    end   = int(rptr_g[n + 1])

    if start < end:
        W_n   = W_g[start:end]    # non-zero weights for destinations of n
        D_n   = D_g[start:end]    # delays (timesteps) for each entry
        col_n = col_g[start:end]  # source node indices
        soz_n = soz_g[start:end]  # same_or_zero mask (col==n or delay==0)

        # Vectorized delayed-index selection (mirrors sequential per-neighbor if)
        t_idx = np.where(soz_n, t - 1, t - D_n)
        t_idx = np.maximum(t_idx, 0)              # clamp against negative wrap

        x_src_raw = Xs_g[col_n, 0, t_idx]         # gather delayed source states
        x_src     = np.where(t >= D_n, x_src_raw, 0.0)  # mask unmatured delays

        # pre(x_src, x_dst) = x_src - 1.0;  post(gx) = 1e-3 * gx
        c_in = 1e-3 * np.dot(W_n, x_src - 1.0)   # vectorized weighted sum
    else:
        c_in = 0.0

    # Vectorized MLP for a single node (numpy dot products, no Python loops)
    x = float(Xs_g[n, 0, t - 1])
    y = float(Xs_g[n, 1, t - 1])
    sv     = np.array([x, y])
    hidden = sv @ layer_1_w_np + layer_1_b_np     # (MLP_L,)
    hidden = np.maximum(hidden, 0)                 # ReLU
    out    = hidden @ layer_2_w_np + layer_2_b_np  # (MLP_M,)

    return [x + dt * float(out[0]), y + dt * (float(out[1]) + c_in)]


def simulate(W: np.ndarray, D: np.ndarray, N: int, M: int,
             dt: float, tf: float, speed: float, chunksize: int = 1):
    """Run TVB simulation with vectorized sparse coupling and shared memory.

    Each center's timestep is handled by one process (one task per center per t).
    Within each task, coupling iterates only over non-zero edges (COO/CSR sparse)
    and uses numpy vectorized ops. MLP uses matrix multiply instead of loops.
    Shared memory is attached once per worker via init_worker (no per-task syscall).
    """
    total_timesteps = int(tf / dt)
    Xs = np.zeros((N, M, total_timesteps))

    # D_timestep[dst, src] = delay in timesteps from src to dst
    D_timestep = (D.T / speed / dt).astype(np.int32)

    # COO sparse structure: keep only non-zero weight entries
    mask    = W != 0
    row_arr = np.where(mask)[0].astype(np.int32)   # destination indices (nnz,)
    col_arr = np.where(mask)[1].astype(np.int32)   # source indices       (nnz,)
    W_arr   = W[row_arr, col_arr].astype(np.float64)
    D_arr   = D_timestep[row_arr, col_arr]

    nnz = len(W_arr)
    print(f"  nnz={nnz}, sparsity={1 - nnz/(N*N):.1%}")

    # same_or_zero: use t-1 index (no delay shift) when col==row or delay==0
    soz_arr = ((col_arr == row_arr) | (D_arr == 0))

    # CSR row pointer from sorted row_arr (np.where output is row-major)
    rptr = np.searchsorted(row_arr, np.arange(N + 1)).astype(np.int32)

    # Allocate shared memory for all arrays
    def _make_shm(arr: np.ndarray):
        shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(view, arr)
        return shm, view

    Xs_shm,   Xs_sh   = _make_shm(Xs)
    W_shm,    W_sh    = _make_shm(W_arr)
    D_shm,    D_sh    = _make_shm(D_arr)
    col_shm,  col_sh  = _make_shm(col_arr)
    soz_shm,  soz_sh  = _make_shm(soz_arr)
    rptr_shm, rptr_sh = _make_shm(rptr)

    Xs_sh[:, :, 0] = -1.0  # initial condition

    initargs = (
        Xs_shm.name,   Xs_sh.shape,   Xs_sh.dtype,
        W_shm.name,    W_sh.shape,    W_sh.dtype,
        D_shm.name,    D_sh.shape,    D_sh.dtype,
        col_shm.name,  col_sh.shape,  col_sh.dtype,
        soz_shm.name,  soz_sh.shape,  soz_sh.dtype,
        rptr_shm.name, rptr_sh.shape, rptr_sh.dtype,
    )

    start = time.time()
    with ProcessPoolExecutor(initializer=init_worker, initargs=initargs) as executor:
        for t in range(1, total_timesteps):
            result = list(executor.map(center_task, [t]*N, range(N), [dt]*N,
                                       chunksize=chunksize))
            for n in range(N):
                Xs_sh[n, 0, t] = result[n][0]
                Xs_sh[n, 1, t] = result[n][1]
    end = time.time()

    np.copyto(Xs, Xs_sh)

    for shm in (Xs_shm, W_shm, D_shm, col_shm, soz_shm, rptr_shm):
        shm.close()
        shm.unlink()

    T = np.arange(total_timesteps) * dt
    print(end - start)
    return T, Xs, end - start


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = 2               # number of state variables per center
    dt = 0.05           # timestep size for the simulation in ms
    tf = 15.0           # final timestep of the simulation in ms
    speed = 4.0         # signal speed in mm/ms

    # Optimal chunk sizes from earlier sweep (tvb_par.py)
    DATASETS = [
        ("TVB76",  data.tvb76_weights_lengths,  57),
        ("TVB192", data.tvb192_weights_lengths, 135),
        ("TVB998", data.tvb998_weights_lengths, 125),
    ]

    # Known timings from prior runs at tf=15ms
    seq_times      = {"TVB76": 0.78,  "TVB192": 2.65,  "TVB998": 45.34}
    par_sm_imp_times = {"TVB76": 1.51, "TVB192": 5.74,  "TVB998": 52.17}

    vec_sparse_times = {}

    for label, loader, best_cs in DATASETS:
        W, D = loader()
        N = len(W)
        print(f"\n=== {label} (N={N}) ===")
        _, _, t_vs = simulate(W, D, N, M, dt, tf, speed, chunksize=best_cs)
        vec_sparse_times[label] = t_vs
        print(f"  par_vec_sparse (cs={best_cs}): {t_vs:.2f}s")

    # --- comparison table ---
    print("\n{:<8} {:>12} {:>14} {:>16}".format(
        "Dataset", "Sequential", "Par (SM imp)", "Par (vec+sparse)"))
    print("-" * 54)
    for label in ["TVB76", "TVB192", "TVB998"]:
        seq = f"{seq_times[label]:.2f}s"
        imp = f"{par_sm_imp_times[label]:.2f}s"
        vs  = f"{vec_sparse_times[label]:.2f}s"
        print(f"{label:<8} {seq:>12} {imp:>14} {vs:>16}")

    # --- bar chart ---
    labels = ["TVB76", "TVB192", "TVB998"]
    x      = np.arange(len(labels))
    width  = 0.25

    seq_vals = [seq_times[l]        for l in labels]
    imp_vals = [par_sm_imp_times[l] for l in labels]
    vs_vals  = [vec_sparse_times[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, seq_vals, width, label='Sequential',         color='#7f7f7f')
    b2 = ax.bar(x,         imp_vals, width, label='Par (SM imp)',        color='#d62728')
    b3 = ax.bar(x + width, vs_vals,  width, label='Par (vec+sparse SM)', color='#9467bd')

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.1f}s", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title(f"TVB: Sequential vs Par(SM imp) vs Par(vec+sparse) (tf={tf}ms)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "comparison_vec_sparse.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {save_path}")
