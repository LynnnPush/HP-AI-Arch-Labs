from lib import data
from concurrent.futures import ProcessPoolExecutor
from lib.mlp_params import layer_1_w_np, layer_1_b_np, layer_2_w_np, layer_2_b_np
import numpy as np
import time
from multiprocessing import shared_memory

# Worker-local globals: read-only sparse arrays, identical across all simulations
global W_g, D_g, row_g, col_g, soz_g
global _W_shm, _D_shm, _row_shm, _col_shm, _soz_shm

def init_worker(
    W_name,   W_shape,   W_dtype,
    D_name,   D_shape,   D_dtype,
    row_name, row_shape, row_dtype,
    col_name, col_shape, col_dtype,
    soz_name, soz_shape, soz_dtype,
):
    """Attach all read-only sparse arrays once per worker process."""
    global W_g, D_g, row_g, col_g, soz_g
    global _W_shm, _D_shm, _row_shm, _col_shm, _soz_shm

    _W_shm   = shared_memory.SharedMemory(name=W_name)
    _D_shm   = shared_memory.SharedMemory(name=D_name)
    _row_shm = shared_memory.SharedMemory(name=row_name)
    _col_shm = shared_memory.SharedMemory(name=col_name)
    _soz_shm = shared_memory.SharedMemory(name=soz_name)

    W_g   = np.ndarray(W_shape,   dtype=W_dtype,   buffer=_W_shm.buf)
    D_g   = np.ndarray(D_shape,   dtype=D_dtype,   buffer=_D_shm.buf)
    row_g = np.ndarray(row_shape, dtype=row_dtype, buffer=_row_shm.buf)
    col_g = np.ndarray(col_shape, dtype=col_dtype, buffer=_col_shm.buf)
    soz_g = np.ndarray(soz_shape, dtype=soz_dtype, buffer=_soz_shm.buf)


def run_simulation(sim_id: int, N: int, M: int, dt: float, tf: float, speed: float) -> int:
    """Run one complete simulation, fully vectorized and sparse.

    Reads W, D, row, col, soz from shared memory (zero-copy, no pickling).
    Xs is allocated locally — each simulation has an independent trajectory.
    """
    global W_g, D_g, row_g, col_g, soz_g

    total_timesteps = int(tf / dt)
    Xs = np.zeros((N, M, total_timesteps))
    Xs[:, :, 0] = -1.0

    for t in range(1, total_timesteps):
        # Vectorized sparse coupling over all N nodes simultaneously (COO format)
        t_idx     = np.where(soz_g, t - 1, t - D_g)         # delayed time index per edge
        t_idx     = np.maximum(t_idx, 0)                      # clamp against negative wrap
        x_src_raw = Xs[col_g, 0, t_idx]                       # gather source states
        x_src     = np.where(t >= D_g, x_src_raw, 0.0)        # mask unmatured delays
        contribs  = W_g * (x_src - 1.0)                       # weighted pre-synaptic values
        c_in_all  = 1e-3 * np.bincount(row_g, weights=contribs, minlength=N)  # scatter-add

        # Vectorized MLP for all N nodes (matrix multiply, no Python loops)
        x      = Xs[:, 0, t - 1]
        y      = Xs[:, 1, t - 1]
        sv     = np.stack([x, y], axis=1)                      # (N, 2)
        hidden = sv @ layer_1_w_np + layer_1_b_np              # (N, MLP_L)
        hidden = np.maximum(hidden, 0)                          # ReLU
        out    = hidden @ layer_2_w_np + layer_2_b_np          # (N, 2)

        Xs[:, 0, t] = x + dt * out[:, 0]
        Xs[:, 1, t] = y + dt * (out[:, 1] + c_in_all)

    return sim_id


def simulate_multi(W: np.ndarray, D: np.ndarray, N: int, M: int,
                   dt: float, tf: float, speed: float, n_sims: int):
    """Launch n_sims independent simulations in parallel across worker processes.

    W and D sparse arrays are shared read-only via shared memory (loaded once
    per worker via init_worker). Each simulation allocates its own local Xs.
    """
    total_timesteps = int(tf / dt)

    # Build COO sparse structure once; identical input for all simulations
    D_timestep = (D.T / speed / dt).astype(np.int32)   # D_timestep[dst, src]
    mask    = W != 0
    row_arr = np.where(mask)[0].astype(np.int32)        # destination indices (nnz,)
    col_arr = np.where(mask)[1].astype(np.int32)        # source indices       (nnz,)
    W_arr   = W[row_arr, col_arr].astype(np.float64)
    D_arr   = D_timestep[row_arr, col_arr]
    soz_arr = (col_arr == row_arr) | (D_arr == 0)       # same_or_zero mask

    nnz = len(W_arr)
    print(f"  nnz={nnz}, sparsity={1 - nnz/(N*N):.1%}")

    # Place read-only sparse arrays in shared memory — attached once per worker,
    # not once per task, eliminating repeated pickling across n_sims tasks
    def _make_shm(arr: np.ndarray):
        shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(view, arr)
        return shm, view

    W_shm,   W_sh   = _make_shm(W_arr)
    D_shm,   D_sh   = _make_shm(D_arr)
    row_shm, row_sh = _make_shm(row_arr)
    col_shm, col_sh = _make_shm(col_arr)
    soz_shm, soz_sh = _make_shm(soz_arr)

    initargs = (
        W_shm.name,   W_sh.shape,   W_sh.dtype,
        D_shm.name,   D_sh.shape,   D_sh.dtype,
        row_shm.name, row_sh.shape, row_sh.dtype,
        col_shm.name, col_sh.shape, col_sh.dtype,
        soz_shm.name, soz_sh.shape, soz_sh.dtype,
    )

    start = time.time()
    with ProcessPoolExecutor(initializer=init_worker, initargs=initargs) as executor:
        list(executor.map(
            run_simulation,
            range(n_sims), [N]*n_sims, [M]*n_sims, [dt]*n_sims, [tf]*n_sims, [speed]*n_sims,
        ))
    end = time.time()

    for shm in (W_shm, D_shm, row_shm, col_shm, soz_shm):
        shm.close()
        shm.unlink()

    elapsed = end - start
    total_steps = n_sims * total_timesteps
    throughput = total_steps / elapsed
    return elapsed, throughput


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M     = 2
    dt    = 0.05
    tf    = 15.0
    speed = 4.0
    N_SIMS = 200

    W, D = data.tvb192_weights_lengths()
    N    = len(W)
    total_timesteps = int(tf / dt)

    print(f"=== Multi-simulation parallel: {N_SIMS} × TVB192 (N={N}, tf={tf}ms) ===")
    elapsed_multi, tp_multi = simulate_multi(W, D, N, M, dt, tf, speed, N_SIMS)
    print(f"  Wall time : {elapsed_multi:.2f}s")
    print(f"  Throughput: {tp_multi:,.0f} timesteps/s")

    # --- Exercise 2.6 (tvb_par_vec_sparse) baseline: known single-run time ---
    # From prior run: TVB192 at tf=15ms → 0.86s per simulation
    t_ex26_single = 0.86
    t_ex26_total  = t_ex26_single * N_SIMS
    tp_ex26       = (N_SIMS * total_timesteps) / t_ex26_total

    print(f"\n=== Exercise 2.6 (par_vec_sparse) back-to-back ===")
    print(f"  Single run: {t_ex26_single:.2f}s  →  {N_SIMS} runs: ~{t_ex26_total:.0f}s")
    print(f"  Throughput: {tp_ex26:,.0f} timesteps/s")

    print(f"\n--- Speedup (multi-sim vs Ex 2.6 back-to-back): {t_ex26_total/elapsed_multi:.1f}×")

    # --- comparison table ---
    print(f"\n{'Implementation':<30} {'Total time':>12} {'Throughput (ts/s)':>20}")
    print("-" * 64)
    print(f"{'Ex 2.6 × 200 (back-to-back)':<30} {t_ex26_total:>10.0f}s {tp_ex26:>20,.0f}")
    print(f"{'Multi-sim parallel (×200)':<30} {elapsed_multi:>10.2f}s {tp_multi:>20,.0f}")

    # --- bar chart: throughput comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    impls      = ["Ex 2.6\n(back-to-back)", "Multi-sim\n(parallel)"]
    times      = [t_ex26_total, elapsed_multi]
    throughputs = [tp_ex26, tp_multi]

    ax = axes[0]
    bars = ax.bar(impls, times, color=["#1f77b4", "#9467bd"], width=0.4)
    for bar, v in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f"{v:.1f}s", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Total Wall Time (s)", fontsize=12)
    ax.set_title(f"Wall Time — {N_SIMS} × TVB192 (tf={tf}ms)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    bars = ax.bar(impls, throughputs, color=["#1f77b4", "#9467bd"], width=0.4)
    for bar, v in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 200,
                f"{v:,.0f}", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Throughput (timesteps/s)", fontsize=12)
    ax.set_title(f"Throughput — {N_SIMS} × TVB192 (tf={tf}ms)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "comparison_multi_sim.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {save_path}")
