from lib import data, plot
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from lib.mlp_params import *
import numpy as np
import os
import time
from multiprocessing import shared_memory

pre = lambda x_src, x_dst: (x_src - 1.0)                                                # pre-synapse function
post = lambda gx: (1e-3 * gx)                                                           # post-synapse function

def f(x, y):
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


def calculate_coupling(Xs, W_row, D_row, t, n):
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

def step(Xs, t, n, c_in, dt):
    x = Xs[n, 0, t-1]
    y = Xs[n, 1, t-1]
    fx, fy = f(x, y)
    dx = dt * fx
    dy = dt * (fy + c_in)
    x_new = x + dx
    y_new = y + dy
    return [x_new, y_new]

def center_task(Xs_shm_name, Xs_shape, Xs_dtype, W_shm_name, W_shape, W_dtype, D_shm_name, D_shape, D_dtype, t, n, dt):

    Xs_shm = shared_memory.SharedMemory(name=Xs_shm_name)
    W_shm = shared_memory.SharedMemory(name=W_shm_name)
    D_shm = shared_memory.SharedMemory(name=D_shm_name)

    Xs = np.ndarray(Xs_shape, dtype=Xs_dtype, buffer=Xs_shm.buf)
    W = np.ndarray(W_shape, dtype=W_dtype, buffer=W_shm.buf)
    D = np.ndarray(D_shape, dtype=D_dtype, buffer=D_shm.buf)

    c_in = calculate_coupling(Xs, W[n], D[n], t, n)
    X_new = step(Xs, t, n, c_in, dt)

    Xs_shm.close()
    W_shm.close()
    D_shm.close()

    return X_new

def simulate(W, D, N, M, dt, tf, speed, chunksize=1):   # chunksize: tasks batched per worker call
    total_timesteps = int(tf/dt) # total number of timesteps the simulation must run
    Xs = np.zeros((N, M, total_timesteps)) # state variable list (M list of total_timesteps values for each N centers)
    D_timestep = ((D / speed) / dt).astype(int) # delay list in terms of timesteps

    Xs_shm = shared_memory.SharedMemory(create=True, size=Xs.nbytes)
    W_shm = shared_memory.SharedMemory(create=True, size=W.nbytes)
    D_shm = shared_memory.SharedMemory(create=True, size=D.nbytes)

    Xs_shared = np.ndarray(Xs.shape, dtype=Xs.dtype, buffer=Xs_shm.buf)
    W_shared = np.ndarray(W.shape, dtype=W.dtype, buffer=W_shm.buf)
    D_shared = np.ndarray(D_timestep.shape, dtype=D_timestep.dtype, buffer=D_shm.buf)

    np.copyto(Xs_shared, Xs)
    np.copyto(W_shared, W)
    np.copyto(D_shared, D_timestep)

    for n in range(N):
        for m in range(M):
            Xs_shared[n, m, 0] = -1.0

    start = time.time()
    with ProcessPoolExecutor() as executor:
        for t in range(1, total_timesteps):
            result = executor.map(center_task, [Xs_shm.name]*N, [Xs_shared.shape]*N, [Xs_shared.dtype]*N, [W_shm.name]*N, [W_shared.shape]*N, [W_shared.dtype]*N, [D_shm.name]*N, [D_shared.shape]*N, [D_shared.dtype]*N, [t]*N, range(N), [dt]*N, chunksize=chunksize)
            result = list(result)
            for n in range(N):
                for m in range(M):
                    Xs_shared[n, m, t] = result[n][m]

    end = time.time()

    np.copyto(Xs, Xs_shared)

    Xs_shm.close()
    W_shm.close()
    D_shm.close()
    Xs_shm.unlink()
    W_shm.unlink()
    D_shm.unlink()

    T = [t * dt for t in range(total_timesteps)]

    print(end-start)

    return T, Xs, end - start


if __name__ == "__main__":
    import sys, matplotlib.pyplot as plt

    # Import MLP sequential implementation from lab1
    sys.path.insert(0, '/home/ubuntu/Labs/lab1_tvb')
    import tvb_seq_mlp
    sys.path.pop(0)

    M = 2               # number of state variables per center
    dt = 0.05           # timestep size for the simulation in ms
    tf = 15.0           # final timestep of the simulation in ms
    speed = 4.0         # signal speed in mm/ms
    freq = 1.0          # frequency parameter for the local dynamics

    # Reuse the chunksize sweep from tvb_par.py, but with this script's shared-memory simulate
    from tvb_par import sweep_chunksizes

    DATASETS = [
        ("TVB76",  data.tvb76_weights_lengths),
        ("TVB192", data.tvb192_weights_lengths),
        ("TVB998", data.tvb998_weights_lengths),
    ]

    # Known tvb_par.py (no shared memory) timings at optimal chunksize, tf=15ms
    par_nosm_times = {"TVB76": 1.68, "TVB192": 5.43, "TVB998": None}

    seq_times    = {}   # sequential MLP baseline
    par_sm_times = {}   # this script (shared memory)

    for label, loader in DATASETS:
        W, D = loader()
        N = len(W)      # number of centers
        print(f"\n=== {label} (N={N}) ===")

        # --- find optimal chunksize via shared-memory sweep (reused from tvb_par.py) ---
        chunk_sizes, sweep_times = sweep_chunksizes(W, D, N, M, dt, tf, speed, label, simulate_fn=simulate)
        best_cs = chunk_sizes[int(np.argmin(sweep_times))]
        print(f"  -> best chunksize for {label}: {best_cs}")

        # --- parallel shared-memory run at optimal chunksize ---
        _, _, t_sm = simulate(W, D, N, M, dt, tf, speed, chunksize=best_cs)
        par_sm_times[label] = t_sm
        print(f"  par_sm (cs={best_cs}): {t_sm:.2f}s")

        # --- sequential MLP baseline ---
        W_list = W.tolist()
        D_list = D.tolist()
        t0 = time.time()
        tvb_seq_mlp.simulate(W_list, D_list, N, M, dt, tf, speed, False)
        seq_times[label] = time.time() - t0
        print(f"  sequential       : {seq_times[label]:.2f}s")

    # --- comparison table ---
    print("\n{:<8} {:>12} {:>12} {:>12}".format("Dataset", "Sequential", "Par (no SM)", "Par (SM)"))
    print("-" * 48)
    for label in ["TVB76", "TVB192", "TVB998"]:
        seq  = f"{seq_times[label]:.2f}s"      if seq_times[label]      else "N/A"
        nosm = f"{par_nosm_times[label]:.2f}s" if par_nosm_times[label] else "N/A"
        sm   = f"{par_sm_times[label]:.2f}s"
        print(f"{label:<8} {seq:>12} {nosm:>12} {sm:>12}")

    # --- bar chart comparison ---
    labels    = ["TVB76", "TVB192", "TVB998"]
    x         = np.arange(len(labels))
    width     = 0.25

    seq_vals  = [seq_times[l]      if seq_times[l]      else 0 for l in labels]
    nosm_vals = [par_nosm_times[l] if par_nosm_times[l] else 0 for l in labels]
    sm_vals   = [par_sm_times[l]                               for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, seq_vals,  width, label='Sequential',  color='#7f7f7f')
    b2 = ax.bar(x,         nosm_vals, width, label='Par (no SM)', color='#1f77b4')
    b3 = ax.bar(x + width, sm_vals,   width, label='Par (SM)',    color='#2ca02c')

    # Annotate bars with timing values
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f"{h:.1f}s", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title(f"TVB Simulation: Sequential vs Parallel (tf={tf}ms)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "comparison_sm.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {save_path}")
    # plot.plot_delay_hist(D, W, speed)
    