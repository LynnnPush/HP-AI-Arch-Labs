from lib import data, plot
from concurrent.futures import ThreadPoolExecutor           # threads share memory; no pickling
from lib.mlp_params import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt

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

def step(Xs, t, n, c_in, dt):
    x = Xs[n][0][t-1]
    y = Xs[n][1][t-1]
    fx, fy = f(x, y)
    dx = dt * fx
    dy = dt * (fy + c_in)
    x_new = x + dx
    y_new = y + dy
    return [x_new, y_new]

def center_task(Xs, W_row, D_row, t, n, dt):
    c_in = calculate_coupling(Xs, W_row, D_row, t, n)
    return step(Xs, t, n, c_in, dt)

def simulate(W, D, N, M, dt, tf, speed, chunksize=1):   # chunksize: tasks batched per worker call
    total_timesteps = int(tf/dt) # total number of timesteps the simulation must run
    Xs = [[[0.0 for _ in range(total_timesteps)] for _ in range(M)] for _ in range(N)] # state variable list (M list of total_timesteps values for each N centers)
    D_timestep = [[int((D[i][j] / speed) / dt) for i in range(N)] for j in range(N)] # delay list in terms of timesteps

    for n in range(N):
        for m in range(M):
            Xs[n][m][0] = -1.0

    start = time.time()
    with ThreadPoolExecutor() as executor:                 # threads: Xs shared by reference, no IPC pickle
        for t in range(1, total_timesteps):
            result = executor.map(center_task, [Xs]*N, W, D_timestep, [t]*N, range(N), [dt]*N, chunksize=chunksize)
            result = list(result)
            for n in range(N):
                for m in range(M):
                    Xs[n][m][t] = result[n][m]
    end = time.time()

    T = [t * dt for t in range(total_timesteps)]

    print(end-start)

    return T, Xs, end - start


def sweep_tf(W, D, N, M, dt, tf_values, speed, chunksize, label, sim_fn=None):
    # Run simulation at fixed optimal chunksize across multiple tf values
    # sim_fn: which simulate() to call; defaults to the local (threaded) one
    if sim_fn is None:
        sim_fn = simulate
    times = []
    print(f"\n--- {label} (N={N}, chunksize={chunksize}) ---")
    for tf in tf_values:
        _, _, elapsed = sim_fn(W, D, N, M, dt, tf, speed, chunksize=chunksize)
        print(f"  tf={tf:5.1f}ms: {elapsed:.2f}s")
        times.append(elapsed)
    return times


if __name__ == "__main__":
    # Import multiprocessed simulate for back-to-back comparison
    from tvb_par import simulate as simulate_mp

    M = 2               # number of state variables per center
    dt = 0.05           # timestep size for the simulation in ms
    speed = 4.0         # signal speed in mm/ms

    # Optimal chunk sizes from tvb_par.py sweep (same for fair comparison)
    BEST_CS_76  = 57
    BEST_CS_192 = 135

    # --- Load datasets ---
    W76, D76 = data.tvb76_weights_lengths()
    W76_l = W76.tolist(); D76_l = D76.tolist(); N76 = len(W76_l)

    W192, D192 = data.tvb192_weights_lengths()
    W192_l = W192.tolist(); D192_l = D192.tolist(); N192 = len(W192_l)

    # ----------------------------------------------------------------
    # Part 1 — Compare at tf=15ms: threading vs multiprocessing
    # ----------------------------------------------------------------
    tf = 15.0
    print("=== Part 1: tf=15ms comparison ===")

    print("\n-- Multithreaded --")
    _, _, mt_76  = simulate(W76_l,  D76_l,  N76,  M, dt, tf, speed, BEST_CS_76)
    _, _, mt_192 = simulate(W192_l, D192_l, N192, M, dt, tf, speed, BEST_CS_192)

    print("\n-- Multiprocessed --")
    _, _, mp_76  = simulate_mp(W76_l,  D76_l,  N76,  M, dt, tf, speed, BEST_CS_76)
    _, _, mp_192 = simulate_mp(W192_l, D192_l, N192, M, dt, tf, speed, BEST_CS_192)

    # Known sequential baselines from lab1
    seq_times = {"TVB76": 0.78, "TVB192": 2.65}

    print(f"\n{'Dataset':<8} {'Sequential':>12} {'Multiproc':>12} {'Multithread':>14}")
    print("-" * 50)
    print(f"{'TVB76':<8} {seq_times['TVB76']:>11.2f}s {mp_76:>11.2f}s {mt_76:>13.2f}s")
    print(f"{'TVB192':<8} {seq_times['TVB192']:>11.2f}s {mp_192:>11.2f}s {mt_192:>13.2f}s")

    # ----------------------------------------------------------------
    # Part 2 — tf sweep 15→60ms: threading vs multiprocessing (TVB76)
    #          TVB192 threading only (multiprocessed at 60ms is very slow)
    # ----------------------------------------------------------------
    tf_values = [15.0, 20.0, 30.0, 45.0, 60.0]

    print("\n=== Part 2: tf sweep ===")
    mt_times_76  = sweep_tf(W76_l,  D76_l,  N76,  M, dt, tf_values, speed, BEST_CS_76,  "TVB76  (threaded)")
    mt_times_192 = sweep_tf(W192_l, D192_l, N192, M, dt, tf_values, speed, BEST_CS_192, "TVB192 (threaded)")

    print("\n-- TVB76 multiprocessed tf sweep (for scaling comparison) --")
    mp_times_76  = sweep_tf(W76_l,  D76_l,  N76,  M, dt, tf_values, speed, BEST_CS_76,  "TVB76  (multiproc)", sim_fn=simulate_mp)
    mp_times_192 = sweep_tf(W192_l, D192_l, N192, M, dt, tf_values, speed, BEST_CS_192, "TVB192 (multiproc)", sim_fn=simulate_mp)

    # ----------------------------------------------------------------
    # Plot 1: bar chart — tf=15ms comparison
    # ----------------------------------------------------------------
    labels = ["TVB76", "TVB192"]
    x      = np.arange(len(labels))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width, [seq_times["TVB76"], seq_times["TVB192"]], width,
                label="Sequential",     color="#7f7f7f")
    b2 = ax.bar(x,         [mp_76,  mp_192],  width,
                label="Multiprocessed", color="#1f77b4")
    b3 = ax.bar(x + width, [mt_76,  mt_192],  width,
                label="Multithreaded",  color="#ff7f0e")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.2f}s", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title(f"Sequential vs Multiprocessed vs Multithreaded (tf={tf}ms)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_mt_15ms.png", dpi=150)
    plt.show()

    # ----------------------------------------------------------------
    # Plot 2: tf scaling — threading vs multiprocessing for TVB76,
    #         plus threading for TVB192
    # ----------------------------------------------------------------
    tf_arr = np.array(tf_values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # TVB76: threaded vs multiprocessed + references
    ax = axes[0]
    mt_arr = np.array(mt_times_76)
    mp_arr = np.array(mp_times_76)
    linear_ref   = mt_arr[0] * (tf_arr / tf_arr[0])
    quadratic_ref = mp_arr[0] * (tf_arr / tf_arr[0])**2

    ax.plot(tf_arr, mt_arr,       'o-',  color="#ff7f0e", linewidth=1.8, markersize=6, label="Multithreaded (measured)")
    ax.plot(tf_arr, mp_arr,       's--', color="#1f77b4", linewidth=1.8, markersize=6, label="Multiprocessed (measured)")
    ax.plot(tf_arr, linear_ref,   'k:',  linewidth=1.2,                                label="Linear reference")
    ax.plot(tf_arr, quadratic_ref,'r:',  linewidth=1.2,                                label="Quadratic reference")
    ax.set_xlabel("Simulation Time tf (ms)", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title(f"TVB76 (N={N76}): tf scaling", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # TVB192: threaded only + references
    ax = axes[1]
    mt_arr_192 = np.array(mt_times_192)
    linear_ref_192   = mt_arr_192[0] * (tf_arr / tf_arr[0])
    quadratic_ref_192 = mt_arr_192[0] * (tf_arr / tf_arr[0])**2

    ax.plot(tf_arr, mt_arr_192,        'o-', color="#ff7f0e", linewidth=1.8, markersize=6, label="Multithreaded (measured)")
    ax.plot(tf_arr, linear_ref_192,    'k:', linewidth=1.2,                                label="Linear reference")
    ax.plot(tf_arr, quadratic_ref_192, 'r:', linewidth=1.2,                                label="Quadratic reference")
    ax.set_xlabel("Simulation Time tf (ms)", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title(f"TVB192 (N={N192}): tf scaling (threaded)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tf_sweep_mt.png", dpi=150)
    plt.show()
    print("\nPlots saved: comparison_mt_15ms.png, tf_sweep_mt.png")
