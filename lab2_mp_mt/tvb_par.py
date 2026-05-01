from lib import data, plot
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
    with ProcessPoolExecutor() as executor:
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


def sweep_tf(W, D, N, M, dt, tf_values, speed, chunksize, label):
    # Run simulation at fixed optimal chunksize across multiple tf values
    times = []
    print(f"\n--- {label} (N={N}, chunksize={chunksize}) ---")
    for tf in tf_values:
        _, _, elapsed = simulate(W, D, N, M, dt, tf, speed, chunksize=chunksize)
        print(f"  tf={tf:5.1f}ms: {elapsed:.2f}s")
        times.append(elapsed)
    return times


def sweep_chunksizes(W, D, N, M, dt, tf, speed, label, simulate_fn=None, early_stop_streak=3):
    # Log-spaced chunk sizes from 2 to N, always including N itself.
    # Early stop: once the running min has been seen, abort after `early_stop_streak`
    # consecutive strictly-increasing samples (trend is clear; remaining cs only get worse).
    if simulate_fn is None:
        simulate_fn = simulate
    raw = np.geomspace(2, N, num=14, dtype=int)
    chunk_sizes = sorted(set(raw.tolist() + [N]))

    times = []
    tested_cs = []
    best_t = float('inf')
    increasing_streak = 0
    print(f"\n--- {label} (N={N}) ---")
    for cs in chunk_sizes:
        _, _, elapsed = simulate_fn(W, D, N, M, dt, tf, speed, chunksize=cs)
        print(f"  chunksize={cs:4d}: {elapsed:.2f}s")
        times.append(elapsed)
        tested_cs.append(cs)

        if elapsed < best_t:
            best_t = elapsed
            increasing_streak = 0
        elif len(times) >= 2 and elapsed > times[-2]:
            increasing_streak += 1
        else:
            increasing_streak = 0

        if increasing_streak >= early_stop_streak:
            print(f"  early stop: {increasing_streak} consecutive increases past min")
            break

    return tested_cs, times

if __name__ == "__main__":
    M = 2               # number of state variables per center
    dt = 0.05           # timestep size for the simulation in ms
    speed = 4.0         # signal speed in mm/ms
    freq = 1.0          # frequency parameter for the local dynamics

    # Optimal chunk sizes from prior sweep
    BEST_CS_76  = 57    # best chunksize for TVB76
    BEST_CS_192 = 135   # best chunksize for TVB192

    tf_values = [15.0, 20.0, 30.0, 45.0, 60.0]  # simulation durations to test

    # TVB76 tf sweep
    W76, D76 = data.tvb76_weights_lengths()
    W76 = W76.tolist()  # weight matrix
    D76 = D76.tolist()  # distance matrix
    N76 = len(W76)      # number of centers
    times_76 = sweep_tf(W76, D76, N76, M, dt, tf_values, speed, BEST_CS_76, "TVB76")

    # TVB192 tf sweep
    W192, D192 = data.tvb192_weights_lengths()
    W192 = W192.tolist() # weight matrix
    D192 = D192.tolist() # distance matrix
    N192 = len(W192)     # number of centers
    times_192 = sweep_tf(W192, D192, N192, M, dt, tf_values, speed, BEST_CS_192, "TVB192")

    # Plot: measured times vs tf, with linear reference from the tf=15 point
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, times, label, N, cs, color in [
        (axes[0], times_76,  "TVB76",  N76,  BEST_CS_76,  "steelblue"),
        (axes[1], times_192, "TVB192", N192, BEST_CS_192, "tomato"),
    ]:
        tf_arr = np.array(tf_values)
        t_arr  = np.array(times)
        # Linear reference anchored at tf=15
        linear_ref = t_arr[0] * (tf_arr / tf_arr[0])
        ax.plot(tf_arr, t_arr, 'o-', color=color, linewidth=1.8, markersize=6, label='Measured')
        ax.plot(tf_arr, linear_ref, 'k--', linewidth=1.2, label='Linear reference')
        ax.set_xlabel('Simulation Time tf (ms)', fontsize=12)
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_title(f'{label} (N={N}, chunksize={cs})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'tf_sweep.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {save_path}")
    # plot.plot_delay_hist(D, W, speed)
    