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

def simulate(W, D, N, M, dt, tf, speed):
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
            result = executor.map(center_task, [Xs_shm.name]*N, [Xs_shared.shape]*N, [Xs_shared.dtype]*N, [W_shm.name]*N, [W_shared.shape]*N, [W_shared.dtype]*N, [D_shm.name]*N, [D_shared.shape]*N, [D_shared.dtype]*N, [t]*N, range(N), [dt]*N, chunksize=40)
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
    
    return T, Xs

if __name__ == "__main__":
    W, D = data.tvb76_weights_lengths()
    N = len(W)          # number of centers
    M = 2               # number of state variables per center

    dt = 0.05           # timestep size for the simulation in ms
    tf = 15.0           # final timestep of the simulation in ms
    speed = 4.0         # signal speed in mm/ms
    freq = 1.0          # frequency parameter for the local dynamics

    T, Xs = simulate(W, D, N, M, dt, tf, speed)
    plot.plot_xs(T, Xs, speed)
    # plot.plot_delay_hist(D, W, speed)
    