"""JAX-based TVB simulation, runnable on CPU or GPU.

Built on top of the vectorized 3_1_2 algorithm. Key JAX considerations:
  * The python `for t in range(...)` loop is replaced by `jax.lax.scan`, so
    only ONE compute graph is traced regardless of timestep count.
  * Functional updates use `Xs.at[...].set(...)` instead of in-place writes.
  * The whole simulation loop is wrapped in `jax.jit` and re-runs per shape;
    we therefore close over `dt` / `N` / `total_timesteps` as static.
  * 64-bit floats are enabled explicitly (JAX defaults to float32).

Backend selection: pass `cpu` or `gpu` as the first CLI arg, or set the
env var `JAX_PLATFORM_NAME` before importing jax. The chosen platform must
be configured before `import jax` for the platform flag to take effect.
"""
import os
import sys
import time
import numpy as np
from typing import List, Tuple

# --- Backend selection (must happen BEFORE `import jax`) ---------------------
# Resolve from CLI arg if provided, else honor existing env var, else "cpu".
_backend_arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
if _backend_arg in ("cpu", "gpu", "tpu"):
    os.environ["JAX_PLATFORM_NAME"] = _backend_arg
_BACKEND = os.environ.get("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

# Enable 64-bit floats (required by the assignment) and confirm backend.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", _BACKEND)

from lib import data
from lib.mlp_params import layer_1_b_np, layer_1_w_np, layer_2_b_np, layer_2_w_np

# Push MLP weights to JAX device arrays once at import time.
layer_1_w = jnp.asarray(layer_1_w_np)   # [M, L]
layer_1_b = jnp.asarray(layer_1_b_np)   # [L]
layer_2_w = jnp.asarray(layer_2_w_np)   # [L, M]
layer_2_b = jnp.asarray(layer_2_b_np)   # [M]


def pre(x_src: jnp.ndarray, x_dst: jnp.ndarray) -> jnp.ndarray:
    return x_src - 1.0


def post(gx: jnp.ndarray) -> jnp.ndarray:
    return 1e-3 * gx


def f(X: jnp.ndarray) -> jnp.ndarray:
    """Two-layer MLP dynamics (linear + ReLU + linear), vectorized over nodes."""
    hidden = jnp.matmul(X, layer_1_w) + layer_1_b           # [N, L]
    hidden = jnp.where(hidden <= 0, 0.0, hidden)            # ReLU
    out = jnp.matmul(hidden, layer_2_w) + layer_2_b         # [N, M]
    return out


def calculate_coupling(
    Xs: jnp.ndarray,
    W: jnp.ndarray,
    D_timestep: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """Delayed coupling, identical math to the NumPy/CuPy versions."""
    N = Xs.shape[0]
    valid_time = (t >= D_timestep)                                       # [N, N]
    use_prev = jnp.eye(N, dtype=bool) | (D_timestep == 0)               # [N, N]
    timesteps_indices = jnp.where(use_prev, t - 1, t - D_timestep)      # [N, N]
    safe_indices = jnp.where(valid_time, timesteps_indices, 0)           # [N, N]

    # Gather delayed source states; JAX supports the same fancy indexing.
    src_idx = jnp.arange(N)[jnp.newaxis, :]                              # [1, N]
    x_src = Xs[src_idx, 0, safe_indices]                                 # [N, N]
    x_src = jnp.where(valid_time, x_src, 0.0)                           # [N, N]

    x_dst = Xs[:, 0, t - 1][:, jnp.newaxis]                             # [N, 1]
    c_in = jnp.sum(W * pre(x_src, x_dst), axis=1)                       # [N]
    return post(c_in)                                                     # [N]


def step(Xs: jnp.ndarray, t: jnp.ndarray, c_ins: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Forward Euler update for all nodes in one shot."""
    X_all = Xs[:, :, t - 1]                                              # [N, M]
    fx = f(X_all)                                                         # [N, M]
    # Coupling injects only into the second state variable. Functional update.
    fx = fx.at[:, 1].add(c_ins)
    return X_all + fx * dt                                                # [N, M]


def make_simulate_fn(N: int, M: int, total_timesteps: int, dt: float):
    """Return a JIT-compiled simulation closure for the given shapes/timestep."""

    def simulate_jit(W: jnp.ndarray, D_timestep: jnp.ndarray) -> jnp.ndarray:
        # Initial state buffer with the t=0 condition baked in.
        Xs0 = jnp.zeros((N, M, total_timesteps), dtype=jnp.float64)
        Xs0 = Xs0.at[:, :, 0].set(-1.0)

        def body(Xs, t):
            # One timestep: compute delayed coupling, advance state, write back.
            c_ins = calculate_coupling(Xs, W, D_timestep, t)
            X_new = step(Xs, t, c_ins, dt)
            Xs = Xs.at[:, :, t].set(X_new)
            return Xs, None

        # Scan over t = 1 .. total_timesteps-1 (t=0 is pre-initialized above).
        ts = jnp.arange(1, total_timesteps)
        Xs_final, _ = jax.lax.scan(body, Xs0, ts)
        return Xs_final

    return jax.jit(simulate_jit)


def simulate(
    W: np.ndarray,
    D: np.ndarray,
    N: int,
    M: int,
    dt: float,
    tf: float,
    speed: float,
) -> Tuple[List[float], np.ndarray]:
    """Compile + run the JAX simulation; report compile and run wall times."""
    total_timesteps = int(tf / dt)

    # Convert distances to integer-timestep delays on host, then upload.
    W_dev = jnp.asarray(W)
    D_timestep_dev = jnp.asarray(((D / speed) / dt).astype(np.int64))

    sim_fn = make_simulate_fn(N, M, total_timesteps, dt)

    # First call triggers tracing + XLA compilation; time it separately so the
    # reported "run" time reflects pure execution. block_until_ready forces
    # async dispatch to complete before stopping the timer.
    compile_start = time.time()
    Xs_warm = sim_fn(W_dev, D_timestep_dev)
    Xs_warm.block_until_ready()
    compile_end = time.time()

    run_start = time.time()
    Xs_dev = sim_fn(W_dev, D_timestep_dev)
    Xs_dev.block_until_ready()
    run_end = time.time()

    print(f"[simulate] JAX backend: {jax.default_backend()}")
    print(f"[simulate] Compile + first run: {compile_end - compile_start:.6f}s")
    print(f"[simulate] Pure run time:       {run_end - run_start:.6f}s")

    T = [t * dt for t in range(total_timesteps)]
    Xs_cpu = np.asarray(Xs_dev)
    return T, Xs_cpu


if __name__ == "__main__":
    datasets = [
        ("TVB76",  data.tvb76_weights_lengths),
        ("TVB192", data.tvb192_weights_lengths),
        ("TVB998", data.tvb998_weights_lengths),
    ]

    M = 2        # state variables per node
    dt = 0.05    # timestep size in ms
    tf = 150.0   # simulation duration in ms
    speed = 4.0  # signal propagation speed in mm/ms

    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    for name, loader in datasets:
        print(f"\n{'='*40}")
        print(f"Running {name}  (tf={tf}ms, dt={dt}ms)")
        print(f"{'='*40}")
        W, D = loader()
        N = len(W)
        wall_start = time.time()
        T, Xs = simulate(W, D, N, M, dt, tf, speed)
        wall_end = time.time()
        print(f"[{name}] Wall-clock total (incl. compile + transfers): {wall_end - wall_start:.6f}s")
