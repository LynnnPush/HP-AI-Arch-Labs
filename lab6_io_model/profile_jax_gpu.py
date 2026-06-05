#
# Profiling target for the jax_gpu backend (Goal B: one fused lax.scan).
#
# Purpose: run ONE timed io_model_jax.simulate under the SAME config as the
# sweep (f64 + kNN + record_every=40) so the profile EXPLAINS the swept curve,
# with the XLA compile / warm-up excluded. The flow mirrors sweep._step_jax:
# build device state once (HtoD, untimed), call once to warm/compile (untimed),
# then run the timed call inside an NVTX range so profilers can scope to it.
# (Note: the time loop lowers to a host-driven XLA `while`, so this is ~3 kernel
# launches per step, not one fused launch -- see the dev log.)
#
# Env knobs (defaults match the sweep.py jax_gpu run):
#   IO_N_CELLS=1024     cells (sweep this across runs to see the launch-bound
#                       small-N region -> the A-crossover mechanism)
#   IO_N_SIMSTEPS=4000  scan trip count (the time-loop length)
#   IO_REPEATS=3        timed instances inside the NVTX range (stable kernel stats)
#   IO_KNN=1 / IO_K=8   local kNN gap coupling (sweep default)
#   IO_RECORD_EVERY=40  strided-recording stride (sweep default; <=0 -> the
#                       no-record throughput path)
#   IO_X64=1            float64 (sweep default; 0 -> f32 fast path). Set before
#                       jax init.
#
import os
import sys
import time

# Absolute imports need the repo root on sys.path (mirrors sweep.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

if os.environ.get("IO_X64", "1") == "1":
    jax.config.update("jax_enable_x64", True)

from lab6_io_model.sweep import build_initial_state
from lab6_io_model.gpu import io_model_jax as jx


def main():
    n_cells = int(os.environ.get("IO_N_CELLS", 1024))
    n_simsteps = int(os.environ.get("IO_N_SIMSTEPS", 4000))
    repeats = int(os.environ.get("IO_REPEATS", 3))
    use_knn = os.environ.get("IO_KNN", "1") == "1"
    k = int(os.environ.get("IO_K", 8))
    record_every = int(os.environ.get("IO_RECORD_EVERY", 40))   # sweep default
    record = record_every > 0
    seed = 1981
    dtype = jnp.float64 if os.environ.get("IO_X64", "1") == "1" else jnp.float32

    # Device state + kNN adjacency, built ONCE outside the timed region (setup).
    st = build_initial_state(n_cells, g_CaL=None, seed=seed)
    state, g_CaL = jx.state_from_dict(st, dtype=dtype)
    neighbours = jx.build_neighbours(n_cells, k=k, seed=seed) if use_knn else None

    def call():
        # record_every>0 -> strided recording, matching the sweep (record_every=40);
        # the trace buffer is (ceil(n_simsteps/record_every), n_cells, 4). Set
        # IO_RECORD_EVERY<=0 for the no-record throughput path instead.
        return jx.simulate(state, g_CaL, neighbours, n_simsteps, 0.01, 1.0,
                           True, 0.0, 2.0, use_knn, record, max(1, record_every))

    # Warm/compile -- excluded from timing AND from the NVTX-scoped capture.
    jax.block_until_ready(call())

    # Timed: each iteration is ONE fused-scan launch; block_until_ready makes the
    # async dispatch complete before the clock stops. The NVTX range lets nsys/ncu
    # target precisely these instances (warm-up stays outside it).
    with jax.profiler.TraceAnnotation("sim_timed"):
        tic = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(call())
        wall = (time.perf_counter() - tic) / repeats

    cellsteps = n_simsteps * n_cells
    print(f"n_cells={n_cells} n_simsteps={n_simsteps} dtype={dtype.__name__} "
          f"knn={use_knn} record_every={record_every if record else 0} repeats={repeats}")
    print(f"wall/sim={wall*1e3:.3f} ms  throughput={cellsteps/wall/1e6:.2f} Mcell-steps/s")


if __name__ == "__main__":
    main()
