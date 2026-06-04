#
# Profiling target for the jax_gpu backend (Goal B: one fused lax.scan).
#
# Purpose: isolate EXACTLY ONE timed run of io_model_jax.simulate so nsys/ncu
# measure the single fused launch, not the XLA compile or warm-up. The flow
# mirrors sweep._step_jax: build device state once (HtoD, untimed), call once to
# warm/compile (untimed), then run the timed call inside an NVTX range so the
# profilers can scope to it (nsys --capture-range nvtx / ncu --nvtx-include).
#
# Env knobs:
#   IO_N_CELLS=1024     cells (sweep this across runs to see the flat launch-
#                       bound small-N region -> the A-crossover mechanism)
#   IO_N_SIMSTEPS=4000  scan trip count (the fused kernel's internal loop length)
#   IO_REPEATS=3        timed instances inside the NVTX range (stable kernel stats)
#   IO_KNN=1 / IO_K=8   local kNN gap coupling (matches the sweep default)
#   IO_X64=0            f32 fast path (1 -> float64; must be set before jax init)
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

if os.environ.get("IO_X64", "0") == "1":
    jax.config.update("jax_enable_x64", True)

from lab6_io_model.sweep import build_initial_state
from lab6_io_model.gpu import io_model_jax as jx


def main():
    n_cells = int(os.environ.get("IO_N_CELLS", 1024))
    n_simsteps = int(os.environ.get("IO_N_SIMSTEPS", 4000))
    repeats = int(os.environ.get("IO_REPEATS", 3))
    use_knn = os.environ.get("IO_KNN", "1") == "1"
    k = int(os.environ.get("IO_K", 8))
    seed = 1981
    dtype = jnp.float64 if os.environ.get("IO_X64", "0") == "1" else jnp.float32

    # Device state + kNN adjacency, built ONCE outside the timed region (setup).
    st = build_initial_state(n_cells, g_CaL=None, seed=seed)
    state, g_CaL = jx.state_from_dict(st, dtype=dtype)
    neighbours = jx.build_neighbours(n_cells, k=k, seed=seed) if use_knn else None

    def call():
        # record=False -> throughput path: the whole sim is one fused lax.scan,
        # one launch, near-zero DtoH (only the final carry crosses PCIe).
        return jx.simulate(state, g_CaL, neighbours, n_simsteps, 0.01, 1.0,
                           True, 0.0, 2.0, use_knn, False, 1)

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
          f"knn={use_knn} repeats={repeats}")
    print(f"wall/sim={wall*1e3:.3f} ms  throughput={cellsteps/wall/1e6:.2f} Mcell-steps/s  "
          f"(1 fused launch per sim)")


if __name__ == "__main__":
    main()
