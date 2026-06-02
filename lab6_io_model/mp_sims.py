#
# Goal C: multiprocessing across INDEPENDENT simulations (jit backbone).
#
# Each sim is fully self-contained: it builds its own state from a seed, so the
# sims share nothing -> embarrassingly parallel, just fan them out across cores
# with a ProcessPoolExecutor (same convention as lab2/tvb_multi_sim.py). Unlike
# lab2 there are no large read-only arrays to share, so no shared_memory is
# needed -- each task carries only scalars.
#
# The per-sim backbone is the scalar Numba backend io_model_jit.simulate.
#

from concurrent.futures import ProcessPoolExecutor
import argparse
import time

import numpy as np

import io_model_jit
from sweep import build_initial_state

SPIKE_THRESHOLD = -20.0  # mV; soma upward crossings counted as spikes


def run_simulation(seed, n_cells, sim_seconds, delta, enable_gj=True,
                   I_pulse10ms=2.0, return_trace=False):
    """Run one independent sim and return its total soma spike count.

    Builds an independent initial state from `seed`, runs the jit backbone, and
    reduces the trace to a single number so workers return almost nothing (no
    big traces to pickle back to the parent).

    With `return_trace=True` it returns the full (n_simsteps, n_cells, 4) voltage
    trace instead -- used by validate.py to confirm the cross-sim pipeline
    reproduces the jit numerics end-to-end (one sim shipped through a worker and
    its trace pickled back). Not used on the throughput path.
    """
    st = build_initial_state(n_cells, None, seed)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    v_trace, _ = io_model_jit.simulate(
        st["V_soma"], st["V_axon"], st["V_dend"],
        st["soma_k"], st["soma_l"], st["soma_h"], st["soma_n"], st["soma_x"],
        st["axon_Sodium_h"], st["axon_Potassium_x"],
        st["dend_Ca2Plus"], st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
        st["g_CaL"], n_cells, n_simsteps, delta, sim_seconds,
        enable_gj, 0.0, I_pulse10ms, True,
    )
    if return_trace:
        return v_trace
    soma = v_trace[:, :, 0]
    return int(((soma[:-1] < SPIKE_THRESHOLD) & (soma[1:] >= SPIKE_THRESHOLD)).sum())


def simulate_multi(n_sims, n_cells, sim_seconds, delta, enable_gj=True,
                   I_pulse10ms=2.0, seed_base=1000):
    """Launch n_sims independent sims in parallel; return (elapsed, throughput, results)."""
    # Compile the jit kernel once in the parent before timing; fork workers then
    # inherit the compiled code (cache=True covers the spawn case too).
    run_simulation(seed_base, 2, 0.00002, delta, enable_gj, I_pulse10ms)

    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    seeds = range(seed_base, seed_base + n_sims)

    start = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            run_simulation, seeds,
            [n_cells] * n_sims, [sim_seconds] * n_sims, [delta] * n_sims,
            [enable_gj] * n_sims, [I_pulse10ms] * n_sims,
        ))
    elapsed = time.time() - start

    throughput = n_sims * n_simsteps * n_cells / elapsed  # cell-steps/s
    return elapsed, throughput, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run many independent IO-model sims in parallel (jit backbone).")
    parser.add_argument("--n-sims", type=int, default=64)
    parser.add_argument("--sim-seconds", type=float, default=1.0)
    parser.add_argument("--n-cells", type=int, default=30)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--no-gj", action="store_true", help="Disable gap junctions.")
    args = parser.parse_args()

    enable_gj = not args.no_gj
    n_simsteps = int(args.sim_seconds * 1000 / args.delta + 0.5)
    print(f"=== {args.n_sims} independent sims | {args.n_cells} cells x "
          f"{n_simsteps} steps each ===")

    # Single-sim reference (warmed): estimates serial back-to-back cost.
    run_simulation(0, args.n_cells, args.sim_seconds, args.delta, enable_gj)  # warm-up
    t = time.time()
    run_simulation(1, args.n_cells, args.sim_seconds, args.delta, enable_gj)
    single = time.time() - t

    # Parallel multi-sim.
    elapsed, throughput, results = simulate_multi(
        args.n_sims, args.n_cells, args.sim_seconds, args.delta, enable_gj)

    serial_est = single * args.n_sims
    print(f"single sim      : {single:.3f}s")
    print(f"serial (est x{args.n_sims}): {serial_est:.2f}s")
    print(f"parallel        : {elapsed:.2f}s   ({throughput:,.0f} cell-steps/s)")
    print(f"speedup         : {serial_est / elapsed:.1f}x")
    print(f"total spikes    : {sum(results)}")
