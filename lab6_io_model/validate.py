#
# Trace-divergence validation for the CPU-backend optimization of io_model.py.
#
# Two distinct comparisons (never conflated):
#   * jit vs vec  -> equality check (same Jacobi numerics)  [added with Goal B]
#   * vec vs baseline -> divergence check. The vec backend uses a Jacobi update
#     while the baseline is Gauss-Seidel, so traces are EXPECTED to drift apart;
#     we measure the divergence (max/RMS, onset time) rather than assert equality.
#
# io_model.py stays the untouched reference; we only drive its update_* helpers
# and reuse sweep.build_initial_state so every backend starts from the exact same
# seeded initial state.
#

import argparse

import numpy as np

# io_model defines the update_* helpers + default globals; importing it does NOT
# run its __main__ simulation. sweep gives us the seeded state initializer.
import io_model
import io_model_vec
import io_model_vec_jit
import io_model_jit
import mp_sims
import mp_sims_intra
from sweep import build_initial_state


def run_baseline_trace(n_cells=30, sim_seconds=1.0, delta=0.01,
                       enable_gapjunctions=True, I_pulse10ms=2.0,
                       g_CaL=None, seed=1981):
    """Run the reference (Gauss-Seidel) baseline loop and return its voltage traces.

    This is the original io_model.__main__ loop, lifted verbatim into a callable
    so validation (and benching) can reproduce the exact baseline numerics from a
    known seed. Behaviour matches the reference exactly:
      * per epoch each cell updates soma -> axon -> dend in sequence, so later
        compartments/cells see this step's freshly-written V (Gauss-Seidel);
      * gap junctions are the all-to-all O(N^2) inner loop inside update_dend.

    Returns (v_trace, n_simsteps): v_trace is a list of (n_simsteps, 4) arrays,
    one per cell (columns: V_soma, V_axon, V_dend, t).
    """
    # The update_* helpers read these as module globals, so push the run params
    # into io_model before stepping (exactly what sweep.run_once does).
    io_model.sim_seconds = sim_seconds
    io_model.delta = delta
    io_model.n_cells = n_cells
    io_model.enable_gapjunctions = enable_gapjunctions
    io_model.I_pulse10ms = I_pulse10ms

    # Seeded initial state, shared with the benchmark harness for apples-to-apples.
    st = build_initial_state(n_cells, g_CaL, seed)

    # Step count + per-cell trace buffers (allocation kept out of the hot loop).
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    v_trace = [np.empty((n_simsteps, 4)) for _ in range(n_cells)]
    t = 0.0

    # Hot loop: outer over time, inner over cells (sequential => Gauss-Seidel).
    for i_epoch in range(n_simsteps):
        for i_cell in range(n_cells):
            io_model.update_soma(
                i_cell, v_trace[i_cell], i_epoch,
                st["V_soma"], st["V_axon"], st["V_dend"],
                st["soma_k"], st["soma_l"], st["soma_h"],
                st["soma_n"], st["soma_x"], st["g_CaL"],
            )
            io_model.update_axon(
                i_cell, v_trace[i_cell], i_epoch,
                st["V_soma"], st["V_axon"],
                st["axon_Sodium_h"], st["axon_Potassium_x"],
            )
            io_model.update_dend(
                i_cell, v_trace[i_cell], i_epoch, t,
                st["V_soma"], st["V_dend"], st["dend_Ca2Plus"],
                st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
            )
            v_trace[i_cell][i_epoch, -1] = t  # record time column
        t += delta  # advance brain time for the next step

    return v_trace, n_simsteps


def run_vec_trace(n_cells=30, sim_seconds=1.0, delta=0.01,
                  enable_gapjunctions=True, I_pulse10ms=2.0,
                  g_CaL=None, seed=1981):
    """Run the vectorized (Jacobi) backend from the same seeded state as baseline.

    Returns (v_trace, n_simsteps): v_trace is a (n_simsteps, n_cells, 4) array
    (columns: V_soma, V_axon, V_dend, t) -- the io_model_vec layout.
    """
    st = build_initial_state(n_cells, g_CaL, seed)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    return io_model_vec.simulate(
        st["V_soma"], st["V_axon"], st["V_dend"],
        st["soma_k"], st["soma_l"], st["soma_h"], st["soma_n"], st["soma_x"],
        st["axon_Sodium_h"], st["axon_Potassium_x"],
        st["dend_Ca2Plus"], st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
        st["g_CaL"], n_cells, n_simsteps, delta, sim_seconds,
        enable_gapjunctions=enable_gapjunctions, I_pulse10ms=I_pulse10ms,
    )


def run_jit_trace(n_cells=30, sim_seconds=1.0, delta=0.01,
                  enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
                  g_CaL=None, seed=1981, record=True):
    """Run the Numba @njit (Jacobi) backend from the same seeded state as baseline.

    Returns (v_trace, n_simsteps) in the (n_simsteps, n_cells, 4) io_model layout.
    Note: the FIRST call triggers JIT compilation -- call warmup_jit() first if
    you want to keep compilation out of a timed region.
    """
    st = build_initial_state(n_cells, g_CaL, seed)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    return io_model_jit.simulate(
        st["V_soma"], st["V_axon"], st["V_dend"],
        st["soma_k"], st["soma_l"], st["soma_h"], st["soma_n"], st["soma_x"],
        st["axon_Sodium_h"], st["axon_Potassium_x"],
        st["dend_Ca2Plus"], st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
        st["g_CaL"], n_cells, n_simsteps, delta, sim_seconds,
        enable_gapjunctions, I_app, I_pulse10ms, record,
    )


def run_vec_jit_trace(n_cells=30, sim_seconds=1.0, delta=0.01,
                      enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
                      g_CaL=None, seed=1981, record=True):
    """Run the njit-compiled VECTORIZED (Jacobi) backend from the same seeded state.

    Returns (v_trace, n_simsteps) in the (n_simsteps, n_cells, 4) io_model layout.
    Like run_jit_trace, the FIRST call triggers compilation -- warmup_jit() first.
    """
    st = build_initial_state(n_cells, g_CaL, seed)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    return io_model_vec_jit.simulate(
        st["V_soma"], st["V_axon"], st["V_dend"],
        st["soma_k"], st["soma_l"], st["soma_h"], st["soma_n"], st["soma_x"],
        st["axon_Sodium_h"], st["axon_Potassium_x"],
        st["dend_Ca2Plus"], st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
        st["g_CaL"], n_cells, n_simsteps, delta, sim_seconds,
        enable_gapjunctions, I_app, I_pulse10ms, record,
    )


def run_mp_trace(n_cells=30, sim_seconds=1.0, delta=0.01,
                 enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
                 g_CaL=None, seed=1981, record=True):
    """Run ONE sim through mp_sims' cross-sim machinery (a ProcessPoolExecutor
    worker) and return its trace, in the (n_simsteps, n_cells, 4) layout.

    This validates the across-sims pipeline end-to-end: the sim runs in a child
    process and its trace is pickled back. One worker is enough -- validation
    needs only the single seed-matched trace to compare, not mp_sims' across-sims
    throughput. The per-sim backbone is io_model_jit, so it should match the
    `jit` backend exactly. g_CaL is ignored (mp_sims randomizes per cell from the
    seed, as the baseline does at g_CaL=None)."""
    from concurrent.futures import ProcessPoolExecutor
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    with ProcessPoolExecutor(max_workers=1) as ex:
        v_trace = ex.submit(
            mp_sims.run_simulation, seed, n_cells, sim_seconds, delta,
            enable_gapjunctions, I_pulse10ms, True).result()  # return_trace=True
    return v_trace, n_simsteps


def run_mp_intra_trace(n_cells=30, sim_seconds=1.0, delta=0.01,
                       enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
                       g_CaL=None, seed=1981, record=True):
    """Run the intra-sim parallel backend (one sim's cells split across
    processes) and return its dense trace, in the (n_simsteps, n_cells, 4)
    layout.

    This validates the WITHIN-sim parallel decomposition: the shared-memory
    state + per-step Sd all-reduce must reproduce the same numerics. The global
    Sd is summed as (sum of per-worker partial sums) rather than one sequential
    pass, so it agrees with `jit` to ~machine epsilon, not bit-for-bit (the
    equality cross-check measures this in the clean early window)."""
    _, _, _, _, v_trace = mp_sims_intra.simulate_intra(
        n_cells, sim_seconds, delta, enable_gj=enable_gapjunctions,
        I_pulse10ms=I_pulse10ms, seed=seed, g_CaL=g_CaL,
        record=record, record_every=1)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    return v_trace, n_simsteps


# Registry of optimized backends selectable on the command line. Order here is
# the canonical print order. _NEEDS_WARMUP marks the njit-compiled ones. The two
# `mp*` entries are the multiprocessing schemes (across-sims and within-sim);
# they self-warm their own JIT kernels, so they are not in _NEEDS_WARMUP. This
# is purely a NUMERIC check -- for a fair speed comparison of the two schemes
# (throughput vs latency) see mp_bench.py.
BACKENDS = {
    "vec": run_vec_trace,
    "vec_jit": run_vec_jit_trace,
    "jit": run_jit_trace,
    "mp": run_mp_trace,
    "mp_intra": run_mp_intra_trace,
}
_NEEDS_WARMUP = {"vec_jit", "jit"}


def warmup_jit(backends=("vec_jit", "jit")):
    """Compile the given njit backends on a 2-step toy problem so later timed runs
    measure steady-state execution, not compilation (lab1 convention)."""
    for name in backends:
        if name in _NEEDS_WARMUP:
            BACKENDS[name](n_cells=2, sim_seconds=0.00002, delta=0.01, seed=0, record=False)


def _baseline_to_array(v_trace_list):
    """Stack the baseline's per-cell list into a (n_simsteps, n_cells, 4) array
    so it lines up with the vec backend's layout for elementwise comparison."""
    return np.stack(v_trace_list, axis=1)


def divergence_metrics(baseline_arr, vec_arr, delta):
    """Quantify how the Jacobi vec backend drifts from the Gauss-Seidel baseline.

    Both inputs are (n_simsteps, n_cells, 4) (V_soma, V_axon, V_dend, t). Returns
    per-compartment max/RMS voltage differences plus the divergence-onset step
    (first step where any cell's soma differs by > 1 mV).
    """
    comps = ("soma", "axon", "dend")
    d = np.abs(baseline_arr[:, :, :3] - vec_arr[:, :, :3])  # (steps, cells, 3)
    metrics = {}
    for c, name in enumerate(comps):
        metrics[name] = {
            "max_mV": float(d[:, :, c].max()),
            "rms_mV": float(np.sqrt(np.mean(d[:, :, c] ** 2))),
        }
    # Onset: first step where any cell's soma diverges past 1 mV.
    soma_step_max = d[:, :, 0].max(axis=1)
    over = np.argmax(soma_step_max > 1.0)
    metrics["onset_step"] = int(over) if soma_step_max.max() > 1.0 else None
    metrics["onset_ms"] = (over * delta) if soma_step_max.max() > 1.0 else None
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare optimized backend(s) against the io_model baseline. "
                    "With no arguments it runs the default setting on all backends.")
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS) + ["all"],
                        default=["all"], metavar="BACKEND",
                        help="Which optimized backend(s) to compare against the "
                             "baseline: any of " + ", ".join(BACKENDS) + ", or 'all' "
                             "(default: all).")
    # Defaults mirror the io_model.py baseline (1.0s, 30 cells, delta 0.01).
    # NB: the Gauss-Seidel baseline is slow at 1.0s -- pass e.g. --sim-seconds 0.2
    # for a quick check.
    parser.add_argument("--sim-seconds", type=float, default=1.0,
                        help="Simulated time in seconds (default 1.0).")
    parser.add_argument("--n-cells", type=int, default=30,
                        help="Cell population size (default 30).")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Time-step duration (default 0.01).")
    parser.add_argument("--seed", type=int, default=1981,
                        help="RNG seed for the shared initial state (default 1981).")
    args = parser.parse_args()

    # Resolve selection to the canonical registry order (dedup, honor 'all').
    if "all" in args.backends:
        selected = list(BACKENDS)
    else:
        selected = [b for b in BACKENDS if b in args.backends]

    kw = dict(n_cells=args.n_cells, sim_seconds=args.sim_seconds,
              delta=args.delta, seed=args.seed)

    import time
    warmup_jit(selected)  # compile only the njit backends we'll actually time

    # Baseline is always the reference everything is compared against.
    t = time.process_time()
    vb, n_simsteps = run_baseline_trace(**kw)
    tb = time.process_time() - t
    base_arr = _baseline_to_array(vb)

    # Run each selected backend (state building excluded from timing inside each).
    results = {}  # name -> (trace_array, wall_time)
    for name in selected:
        t = time.process_time()
        arr, _ = BACKENDS[name](**kw)
        results[name] = (arr, time.process_time() - t)

    # --- timing / speedup ---
    perf = "   ".join(f"{n} {results[n][1]:.3f}s ({tb / results[n][1]:.2f}x)" for n in selected)
    print(f"baseline {tb:.3f}s   {perf}   ({n_simsteps} steps x {args.n_cells} cells)")

    # --- divergence vs baseline (Jacobi vs Gauss-Seidel; measured, expected) ---
    print("divergence vs baseline (expected -- different integration scheme):")
    for name in selected:
        m = divergence_metrics(base_arr, results[name][0], args.delta)
        onset = ("no >1mV soma onset" if m["onset_step"] is None
                 else f"soma onset step {m['onset_step']} (t={m['onset_ms']:.2f} ms)")
        print(f"  {name}: soma max|d|={m['soma']['max_mV']:.4g} (rms {m['soma']['rms_mV']:.4g})  "
              f"axon max|d|={m['axon']['max_mV']:.4g}  dend max|d|={m['dend']['max_mV']:.4g} mV  | {onset}")

    # --- equality cross-check among the selected backends (same numerics) ---
    # All optimized backends use identical Jacobi+O(N) formulas, so they should
    # agree to machine epsilon; the early-window value is the clean check (libm
    # ulp differences get amplified by the chaotic dynamics over long runs).
    if len(selected) >= 2:
        ref = selected[0]
        ref_arr = results[ref][0]
        w = min(100, n_simsteps)
        print(f"equality vs {ref} (same numerics, expect ~machine epsilon):")
        for name in selected[1:]:
            d = np.abs(results[name][0][:, :, :3] - ref_arr[:, :, :3])
            print(f"  {name}: max|d| first {w} steps = {d[:w].max():.3e} mV"
                  f"  (whole run = {d.max():.3e} mV)")
