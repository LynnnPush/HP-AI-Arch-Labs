#
# Spike-based numerical-quality analysis for the io_model backends (Goal C:
# "numerical quality beyond divergence").
#
# WHY THIS EXISTS -- the chaos caveat
# -----------------------------------
# The IO network is *chaotic*: the pointwise V-trace L2 error between two equally
# correct backends grows EXPONENTIALLY from floating-point rounding alone. So a
# rising `allclose`/L2 residual between, say, the CPU `jit` and the GPU `jax_gpu`
# backend is EXPECTED -- it is the Lyapunov amplification of ulp-level differences,
# NOT by itself evidence of a bug. validate.py already measures that divergence;
# this script does NOT re-litigate it.
#
# Instead we compare backends with a rate-level metric that stays stable even when
# the raw traces have fully decorrelated: spike COUNT / mean firing RATE per cell
# (plus mean ISI as the rhythm period). For functionally identical math these match
# the reference to ~0 relative error even after the L2 has blown up; a divergence
# HERE is a real bug, unlike a rising L2. Timing structure is conveyed by the two
# figures (raster + V_soma overlay) rather than a scalar spike-distance, which would
# add a parameter (q / tau) we'd have to justify.
#
# Same initial state: every trace comes from validate.run_*_trace, which feeds the
# shared sweep.build_initial_state(seed) to all backends -- so the comparison is
# apples-to-apples. We default the reference to `jit` and the backend under test to
# `jax`, because validate forces BOTH onto f64 + all-to-all coupling (run_jax_trace
# sets f64+use_knn=False, run_jit_trace defaults to all-to-all). Do NOT compare an
# f32/kNN run against an f64/all-to-all one -- that mismatch is a config error, not
# a backend bug.
#
# Spike detection (the IO-specific catch): the soma carries a subthreshold
# oscillation (STO) whose crests must NOT be counted as spikes. We run
# scipy.signal.find_peaks on raw column 0 (V_soma) with a `height` threshold set
# ABOVE the STO crest band (default -45 mV; inspect with `--inspect` to place it in
# the gap for your config), or a `prominence` criterion instead, plus a `distance`
# refractory window so one broad depolarization is not double-counted. Picking the
# threshold from data is the whole game -- see `--inspect`.
#
# Usage (repo root + this folder on PYTHONPATH, like validate.py/sweep.py):
#   PYTHONPATH="..:." py -3 spikes.py                       # jit vs jax, default cfg
#   PYTHONPATH="..:." py -3 spikes.py --inspect             # peak-height histogram
#   PYTHONPATH="..:." py -3 spikes.py --backends jit baseline --reference jit
#   PYTHONPATH="..:." py -3 spikes.py --v-th -44 --refractory-ms 5 --save
#
import argparse
import csv
import os

import numpy as np
from scipy.signal import find_peaks

import validate


# ---------------------------------------------------------------------------
# Trace acquisition -- reuse validate's seed-matched backends verbatim.
# ---------------------------------------------------------------------------
def run_jit_knn_trace(k=8, n_cells=30, sim_seconds=1.0, delta=0.01,
                      enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
                      g_CaL=None, seed=1981, record=True):
    """Like validate.run_jit_trace but with LOCAL k-nearest-neighbour gap coupling
    (the sweep/production default), instead of validate's all-to-all.

    Returns (steps, cells, 4). NOTE the coupling caveat: this is a DIFFERENT
    physical model from the all-to-all baseline -- each cell couples only to its k
    nearest neighbours -- so spike differences vs an all-to-all reference mix the
    integration-scheme drift with a genuine topology change. Compare kNN to kNN
    (or all-to-all to all-to-all) for a clean numerics check; use this only when
    you deliberately want to see the topology's effect."""
    from sweep import build_initial_state
    import lab6_io_model.cpu.io_model_jit as io_model_jit
    st = build_initial_state(n_cells, g_CaL, seed)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    neighbours = io_model_jit.build_neighbours(n_cells, k=k, seed=seed)
    arr, _ = io_model_jit.simulate(
        st["V_soma"], st["V_axon"], st["V_dend"],
        st["soma_k"], st["soma_l"], st["soma_h"], st["soma_n"], st["soma_x"],
        st["axon_Sodium_h"], st["axon_Potassium_x"],
        st["dend_Ca2Plus"], st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
        st["g_CaL"], n_cells, n_simsteps, delta, sim_seconds,
        enable_gapjunctions, I_app, I_pulse10ms, record,
        use_knn=True, neighbours=neighbours)
    return np.asarray(arr), n_simsteps


def get_trace(name, kw, knn=False, k=8):
    """Return one backend's voltage trace as a (n_simsteps, n_cells, 4) array.

    `name` is "baseline" or any key in validate.BACKENDS. All backends start from
    the identical seeded state (sweep.build_initial_state), so the only differences
    are the integration scheme / precision / hardware -- exactly what we want to
    measure. The baseline returns a per-cell list, which we stack to match the
    (steps, cells, 4) layout of the optimized backends.

    With knn=True the `jit` backend uses local k-NN coupling (see
    run_jit_knn_trace); every other backend stays all-to-all (baseline only has
    all-to-all; validate's run_jax_trace forces all-to-all for its equality check).
    """
    if name == "baseline":
        vb, _ = validate.run_baseline_trace(**kw)
        return validate._baseline_to_array(vb)
    if name == "jit" and knn:
        arr, _ = run_jit_knn_trace(k=k, **kw)
        return arr
    arr, _ = validate.BACKENDS[name](**kw)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Spike detection on the soma trace.
# ---------------------------------------------------------------------------
def detect_spikes(trace, delta, v_th=-45.0, prominence=None, refractory_ms=5.0):
    """Detect spikes per cell on raw column 0 (V_soma) of a (steps, cells, 4) trace.

    Returns a list of length n_cells; entry i is a 1-D array of spike TIMES in ms
    (read from the trace's own time column so it stays exact under strided/recorded
    traces). Detection is scipy.signal.find_peaks with:
      * height=v_th  -- a hard voltage floor placed ABOVE the STO crest band so
        oscillation peaks are not counted (set to None to rely on prominence only);
      * prominence    -- optional alternative/added criterion (mV a peak must rise
        above its surrounding troughs); robust when the STO/spike gap is narrow;
      * distance=refractory steps -- a refractory window (ms -> steps) so a single
        broad depolarization yields one spike, not several.
    """
    n_cells = trace.shape[1]
    refractory_steps = max(1, int(round(refractory_ms / delta)))
    trains = []
    for i in range(n_cells):
        v = trace[:, i, 0]
        t = trace[:, i, 3]
        peaks, _ = find_peaks(v, height=v_th, prominence=prominence,
                              distance=refractory_steps)
        trains.append(t[peaks])
    return trains


def peak_height_histogram(trace, bins=30):
    """Diagnostic for choosing v_th: pool every local soma maximum across cells and
    return (counts, bin_edges). The STO crests form a dense low band; true spikes
    are the sparse high tail -- put v_th in the gap between them."""
    heights = []
    for i in range(trace.shape[1]):
        peaks, _ = find_peaks(trace[:, i, 0])
        heights.append(trace[peaks, i, 0])
    heights = np.concatenate(heights) if heights else np.array([])
    return np.histogram(heights, bins=bins)


# ---------------------------------------------------------------------------
# Rate-level metrics (coarse; should be near-exact between correct backends).
# ---------------------------------------------------------------------------
def rate_metrics(trains, sim_seconds):
    """Per-cell + aggregate rate statistics for a list of spike-time trains.

    Returns a dict with per-cell arrays (count, rate_hz, mean_isi_ms) and scalar
    aggregates (total_count, mean_rate_hz, mean_isi_ms over all inter-spike gaps).
    Mean ISI is NaN for cells with <2 spikes (no interval defined there).
    """
    counts = np.array([len(t) for t in trains], dtype=float)
    rate_hz = counts / sim_seconds
    mean_isi = np.array([np.mean(np.diff(t)) if len(t) >= 2 else np.nan
                         for t in trains])
    all_isis = np.concatenate([np.diff(t) for t in trains if len(t) >= 2]) \
        if any(len(t) >= 2 for t in trains) else np.array([])
    return {
        "count": counts,
        "rate_hz": rate_hz,
        "mean_isi_ms": mean_isi,
        "total_count": float(counts.sum()),
        "mean_rate_hz": float(rate_hz.mean()),
        "mean_isi_ms_global": float(all_isis.mean()) if all_isis.size else float("nan"),
    }


def compare_trains(ref_trains, test_trains, sim_seconds):
    """Compare a backend's spike trains against the reference's (rate-level).

    Relative error of total spike count and of mean firing rate -- should be ~0 for
    functionally identical math even after the raw V-trace L2 has blown up; a
    divergence here is a real bug. (Timing structure is left to the figures.)
    Returns a flat dict of scalars. Note count_rel_err == rate_rel_err: mean rate
    is just total_count / (N * sim_seconds), so the shared constant cancels -- count
    is the raw form, rate the run-length-normalised (Hz) form of one quantity.
    """
    ref = rate_metrics(ref_trains, sim_seconds)
    test = rate_metrics(test_trains, sim_seconds)

    def rel(a, b):
        return abs(a - b) / b if b else (0.0 if a == 0 else float("inf"))

    return {
        "ref_total": ref["total_count"],
        "test_total": test["total_count"],
        "count_rel_err": rel(test["total_count"], ref["total_count"]),
        "rate_rel_err": rel(test["mean_rate_hz"], ref["mean_rate_hz"]),
    }


# ---------------------------------------------------------------------------
# Raw data: persist the detected spike trains so the analysis is reproducible
# without re-running the (slow) sims.
# ---------------------------------------------------------------------------
def write_spike_csv(trains_by_backend, outpath):
    """Write every detected spike to a tidy long-format CSV.

    Columns: backend, cell, spike_index (ordinal within the cell), spike_time_ms.
    One row per spike; this is the raw spike trains behind the rate metrics and the
    figures, so downstream analysis (or a different distance metric) can reload it
    without re-simulating. Cells with no spikes contribute no rows."""
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "cell", "spike_index", "spike_time_ms"])
        for name, trains in trains_by_backend.items():
            for cell, times in enumerate(trains):
                for k, t in enumerate(times):
                    w.writerow([name, cell, k, f"{t:.6f}"])
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Figures: visualize the spikes for the baseline and each backend.
# ---------------------------------------------------------------------------
def plot_rasters(trains_by_backend, outpath=None, show=True):
    """Stacked spike-raster panels, one per backend (top = first selected).

    Each panel scatters detected spike times (x, ms) against cell index (y); the
    shared x-axis makes timing differences between backends read off vertically.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(trains_by_backend)
    fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                             figsize=(10, 1.6 * len(names) + 1))
    axes = axes[:, 0]
    for ax, name in zip(axes, names):
        trains = trains_by_backend[name]
        for i, t in enumerate(trains):
            if len(t):
                ax.scatter(t, np.full(len(t), i), marker="|", s=40,
                           color="C0", linewidths=1.0)
        total = sum(len(t) for t in trains)
        ax.set_ylabel(f"{name}\n(cell)")
        ax.set_ylim(-1, len(trains))
        ax.text(0.99, 0.92, f"{total} spikes", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")
    axes[-1].set_xlabel("time (ms)")
    axes[0].set_title("Soma spike rasters by backend (same seeded initial state)")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=130)
        print(f"  wrote {outpath}")
    if show:
        plt.show()
    plt.close(fig)


def plot_trace_with_spikes(traces_by_backend, trains_by_backend, cell, v_th,
                           outpath=None, show=True):
    """Overlay the chosen cell's V_soma trace + detected spike markers per backend.

    This is the visual sanity check on detection: the dashed line is v_th, the STO
    band sits below it, and the markers should land on the supra-threshold
    depolarizations only -- not on oscillation crests."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(traces_by_backend)
    fig, axes = plt.subplots(len(names), 1, sharex=True, sharey=True,
                             squeeze=False, figsize=(10, 1.8 * len(names) + 1))
    axes = axes[:, 0]
    for ax, name in zip(axes, names):
        tr = traces_by_backend[name]
        t = tr[:, cell, 3]
        v = tr[:, cell, 0]
        ax.plot(t, v, color="gray", lw=0.6)
        st = trains_by_backend[name]
        if cell < len(st) and len(st[cell]):
            # mark each detected spike at v_th for a clean row of ticks
            ax.scatter(st[cell], np.full(len(st[cell]), v_th), marker="v",
                       color="C3", s=30, zorder=3)
        if v_th is not None:
            ax.axhline(v_th, color="C3", ls="--", lw=0.7)
        ax.set_ylabel(f"{name}\nV_soma")
    axes[-1].set_xlabel("time (ms)")
    axes[0].set_title(f"V_soma + detected spikes, cell {cell}")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=130)
        print(f"  wrote {outpath}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Spike-based numerical-quality analysis of the io_model "
                    "backends. Compares a backend (default jax) against a reference "
                    "(default jit) using rate metrics (count / rate / ISI) that stay "
                    "stable under the model's chaotic L2 divergence, and plots spike "
                    "rasters + a V_soma overlay. See the module docstring for the "
                    "chaos caveat.")
    parser.add_argument("--backends", nargs="+",
                        choices=["baseline"] + list(validate.BACKENDS),
                        default=["jit", "jax"], metavar="BACKEND",
                        help="Backends to analyze/plot (default: jit jax). "
                             "Choices: baseline, " + ", ".join(validate.BACKENDS) + ".")
    parser.add_argument("--reference", default="jit",
                        choices=["baseline"] + list(validate.BACKENDS),
                        help="Reference backend each other is compared against "
                             "(default jit). Forced into the analysis set if not "
                             "already selected.")
    parser.add_argument("--sim-seconds", type=float, default=1.0,
                        help="Simulated time in seconds (default 1.0).")
    parser.add_argument("--n-cells", type=int, default=30,
                        help="Cell population size (default 30).")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Time-step duration in ms (default 0.01).")
    parser.add_argument("--seed", type=int, default=1981,
                        help="RNG seed for the shared initial state (default 1981).")
    # detection knobs
    parser.add_argument("--v-th", type=float, default=-45.0,
                        help="Spike voltage threshold in mV, placed above the STO "
                             "crest band (default -45). Use --inspect to choose.")
    parser.add_argument("--prominence", type=float, default=None,
                        help="Optional prominence (mV) criterion instead of/added "
                             "to --v-th; robust when the STO/spike gap is narrow.")
    parser.add_argument("--refractory-ms", type=float, default=5.0,
                        help="Refractory window in ms so one depolarization yields "
                             "one spike (default 5).")
    # coupling
    parser.add_argument("--knn", action="store_true",
                        help="Use local k-NN gap coupling for the `jit` backend "
                             "(sweep default) instead of all-to-all. NOTE this is a "
                             "different physical model from the all-to-all baseline, "
                             "so the spike difference mixes scheme drift + topology.")
    parser.add_argument("--k", type=int, default=8,
                        help="Neighbour count for --knn (default 8).")
    # output
    parser.add_argument("--inspect", action="store_true",
                        help="Print the pooled peak-height histogram for the "
                             "reference backend (to choose --v-th) and exit.")
    parser.add_argument("--cell", type=int, default=None,
                        help="Cell index for the trace+spikes figure (default: the "
                             "reference's most-active cell).")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to --outdir instead of showing them "
                             "(headless-friendly; e.g. on the GPU server).")
    parser.add_argument("--outdir", default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "spike_results"),
                        help="Directory for saved figures/summary (default "
                             "lab6_io_model/spike_results).")
    args = parser.parse_args()

    # Build the analysis set: selected backends + the reference, in a stable order
    # (reference first, then the rest as given), deduped.
    selected = [args.reference] + [b for b in args.backends if b != args.reference]

    kw = dict(n_cells=args.n_cells, sim_seconds=args.sim_seconds,
              delta=args.delta, seed=args.seed)

    # --inspect: histogram of soma local-maxima heights to place v_th, then exit.
    # Only the reference is needed here, so warm just that backend (avoids pulling
    # in JAX on a CPU-only host when the reference is e.g. jit).
    if args.inspect:
        validate.warmup_jit([args.reference] if args.reference in validate._NEEDS_WARMUP else [])
        if args.reference == "jax":
            validate.run_jax_trace(**kw)
        ref_trace = get_trace(args.reference, kw, knn=args.knn, k=args.k)
        counts, edges = peak_height_histogram(ref_trace)
        print(f"peak-height histogram for '{args.reference}' "
              f"(N={args.n_cells}, {args.sim_seconds}s) -- put --v-th in the gap "
              f"between the dense STO band and the sparse spike tail:")
        for c, e in zip(counts, edges):
            bar = "#" * int(c)
            print(f"  {e:7.2f} mV | {c:4d} {bar}")
        return

    # Warm the njit backends (and JAX at the real shape) so the FIRST recorded run
    # isn't a compile -- correctness only, but it keeps the traces clean.
    validate.warmup_jit([b for b in selected if b in validate._NEEDS_WARMUP])
    if "jax" in selected:
        validate.run_jax_trace(**kw)

    if args.knn and "jax" in selected:
        print("note: --knn applies to the `jit` backend only; `jax` stays "
              "all-to-all (validate.run_jax_trace forces it).")

    # Run every selected backend once and detect its spikes.
    traces = {name: get_trace(name, kw, knn=args.knn, k=args.k) for name in selected}
    trains = {name: detect_spikes(t, args.delta, v_th=args.v_th,
                                  prominence=args.prominence,
                                  refractory_ms=args.refractory_ms)
              for name, t in traces.items()}

    # --- the chaos caveat, stated up front so the numbers below are read right ---
    print("NOTE: this model is chaotic -- the raw V-trace L2 between two correct "
          "backends grows exponentially from fp rounding and is NOT a bug signal. "
          "Use the rate metrics below (+ the figures) instead.\n")

    # --- per-backend rate summary ---
    coupling = f"kNN(k={args.k}) on jit" if args.knn else "all-to-all"
    print(f"spike detection: v_th={args.v_th} mV prominence={args.prominence} "
          f"refractory={args.refractory_ms} ms  | coupling: {coupling}\n")
    print("per-backend rate metrics:")
    for name in selected:
        m = rate_metrics(trains[name], args.sim_seconds)
        print(f"  {name:9s}: total={m['total_count']:.0f}  "
              f"mean_rate={m['mean_rate_hz']:.3f} Hz  "
              f"mean_ISI={m['mean_isi_ms_global']:.2f} ms")

    # --- backend vs reference (rate rel-err ~0 expected even after L2 blow-up) ---
    others = [b for b in selected if b != args.reference]
    if others:
        print(f"\nvs reference '{args.reference}' "
              f"(rate rel-err should be ~0 even when raw L2 has blown up):")
        for name in others:
            c = compare_trains(trains[args.reference], trains[name], args.sim_seconds)
            print(f"  {name:9s}: count {c['ref_total']:.0f}->{c['test_total']:.0f} "
                  f"(rel {c['count_rel_err']:.2e})  rate rel {c['rate_rel_err']:.2e}")

    # --- figures ---
    if args.cell is not None:
        cell = args.cell
    else:
        ref_counts = [len(t) for t in trains[args.reference]]
        cell = int(np.argmax(ref_counts)) if ref_counts else 0

    if args.save:
        os.makedirs(args.outdir, exist_ok=True)
        stamp = f"N{args.n_cells}_{args.sim_seconds}s_seed{args.seed}"
        if args.knn:
            stamp += f"_knn{args.k}"
        raster_path = os.path.join(args.outdir, f"raster_{stamp}.png")
        trace_path = os.path.join(args.outdir, f"trace_cell{cell}_{stamp}.png")
        # Raw spike trains behind the metrics/figures, for reproducible re-analysis.
        print("\nraw data:")
        write_spike_csv(trains, os.path.join(args.outdir, f"spikes_{stamp}.csv"))
    else:
        raster_path = trace_path = None

    print("\nfigures:")
    plot_rasters(trains, outpath=raster_path, show=not args.save)
    plot_trace_with_spikes(traces, trains, cell, args.v_th,
                           outpath=trace_path, show=not args.save)


if __name__ == "__main__":
    main()
