"""
Controlled profiling of the baseline io_model.

We replicate the __main__ state init (from sweep.py) but run a SHORT sim
(few thousand steps) so we can profile relative costs quickly. Relative
breakdown (per-compartment, gap-junction on/off, scaling with N) is what we
care about, not absolute wall time of a full 100k-step run.
"""
import time
import cProfile
import pstats
import io as _io
import numpy as np

import lab6_io_model.cpu.io_model as M


def build_state(n_cells, seed=1981):
    np.random.seed(seed)
    return {
        "g_CaL": np.random.normal(0.7, 0.1, n_cells),
        "V_soma": np.random.uniform(-70, -40, size=(n_cells,)),
        "soma_k": np.array([0.7423159] * n_cells),
        "soma_l": np.array([0.0321349] * n_cells),
        "soma_h": np.array([0.3596066] * n_cells),
        "soma_n": np.array([0.2369847] * n_cells),
        "soma_x": np.array([0.1] * n_cells),
        "V_axon": np.random.uniform(-70, -40, size=(n_cells,)),
        "axon_Sodium_h": np.array([0.9] * n_cells),
        "axon_Potassium_x": np.array([0.2369847] * n_cells),
        "V_dend": np.random.uniform(-70, -40, size=(n_cells,)),
        "dend_Ca2Plus": np.array([3.715] * n_cells),
        "dend_Calcium_r": np.array([0.0113] * n_cells),
        "dend_Potassium_s": np.array([0.0049291] * n_cells),
        "dend_Hcurrent_q": np.array([0.0337836] * n_cells),
    }


def run_sim(n_cells, n_steps, enable_gj, delta=0.01, record=False):
    """Run the baseline scalar loop for n_steps. Returns wall time (s)."""
    M.n_cells = n_cells
    M.delta = delta
    M.sim_seconds = 1.0
    M.enable_gapjunctions = enable_gj

    s = build_state(n_cells)
    at = -1  # don't record into trace (avoids needing a trace array)
    vtr = np.empty((1, 4))  # dummy, never written when at < 0
    t = 0.0

    tic = time.perf_counter()
    for _ in range(n_steps):
        for i in range(n_cells):
            M.update_soma(i, vtr, at, s["V_soma"], s["V_axon"], s["V_dend"],
                          s["soma_k"], s["soma_l"], s["soma_h"], s["soma_n"],
                          s["soma_x"], s["g_CaL"])
            M.update_axon(i, vtr, at, s["V_soma"], s["V_axon"],
                          s["axon_Sodium_h"], s["axon_Potassium_x"])
            M.update_dend(i, vtr, at, t, s["V_soma"], s["V_dend"],
                          s["dend_Ca2Plus"], s["dend_Calcium_r"],
                          s["dend_Potassium_s"], s["dend_Hcurrent_q"])
        t += delta
    return time.perf_counter() - tic


# ---------------------------------------------------------------------------
# 1) cProfile breakdown by compartment at baseline N=30
# ---------------------------------------------------------------------------
N = 30
STEPS = 2000
print(f"=== cProfile: N={N}, steps={STEPS}, GJ=on ===")
pr = cProfile.Profile()
pr.enable()
run_sim(N, STEPS, enable_gj=True)
pr.disable()
sbuf = _io.StringIO()
ps = pstats.Stats(pr, stream=sbuf).sort_stats("cumulative")
ps.print_stats(12)
print(sbuf.getvalue())

# ---------------------------------------------------------------------------
# 2) Gap-junction on/off at N=30 (isolates the O(N^2) term's share)
# ---------------------------------------------------------------------------
print(f"=== GJ on vs off, N={N}, steps={STEPS} ===")
for gj in (True, False):
    # warm + best of 2
    ts = [run_sim(N, STEPS, enable_gj=gj) for _ in range(2)]
    print(f"  GJ={'on ' if gj else 'off'}: {min(ts):.3f} s  "
          f"({min(ts)/STEPS*1e3:.3f} ms/step)")

# ---------------------------------------------------------------------------
# 3) Scaling with N (does it look linear or quadratic at these sizes?)
# ---------------------------------------------------------------------------
print(f"=== scaling with N (steps={STEPS}, GJ=on) ===")
base = None
for n in (1, 2, 10, 30, 60):
    ts = [run_sim(n, STEPS, enable_gj=True) for _ in range(2)]
    tmin = min(ts)
    if base is None:
        base = tmin
    print(f"  N={n:>3}: {tmin:.3f} s  | per-cell-step={tmin/STEPS/n*1e6:.2f} us "
          f"| total/N={tmin/n:.4f}")
