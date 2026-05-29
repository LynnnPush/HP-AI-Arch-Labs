#
# Parameter sweep + performance benchmarking harness for io_model.py
#
# Adapted for the CESE5040 course.
#
# This script imports the Inferior-Olive (de Gruijl) model from io_model.py,
# overrides its module-level parameters, re-runs the simulation, and records
# performance metrics (latency, throughput, wall/CPU time, real-time factor).
#
# The USER chooses *which* parameters to sweep on the command line; the sweep
# RANGES are defined here in PARAM_SPECS.
#
# Examples
# --------
#   # list the sweepable parameters and their ranges
#   python3 sweep.py --list
#
#   # sweep a single parameter
#   python3 sweep.py n_cells
#
#   # sweep several parameters (each varied independently, others held at baseline)
#   python3 sweep.py n_cells delta I_pulse10ms
#
#   # sweep everything
#   python3 sweep.py all
#
#   # repeat every run 3x and keep the best (lowest) wall time
#   python3 sweep.py n_cells --repeats 3
#

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

# Import the model under test. Importing only defines the functions / default
# globals; the simulation in its __main__ block does NOT run on import.
import io_model


# ---------------------------------------------------------------------------
# Baseline configuration. Every sweep varies ONE parameter across its range
# while holding the rest at these baseline values (with documented exceptions).
# ---------------------------------------------------------------------------
BASELINE = {
    "sim_seconds": 1.0,
    "delta": 0.01,
    "n_cells": 30,
    "enable_gapjunctions": True,
    "I_pulse10ms": 2.0,
    # g_CaL is None  -> randomized per-cell as in io_model line 208.
    # g_CaL is a float -> every cell gets that constant conductance.
    "g_CaL": None,
}


# ---------------------------------------------------------------------------
# Sweep definitions: the value range for each parameter and any baseline
# overrides that the assignment requires for that particular sweep.
# ---------------------------------------------------------------------------
PARAM_SPECS = {
    "sim_seconds": {
        "values": [0.5, 1.0, 1.5, 2.0],
        "overrides": {},
        "help": "Simulated brain time in seconds.",
    },
    "delta": {
        "values": [0.01, 0.03, 0.05],
        "overrides": {},
        "help": "Time-step duration in seconds.",
    },
    "n_cells": {
        "values": [1, 2, 10, 30],
        "overrides": {},
        "help": "Cell population size.",
    },
    "enable_gapjunctions": {
        "values": [True, False],
        # GJ contribution is only meaningful for a populated network.
        "overrides": {"n_cells": 30},
        "help": "Enable/ignore gap-junction contribution (n_cells fixed at 30).",
    },
    "g_CaL": {
        "values": [0.7, 1.8, 2.0],
        # Constant g_CaL replaces the per-cell randomization; single cell.
        "overrides": {"n_cells": 1},
        "help": "Low-voltage-activated Ca conductance, constant (n_cells fixed at 1).",
    },
    "I_pulse10ms": {
        "values": [2.0, 5.0, 8.0, 10.0],
        "overrides": {},
        "help": "Amplitude of the 10 ms square current pulse applied to the population.",
    },
}


def build_initial_state(n_cells, g_CaL, seed):
    """Replicate io_model.__main__ state initialisation for `n_cells` cells.

    g_CaL: None -> randomized per cell (np.random.normal(0.7, 0.1, n_cells))
           float -> constant conductance for every cell.
    """
    if seed is not None:
        np.random.seed(seed)

    if g_CaL is None:
        g_CaL_arr = np.random.normal(0.7, 0.1, n_cells)
    else:
        g_CaL_arr = np.full(n_cells, float(g_CaL))

    state = {
        "g_CaL": g_CaL_arr,
        # Soma state
        "V_soma": np.random.uniform(low=-70, high=-40, size=(n_cells,)),
        "soma_k": np.array([0.7423159] * n_cells),
        "soma_l": np.array([0.0321349] * n_cells),
        "soma_h": np.array([0.3596066] * n_cells),
        "soma_n": np.array([0.2369847] * n_cells),
        "soma_x": np.array([0.1] * n_cells),
        # Axon state
        "V_axon": np.random.uniform(low=-70, high=-40, size=(n_cells,)),
        "axon_Sodium_h": np.array([0.9] * n_cells),
        "axon_Potassium_x": np.array([0.2369847] * n_cells),
        # Dend state
        "V_dend": np.random.uniform(low=-70, high=-40, size=(n_cells,)),
        "dend_Ca2Plus": np.array([3.715] * n_cells),
        "dend_Calcium_r": np.array([0.0113] * n_cells),
        "dend_Potassium_s": np.array([0.0049291] * n_cells),
        "dend_Hcurrent_q": np.array([0.0337836] * n_cells),
    }
    return state


def run_once(cfg, seed):
    """Run a single simulation with the given config dict and return metrics.

    The model's update functions read several module-level globals, so we push
    the config into the io_model namespace before running.
    """
    sim_seconds = cfg["sim_seconds"]
    delta = cfg["delta"]
    n_cells = cfg["n_cells"]

    # Push parameters into the model's global namespace (the update_* functions
    # read these directly).
    io_model.sim_seconds = sim_seconds
    io_model.delta = delta
    io_model.n_cells = n_cells
    io_model.enable_gapjunctions = cfg["enable_gapjunctions"]
    io_model.I_pulse10ms = cfg["I_pulse10ms"]

    st = build_initial_state(n_cells, cfg["g_CaL"], seed)

    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    v_trace = [np.empty((n_simsteps, 4)) for _ in range(n_cells)]
    t = 0.0

    wall_tic = time.perf_counter()
    cpu_tic = time.process_time()

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
            v_trace[i_cell][i_epoch, -1] = t
        t += delta

    wall_time = time.perf_counter() - wall_tic
    cpu_time = time.process_time() - cpu_tic

    total_cell_steps = n_simsteps * n_cells

    metrics = {
        "n_simsteps": n_simsteps,
        "total_cell_steps": total_cell_steps,
        "wall_time_s": wall_time,
        "cpu_time_s": cpu_time,
        # Throughput
        "throughput_steps_per_s": n_simsteps / wall_time if wall_time > 0 else float("nan"),
        "throughput_cellsteps_per_s": total_cell_steps / wall_time if wall_time > 0 else float("nan"),
        # Latency (both in microseconds)
        "latency_us_per_step": (wall_time / n_simsteps * 1e6) if n_simsteps else float("nan"),
        "latency_us_per_cellstep": (wall_time / total_cell_steps * 1e6) if total_cell_steps else float("nan"),
        # How much faster/slower than biological real time (sim_s of brain time
        # simulated per wall-clock second).
        "realtime_factor": sim_seconds / wall_time if wall_time > 0 else float("nan"),
    }
    return metrics


def run_repeated(cfg, repeats, seed):
    """Run a config `repeats` times; keep the fastest (lowest wall time) run."""
    best = None
    all_wall = []
    for _ in range(repeats):
        m = run_once(cfg, seed)
        all_wall.append(m["wall_time_s"])
        if best is None or m["wall_time_s"] < best["wall_time_s"]:
            best = m
    best["repeats"] = repeats
    best["wall_time_mean_s"] = float(np.mean(all_wall))
    best["wall_time_std_s"] = float(np.std(all_wall))
    return best


def make_config(param, value):
    """Build a full config for sweeping `param` to `value`, applying overrides."""
    cfg = dict(BASELINE)
    cfg.update(PARAM_SPECS[param]["overrides"])
    cfg[param] = value
    return cfg


# Column order for the CSV / printed table.
CONFIG_KEYS = ["sim_seconds", "delta", "n_cells", "enable_gapjunctions", "I_pulse10ms", "g_CaL"]
METRIC_KEYS = [
    "n_simsteps", "total_cell_steps", "wall_time_s", "cpu_time_s",
    "throughput_steps_per_s", "throughput_cellsteps_per_s",
    "latency_us_per_step", "latency_us_per_cellstep", "realtime_factor",
    "repeats", "wall_time_mean_s", "wall_time_std_s",
]


def print_table(rows):
    """Pretty-print a compact summary table to stdout."""
    cols = ["swept_param", "swept_value", "n_cells", "delta", "sim_seconds",
            "wall_time_s", "throughput_cellsteps_per_s", "latency_us_per_step",
            "realtime_factor"]
    widths = {c: len(c) for c in cols}
    fmt_rows = []
    for r in rows:
        fr = {}
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.4g}"
            fr[c] = str(v)
            widths[c] = max(widths[c], len(fr[c]))
        fmt_rows.append(fr)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for fr in fmt_rows:
        print("  ".join(fr[c].ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(
        description="Sweep io_model parameters and report performance metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "params", nargs="*",
        help="Parameters to sweep: any of "
             + ", ".join(PARAM_SPECS) + ", or 'all'.",
    )
    parser.add_argument("--list", action="store_true",
                        help="List sweepable parameters and their ranges, then exit.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each run N times; keep the fastest (default: 1).")
    parser.add_argument("--seed", type=int, default=1981,
                        help="RNG seed for reproducible initial state (default: 1981). "
                             "Use -1 for non-deterministic runs.")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "sweep_results"),
                        help="Directory to store result files.")
    args = parser.parse_args()

    if args.list:
        print("Sweepable parameters (ranges defined in sweep.py):\n")
        for name, spec in PARAM_SPECS.items():
            ov = spec["overrides"]
            ov_str = f"  [overrides: {ov}]" if ov else ""
            print(f"  {name:<20} {spec['values']}{ov_str}")
            print(f"  {'':<20} {spec['help']}\n")
        return

    if not args.params:
        parser.error("no parameters given. Pass one or more of: "
                     + ", ".join(PARAM_SPECS) + ", or 'all' (see --list).")

    if "all" in args.params:
        selected = list(PARAM_SPECS)
    else:
        selected = []
        for p in args.params:
            if p not in PARAM_SPECS:
                parser.error(f"unknown parameter '{p}'. Choose from: "
                             + ", ".join(PARAM_SPECS) + ", or 'all'.")
            if p not in selected:
                selected.append(p)

    seed = None if args.seed == -1 else args.seed

    rows = []
    print(f"Sweeping: {', '.join(selected)}  (repeats={args.repeats}, seed={seed})\n")
    for param in selected:
        spec = PARAM_SPECS[param]
        for value in spec["values"]:
            cfg = make_config(param, value)
            metrics = run_repeated(cfg, args.repeats, seed)

            row = {"swept_param": param, "swept_value": value}
            row.update({k: cfg[k] for k in CONFIG_KEYS})
            row.update({k: metrics.get(k) for k in METRIC_KEYS})
            rows.append(row)

            print(f"[{param}={value}] "
                  f"wall={metrics['wall_time_s']:.3f}s  "
                  f"throughput={metrics['throughput_cellsteps_per_s']:.3g} cell-steps/s  "
                  f"latency={metrics['latency_us_per_step']:.4g} us/step  "
                  f"rt_factor={metrics['realtime_factor']:.3g}x")

    print()
    print_table(rows)

    # ---- Persist results ----
    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "all" if set(selected) == set(PARAM_SPECS) else "_".join(selected)
    base = os.path.join(args.outdir, f"sweep_{tag}_{stamp}")

    fieldnames = ["swept_param", "swept_value"] + CONFIG_KEYS + METRIC_KEYS
    csv_path = base + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    meta = {
        "timestamp": stamp,
        "swept_params": selected,
        "repeats": args.repeats,
        "seed": seed,
        "baseline": BASELINE,
        "param_specs": {k: {"values": v["values"], "overrides": v["overrides"]}
                        for k, v in PARAM_SPECS.items()},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "results": rows,
    }
    json_path = base + ".json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nResults written to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
