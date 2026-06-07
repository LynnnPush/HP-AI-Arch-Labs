#!/usr/bin/env python3
"""Plot n_cells sweep results for the IO network model.

Generates two figures from the CSVs in sweep_results/:

  1. throughput_vs_ncells.png
     Cell-step throughput (cell-steps/s) vs n_cells, log-x, with jit,
     jax_gpu and vec overlaid.

  2. speedup_vs_baseline.png
     Speedup = wall_baseline / wall_backend (>1 == faster than the
     pure-Python baseline) vs n_cells, for jit and jax_gpu. Plotted only
     over the small-N range where the baseline sweep has data points.

Usage:
    python3 plot_sweep.py                  # auto-pick latest CSV per backend
    python3 plot_sweep.py --show           # also display interactively

Reads only stdlib csv + numpy + matplotlib (no pandas required).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")  # overridden by --show
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "sweep_results")
OUT_DIR = os.path.join(HERE, "plots")

# Glob pattern + display style for each backend. The newest file matching
# the pattern (lexicographically, which == chronologically given the
# YYYYMMDD_HHMMSS suffix) is used.
# jit uses the dedicated n_cells sweep (rows scale up to 100000); the "2*"
# pins to the timestamp-suffixed file, excluding the knn8/noknn variants.
# The load step filters every file to swept_param == 'n_cells'.
BACKENDS = {
    "jit": dict(pattern="sweep_jit_n_cells_2*.csv", color="tab:blue", marker="o"),
    "jax_gpu": dict(pattern="sweep_jax_gpu_n_cells_*.csv", color="tab:green", marker="s"),
}
BASELINE_PATTERN = "sweep_baseline_*.csv"

# Throughput overlay and the speedup plot (vs baseline).
THROUGHPUT_BACKENDS = ["jit", "jax_gpu"]
SPEEDUP_BACKENDS = ["jit", "jax_gpu"]


def latest(pattern: str) -> str | None:
    """Newest file in RESULTS_DIR matching pattern, or None."""
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return matches[-1] if matches else None


def load_n_cells_sweep(path: str) -> dict[int, dict]:
    """Read a sweep CSV, keeping only ok rows from the n_cells sweep.

    Returns {n_cells: row_dict}. Files that also contain other swept
    params (e.g. the baseline 'all' sweep) are filtered to the rows where
    swept_param == 'n_cells'.
    """
    rows: dict[int, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("swept_param") != "n_cells":
                continue
            if row.get("status") != "ok":
                continue
            n = int(row["n_cells"])
            rows[n] = row
    return rows


def wall_of(row: dict) -> float:
    """Prefer the mean over repeats; fall back to the single wall time."""
    val = row.get("wall_time_mean_s") or row.get("wall_time_s")
    return float(val)


def plot_throughput(data: dict[str, dict[int, dict]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in THROUGHPUT_BACKENDS:
        rows = data.get(name)
        if not rows:
            print(f"  [throughput] skipping {name!r}: no data", file=sys.stderr)
            continue
        ns = sorted(rows)
        tput = [float(rows[n]["throughput_cellsteps_per_s"]) for n in ns]
        style = BACKENDS[name]
        ax.plot(ns, tput, marker=style["marker"], color=style["color"], label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_cells")
    ax.set_ylabel("throughput (cell-steps / s)")
    ax.set_title("Cell-step throughput vs n_cells")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_speedup(
    data: dict[str, dict[int, dict]],
    baseline: dict[int, dict],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    base_ns = set(baseline)
    for name in SPEEDUP_BACKENDS:
        rows = data.get(name)
        if not rows:
            print(f"  [speedup] skipping {name!r}: no data", file=sys.stderr)
            continue
        # Only the small-N range where the baseline actually has points.
        ns = sorted(base_ns & set(rows))
        if not ns:
            print(f"  [speedup] skipping {name!r}: no overlap with baseline", file=sys.stderr)
            continue
        speedup = [wall_of(baseline[n]) / wall_of(rows[n]) for n in ns]
        style = BACKENDS[name]
        ax.plot(ns, speedup, marker=style["marker"], color=style["color"], label=name)

    ax.axhline(1.0, color="gray", ls="--", lw=1, label="baseline (1×)")
    ax.set_xscale("log")
    ax.set_xlabel("n_cells")
    ax.set_ylabel("speedup vs baseline  (wall_baseline / wall_backend)")
    ax.set_title("Speedup vs pure-Python baseline (>1 = faster)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    data: dict[str, dict[int, dict]] = {}
    for name, spec in BACKENDS.items():
        path = latest(spec["pattern"])
        if path is None:
            print(f"warning: no file matching {spec['pattern']!r}", file=sys.stderr)
            continue
        data[name] = load_n_cells_sweep(path)
        print(f"{name}: {os.path.basename(path)}  ({len(data[name])} points)")

    base_path = latest(BASELINE_PATTERN)
    if base_path is None:
        print(f"warning: no baseline file matching {BASELINE_PATTERN!r}", file=sys.stderr)
        baseline: dict[int, dict] = {}
    else:
        baseline = load_n_cells_sweep(base_path)
        print(f"baseline: {os.path.basename(base_path)}  ({len(baseline)} points)")

    plot_throughput(data, os.path.join(OUT_DIR, "throughput_vs_ncells.png"))
    if baseline:
        plot_speedup(data, baseline, os.path.join(OUT_DIR, "speedup_vs_baseline.png"))
    else:
        print("skipping speedup plot: no baseline data", file=sys.stderr)

    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
