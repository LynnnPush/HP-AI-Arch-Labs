# Lab 6 — CPU Backend Optimization of the Inferior-Olive Model

This lab takes the reference Inferior-Olive (de Gruijl) neuron simulator and
speeds it up on the CPU, then scales it out across processes. The model
integrates `n_cells` neurons, each with 3 compartments (soma / axon / dendrite)
and ~15 state variables, coupled through dendritic gap junctions.

There are **three optimized single-sim backends** plus a **multiprocessing**
driver, all validated against the untouched reference.

---

## Files

| File | What it is |
|------|------------|
| `io_model.py` | **Reference** baseline (unchanged). Pure-Python scalar loops, Gauss-Seidel update order, O(N²) all-to-all gap junction. The correctness/standard reference. |
| `io_model_vec.py` | **vec** — vectorized NumPy over all cells. Jacobi double-buffer + O(N) closed-form gap junction. |
| `io_model_jit.py` | **jit** — scalar per-cell loops compiled with Numba `@njit`. Same Jacobi + O(N) numerics as `vec`. Fastest single-sim backend. |
| `io_model_vec_jit.py` | **vec_jit** — `@njit` applied to the *vectorized* loop. Same numerics again; shows the cost of array temporaries under JIT. |
| `validate.py` | Runs the baseline + chosen backends from one seeded state; reports speedup, divergence-vs-baseline, and machine-epsilon equality among the optimized backends. |
| `sweep.py` | Parameter-sweep / performance harness for the **baseline** (wall/CPU time, throughput, latency, real-time factor). |
| `profile_io.py` | `cProfile` breakdown of the baseline (per-compartment cost, gap-junction on/off, scaling with N). |
| `mp_sims.py` | Multiprocessing across **independent** sims, using the `jit` backend as the per-sim backbone. |
| `sweep_results/` | CSV/JSON output from `sweep.py`. |

---

## Key concept: Jacobi vs Gauss-Seidel

The baseline updates cells **sequentially** within a timestep, so the axon/dend
of a cell see the *freshly updated* soma voltage, and the gap junction sees
already-updated neighbours (**Gauss-Seidel**).

All three optimized backends compute every derivative from a **start-of-step
snapshot** and write the three coupled voltages together (**Jacobi**), which is
what enables vectorization and the O(N)→ closed-form gap junction.

Consequence: the optimized backends are **not bit-identical** to the baseline —
they use a different (equally valid) integration scheme, so traces drift apart
over time in this chaotic system. Therefore:

- **optimized vs baseline** → *divergence* (measured, not asserted equal),
- **optimized vs optimized** (vec / vec_jit / jit) → *equality* to machine
  epsilon (they share identical Jacobi + O(N) formulas).


---

## Usage

> **Interpreter note.** The `py -3` launcher in the commands below is the
> **Windows dev-host** convention (it resolves to a Python 3.12 with the needed
> packages; the bare `python` on that host does *not*). **On the Linux server the
> interpreter will differ** — use whatever the server provides (`python3`, a
> `module load python/...`, or an activated venv/conda env) and substitute it for
> `py -3` everywhere. Requires `numpy` + `matplotlib`, and `numba` for the JIT
> backends (`io_model_jit.py`, `io_model_vec_jit.py`); install with
> `<your-python> -m pip install numba` if missing.

### Run a single backend directly

Each backend's `__main__` runs the default config (30 cells, 1.0 s, δ=0.01),
prints its execution time, and plots the soma traces. The JIT backends warm up
(compile) before the timed run.

```bash
py -3 io_model.py          # baseline reference
py -3 io_model_vec.py      # vectorized
py -3 io_model_jit.py      # scalar njit (fastest)
py -3 io_model_vec_jit.py  # njit on the vectorized loop
```

### Validate + compare backends

`validate.py` runs the baseline plus the backends you select, all from the same
seeded initial state.

```bash
# all backends, default 1.0 s setting (baseline is slow at 1.0 s)
py -3 validate.py

# pick which backend(s) to compare against the baseline
py -3 validate.py --backends jit
py -3 validate.py --backends vec jit

# quick check on a shorter window
py -3 validate.py --backends vec vec_jit jit --sim-seconds 0.2
```

Flags: `--backends {vec,vec_jit,jit,all}`, `--sim-seconds`, `--n-cells`,
`--delta`, `--seed`. Output has three parts: a timing/speedup line, per-backend
divergence vs baseline, and (when ≥2 backends are selected) an equality
cross-check.

Representative result (N=30, 0.2 s = 20,000 steps):

```
baseline 29.45s   vec 4.94s (5.97x)   vec_jit 0.27s (110.88x)   jit 0.14s (209.44x)
...
equality vs vec:  vec_jit 5.7e-14 mV   jit 5.7e-14 mV   (machine epsilon)
```

Performance order at small N: **jit > vec_jit > vec** — the scalar njit avoids
array temporaries that the (compiled or not) vectorized form still allocates.

### Multiprocessing across independent sims

`mp_sims.py` runs many independent sims (each its own seed → its own state) in
parallel with a `ProcessPoolExecutor`, on the `jit` backbone.

```bash
py -3 mp_sims.py                                  # 64 sims, default workers
py -3 mp_sims.py --n-sims 200 --sim-seconds 2.0   # heavier batch
py -3 mp_sims.py --n-sims 32 --n-cells 100 --no-gj
```

Flags: `--n-sims`, `--sim-seconds`, `--n-cells`, `--delta`, `--no-gj`. It prints
the single-sim time, the estimated serial cost, the parallel time + throughput,
and the speedup.

> **Note on platforms:** On Linux (the cluster) the pool uses `fork`, so workers
> inherit the parent's already-compiled JIT kernel — startup is negligible and
> speedup scales with cores. On Windows (`spawn`) each worker re-imports and
> reloads the JIT cache, so small batches can look slower than serial; use a
> larger `--n-sims` to amortize startup.

### Baseline sweep / profiling

```bash
py -3 sweep.py --list           # show sweepable parameters
py -3 sweep.py n_cells          # sweep one parameter
py -3 sweep.py all --repeats 3  # sweep everything, best of 3
py -3 profile_io.py             # cProfile breakdown of the baseline
```

---

## Suggested workflow

1. Profile the baseline (`profile_io.py`) to see where the time goes.
2. Validate the optimized backends against the baseline (`validate.py`) — confirm
   speedup, bounded divergence, and machine-epsilon agreement.
3. Pick `jit` as the single-sim backbone and scale out with `mp_sims.py`.
