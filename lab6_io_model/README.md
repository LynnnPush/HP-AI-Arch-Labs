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
| `validate.py` | Runs the baseline + chosen backends from one seeded state; reports speedup, divergence-vs-baseline, and machine-epsilon equality among the optimized backends. Both multiprocessing schemes (`mp`, `mp_intra`) are selectable too — a pure numeric check that they reproduce the `jit` numerics. |
| `sweep.py` | Parameter-sweep / performance harness for **any backend** (`--backend baseline\|vec\|jit\|vec_jit\|mp_intra`): wall/CPU time, throughput, latency, real-time factor. Catches unstable configs and records them rather than aborting. |
| `profile_io.py` | `cProfile` breakdown of the baseline (per-compartment cost, gap-junction on/off, scaling with N). |
| `mp_sims.py` | Multiprocessing across **independent** sims, using the `jit` backend as the per-sim backbone. |
| `mp_sims_intra.py` | Multiprocessing **within one** sim: splits the cells of a single sim across processes (domain decomposition). State lives in shared memory; the only per-step IPC is the global gap-junction sum (`Sd`) all-reduce + a barrier. |
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

# numeric check of the two multiprocessing schemes vs the baseline/jit
py -3 validate.py --backends jit mp mp_intra --sim-seconds 0.2
```

Flags: `--backends {vec,vec_jit,jit,mp,mp_intra,all}`, `--sim-seconds`,
`--n-cells`, `--delta`, `--seed`. Output has three parts: a timing/speedup line,
per-backend divergence vs baseline, and (when ≥2 backends are selected) an
equality cross-check.

Representative result (N=30, 0.2 s = 20,000 steps):

```
baseline 29.45s   vec 4.94s (5.97x)   vec_jit 0.27s (110.88x)   jit 0.14s (209.44x)
...
equality vs vec:  vec_jit 5.7e-14 mV   jit 5.7e-14 mV   (machine epsilon)
```

Performance order at small N: **jit > vec_jit > vec** — the scalar njit avoids
array temporaries that the (compiled or not) vectorized form still allocates.

**The `mp` and `mp_intra` backends here are a numeric check, not a speed test**
(validate runs one small sim; the multiprocessing overhead makes the *timing*
column meaningless — for real mp performance use `mp_sims.py` / `sweep.py`). `mp`
runs one sim through the across-sims pipeline (a `ProcessPoolExecutor` worker), so
it reproduces `jit` **bit-for-bit**. `mp_intra` splits the sim's cells across
processes and sums the global `Sd` as a sum of per-worker partials, so it agrees
with `jit` to **machine epsilon** (the equality cross-check) rather than exactly.

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

### Multiprocessing within a single sim

`mp_sims_intra.py` parallelises **one** sim by splitting its `n_cells` across
worker processes (SPMD / domain decomposition), in contrast to `mp_sims.py`
which fans *independent* sims across cores.

Inside one sim the cells are **not** independent: every timestep each dendrite
reads the O(N) gap-junction term, whose only cross-cell input is the single
global scalar `Sd = sum(V_dend)`. Everything else (soma/axon/dend dynamics + all
15 gating variables) is per-cell. So each step needs exactly one all-reduce (the
partial `V_dend` sums) plus a barrier — no per-cell data crosses processes.

**Minimising IPC** is the design goal: the 15 state arrays are allocated *once*
in a single shared-memory block by `build_shared_state()` (the
"buildInitialState" for this script). Workers attach to it by name at startup
and mutate it in place — state is never pickled per step. The only per-step
traffic is a `(n_workers,)` shared partial-sum array and a `Barrier`, i.e.
`O(n_workers)` scalars, independent of `n_cells`. With gap junctions **off** the
cells are fully independent, so the barriers are dropped entirely.

```bash
py -3 mp_sims_intra.py --n-cells 2000 --sim-seconds 0.2          # gj on, all-reduce/step
py -3 mp_sims_intra.py --n-cells 4000 --sim-seconds 0.1 --no-gj  # no barriers
py -3 mp_sims_intra.py --n-cells 200  --workers 4 --validate     # check spikes vs serial jit
```

Flags: `--n-cells`, `--sim-seconds`, `--delta`, `--workers`, `--no-gj`,
`--seed`, `--validate`. The per-cell **kernel** is identical to `io_model_jit`
(constants are imported from it); the only numerical difference is that the
global `Sd` is summed as a sum of per-worker partial sums, so traces agree with
`jit` to machine epsilon (the `validate.py` equality check) while the total spike
count still matches exactly (what `--validate` confirms).

It is also a **sweep backend** (`sweep.py --backend mp_intra`, see below), so it
reuses the same metrics/CSV machinery as the single-sim backends.

> **Cost of tight coupling.** Two barrier syncs per step (with gap junctions on)
> means this only pays off when per-step compute is large enough to amortise the
> synchronisation — i.e. **big `n_cells`**. At small `n_cells` the barriers
> dominate and it runs *slower* than the serial `jit` backend (just as the
> `spawn` note above applies to `mp_sims.py`). The `--no-gj` path, which needs no
> barriers, scales much more readily.

> **Stiff configs fail cleanly (no hang).** The explicit-Euler model goes stiff
> at large `n_cells` *with gap junctions on* (same instability `sweep.py`
> catches), so the largest stable populations here use `--no-gj`. When a worker
> does hit it, it flags the failure in shared memory and calls `barrier.abort()`
> so its peers unblock immediately instead of waiting at the barrier forever;
> every barrier wait also has a `barrier_timeout` (default 30 s) as a backstop.
> The parent then raises `ArithmeticError` — which `sweep.py` catches and records
> as `unstable`, so a sweep keeps going.

### Parameter sweep / profiling

`sweep.py` benchmarks a chosen backend across parameter ranges (defined in
`PARAM_SPECS`). Every sweep varies **one** parameter at a time while holding the
rest at baseline. The seeded initial state is built identically for every
backend, so a given `--seed` produces the same starting state whether you run
`baseline` or `jit`.

```bash
py -3 sweep.py --list                    # show sweepable parameters
py -3 sweep.py n_cells                    # sweep one parameter (baseline)
py -3 sweep.py all --repeats 3            # sweep everything, best of 3
py -3 sweep.py n_cells --backend jit      # sweep n_cells on the jit backend
py -3 sweep.py all --backend vec_jit      # full sweep on vec_jit
py -3 sweep.py n_cells --backend mp_intra --workers 4   # intra-sim parallel backend
py -3 profile_io.py                       # cProfile breakdown of the baseline
```

Flags: `--backend {baseline,vec,jit,vec_jit,mp_intra}` (default `baseline`),
`--repeats`, `--seed`, `--outdir`, plus the two added this lab below. JIT backends
are warmed up (compiled) once before timing so compilation is not charged to the
first run. Results are written to
`sweep_results/sweep_<backend>_<params>_<stamp>.{csv,json}`.

**`--workers W` (mp_intra only).** Process count for the intra-sim parallel
backend (one sim's cells split across `W` workers; default `os.cpu_count()`).
Ignored by the other backends. The backend builds its own shared-memory state
from the seed and self-warms its JIT kernels, so it slots into the same
apples-to-apples timing as the rest; its `cpu_time` aggregates the workers' CPU
via `os.times()` children counters. Note `sweep.py` holds `sim_seconds` at the
baseline (1.0 s = 100k steps), so at the small sweep `n_cells` values the barrier
overhead makes `mp_intra` slower than `jit` — the wiring is about reusing the
harness, not a win at these sizes.

**`--record-every N` (default 40).** On the optimized backends the voltage trace
is logged only every *N* steps (the state still advances every step), so the
trace buffer is bounded while recording cost still lands in the timing — the same
allocate-then-discard pattern as the baseline. This matters at large `n_cells`:
the dense `(n_simsteps, n_cells, 4)` trace is what runs out of memory (e.g. 100k
steps × 10k cells × 4 × 8 B ≈ 32 GB). `--record-every 0` disables recording
entirely for the very largest runs. The baseline always records densely (it
drives the reference `update_*` helpers, which log every step).

```bash
py -3 sweep.py n_cells --backend jit --record-every 40   # bounded trace
py -3 sweep.py n_cells --backend jit --record-every 0    # no recording
```

**Unstable configs are caught, not fatal.** The explicit-Euler model goes stiff
at large `n_cells` *with gap junctions on* — the O(N) gap current
`C_gap·(N·Vd − ΣVd)` grows with population, voltages blow up, and a gating
denominator hits zero (a `ZeroDivisionError` under Numba). The sweep records that
point with `status = "unstable: <Error>"` in the CSV/table and **continues** to
the next value instead of aborting. (With `enable_gapjunctions` off the same large
populations run fine.) For `--backend mp_intra` the same blow-up happens inside a
worker process; it is surfaced as an `ArithmeticError` to the parent (workers
flag it and `abort()` the barrier so nothing hangs), which the sweep catches the
same way. The default sweep `n_cells` range (≤ 30) never reaches it.

---

## Suggested workflow

1. Profile the baseline (`profile_io.py`) to see where the time goes.
2. Validate the optimized backends against the baseline (`validate.py`) — confirm
   speedup, bounded divergence, and machine-epsilon agreement.
3. Pick `jit` as the single-sim backbone and scale out with `mp_sims.py`.
