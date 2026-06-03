# Lab 6 — CPU + GPU Backend Optimization of the Inferior-Olive Model

This lab takes the reference Inferior-Olive (de Gruijl) neuron simulator and
speeds it up on the CPU, scales it out across processes, then ports it to the GPU
with JAX + XLA. The model integrates `n_cells` neurons, each with 3 compartments
(soma / axon / dendrite) and ~15 state variables, coupled through dendritic gap
junctions.

There are **three optimized single-sim CPU backends** plus a **multiprocessing**
driver and a **JAX/XLA GPU backend**, all validated against the untouched
reference.

The single-file backends are split by hardware target: the CPU ones live under
`cpu/`, the GPU one under `gpu/`. The harness scripts (`sweep.py`, `validate.py`,
`profile_io.py`) stay at the top level and import both.

---

## Files

| File | What it is |
|------|------------|
| `cpu/io_model.py` | **Reference** baseline (unchanged). Pure-Python scalar loops, Gauss-Seidel update order, O(N²) all-to-all gap junction. The correctness/standard reference. |
| `cpu/io_model_vec.py` | **vec** — vectorized NumPy over all cells. Jacobi double-buffer + O(N) closed-form gap junction. |
| `cpu/io_model_jit.py` | **jit** — scalar per-cell loops compiled with Numba `@njit`. Same Jacobi + O(N) numerics as `vec`. Fastest single-sim CPU backend. Also offers optional **local (k-nearest-neighbour) gap-junction coupling** (`use_knn` + `build_neighbours`) that stays stable at large `n_cells`. |
| `cpu/io_model_vec_jit.py` | **vec_jit** — `@njit` applied to the *vectorized* loop. Same numerics again; shows the cost of array temporaries under JIT. |
| `gpu/io_model_jax.py` | **jax_gpu** — JAX + XLA GPU backend. Pure functional step (Goal A, immutable Jacobi) fused over the whole time loop with `jax.jit` + `lax.scan` (Goal B): **one XLA graph, one launch per sim**. Supports the same local kNN / all-to-all gap and f32 (fast) / f64 (exact) precision. Runs on CPU too when no GPU is visible. Batching across networks (`vmap`, Goal C) is the next layer and not yet wired. |
| `validate.py` | Runs the baseline + chosen backends from one seeded state; reports speedup, divergence-vs-baseline, and machine-epsilon equality among the optimized backends. The multiprocessing schemes (`mp`, `mp_intra`) and the GPU backend (`jax`) are selectable too — a pure numeric check that each reproduces the `jit` numerics. |
| `sweep.py` | Parameter-sweep / performance harness for **any backend** (`--backend baseline\|vec\|jit\|vec_jit\|mp_intra\|jax_gpu`): wall/CPU time, throughput, latency, real-time factor. Catches unstable configs and records them rather than aborting. Supports local kNN gap-junction coupling on the jit/jax_gpu backends (`--knn`, **default on**). |
| `profile_io.py` | `cProfile` breakdown of the baseline (per-compartment cost, gap-junction on/off, scaling with N). |
| `cpu/mp_sims.py` | Multiprocessing across **independent** sims, using the `jit` backend as the per-sim backbone. |
| `cpu/mp_sims_intra.py` | Multiprocessing **within one** sim: splits the cells of a single sim across processes (domain decomposition). State lives in shared memory; the only per-step IPC is the global gap-junction sum (`Sd`) all-reduce + a barrier. |
| `sweep_results/` | CSV/JSON output from `sweep.py`. |

---

## Key concept: Jacobi vs Gauss-Seidel

The baseline updates cells **sequentially** within a timestep, so the axon/dend
of a cell see the *freshly updated* soma voltage, and the gap junction sees
already-updated neighbours (**Gauss-Seidel**).

Every optimized backend computes each derivative from a **start-of-step
snapshot** and writes the three coupled voltages together (**Jacobi**), which is
what enables vectorization and the O(N)→ closed-form gap junction. On the JAX
backend this is not even a choice: JAX arrays are immutable, so the step reads the
old state and returns new arrays — the Jacobi double-buffer is *structural*.

Consequence: the optimized backends are **not bit-identical** to the baseline —
they use a different (equally valid) integration scheme, so traces drift apart
over time in this chaotic system. Therefore:

- **optimized vs baseline** → *divergence* (measured, not asserted equal),
- **optimized vs optimized** (vec / vec_jit / jit / jax) → *equality* to machine
  epsilon (they share identical Jacobi + O(N) formulas; the f64 JAX backend
  matches to ~epsilon, an f32 JAX run drifts within an f32 tolerance instead).


---

## Usage

> **Interpreter note.** The `py -3` launcher in the commands below is the
> **Windows dev-host** convention (it resolves to a Python 3.12 with the needed
> packages; the bare `python` on that host does *not*). **On the Linux server the
> interpreter will differ** — use whatever the server provides (`python3`, a
> `module load python/...`, or an activated venv/conda env) and substitute it for
> `py -3` everywhere. Requires `numpy` + `matplotlib`, `numba` for the JIT
> backends (`cpu/io_model_jit.py`, `cpu/io_model_vec_jit.py`), and `jax` for the
> GPU backend (`gpu/io_model_jax.py`); install with
> `<your-python> -m pip install numba jax` (add the CUDA `jax[cuda12]` wheel on the
> GPU host) if missing.

> **Running the harness scripts.** `sweep.py` / `validate.py` import the backends
> by package path (`lab6_io_model.cpu.*` / `lab6_io_model.gpu.*`) **and** import
> each other by bare name (`from sweep import …`), so both the repo root
> (`labs_scripts/`) and this folder (`lab6_io_model/`) must be on the path. Run
> them from this folder with both on `PYTHONPATH`, e.g. (bash)
> `PYTHONPATH="..:." py -3 sweep.py …`. The single-file backends below are
> self-contained and need no such setup.

### Run a single backend directly

Each backend's `__main__` runs the default config (30 cells, 1.0 s, δ=0.01),
prints its execution time, and plots the soma traces. The JIT / JAX backends warm
up (compile) before the timed run.

```bash
py -3 cpu/io_model.py          # baseline reference
py -3 cpu/io_model_vec.py      # vectorized
py -3 cpu/io_model_jit.py      # scalar njit (fastest CPU)
py -3 cpu/io_model_vec_jit.py  # njit on the vectorized loop
py -3 gpu/io_model_jax.py      # JAX + XLA (jit + lax.scan); runs on CPU if no GPU
```

`cpu/io_model_jit.py`'s `__main__` runs with local k-nearest-neighbour
gap-junction coupling by default (the `USE_KNN` / `K` toggle at the top, matching
`sweep.py`); set `USE_KNN = False` to fall back to the original all-to-all
coupling. `gpu/io_model_jax.py` has the same toggle and defaults to **float64**
(`jax_enable_x64`) so its demo trace lines up with the CPU backends; it reports
the one-time XLA compile (warm-up) time separately from the timed run.

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

# GPU backend vs the CPU jit (both f64, all-to-all -> equality to ~machine epsilon)
py -3 validate.py --backends jit jax --sim-seconds 0.2
```

Flags: `--backends {vec,vec_jit,jit,mp,mp_intra,jax,all}`, `--sim-seconds`,
`--n-cells`, `--delta`, `--seed`. Output has three parts: a timing/speedup line,
per-backend divergence vs baseline, and (when ≥2 backends are selected) an
equality cross-check. The `jax` backend runs in **float64** and **all-to-all**
coupling (matching the other validate backends), and is warmed at the real shape
before timing; it compiles per shape, so its first run at a new `--n-cells` /
`--sim-seconds` pays a one-time XLA compile.

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
py -3 sweep.py n_cells --backend jit      # sweep n_cells on the jit backend (kNN coupling, default)
py -3 sweep.py n_cells --backend jit --no-knn   # same sweep, original all-to-all coupling
py -3 sweep.py all --backend vec_jit      # full sweep on vec_jit
py -3 sweep.py n_cells --backend mp_intra --workers 4   # intra-sim parallel backend
py -3 sweep.py n_cells --backend jax_gpu                # JAX/XLA GPU backend (f32 default)
py -3 sweep.py n_cells --backend jax_gpu --jax-x64      # GPU in f64 (exact)
py -3 profile_io.py                       # cProfile breakdown of the baseline
```

Flags: `--backend {baseline,vec,jit,vec_jit,mp_intra,jax_gpu}` (default
`baseline`), `--repeats`, `--seed`, `--outdir`, plus `--knn`/`--k`,
`--record-every`, `--workers`, and `--jax-x64` documented below. The JIT/JAX
backends are warmed up (compiled) once before timing so compilation is not charged
to the first run. Results are written to
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

The `jax_gpu` backend honours this **on-device too**: rather than emitting one
sample per step into a `(n_simsteps, n_cells, 3)` device buffer and slicing after
(which would OOM the GPU at large `n_cells`), `simulate` uses a **nested
`lax.scan`** — an outer scan over the `n_rec = ceil(n_simsteps / N)` blocks emits
one strided sample each, while an inner scan advances `N` steps emitting nothing.
The recorded buffer is `(n_rec, n_cells, 3)`, i.e. `N×` smaller, and the recorded
step positions match the CPU backends exactly (the last block is handled
separately so the loop still advances *exactly* `n_simsteps` steps). `--record-every
0` takes the throughput path (no per-step output; only the final state returns).

```bash
py -3 sweep.py n_cells --backend jit --record-every 40       # bounded trace
py -3 sweep.py n_cells --backend jit --record-every 0        # no recording
py -3 sweep.py n_cells --backend jax_gpu --record-every 40   # strided on-device (N× smaller buffer)
```

**`--knn` / `--no-knn` (default `--knn`) and `--k K` (default 8) — jit & jax_gpu.**
Local gap-junction coupling: each cell couples only to its `K` nearest neighbours
instead of every other cell. Because the coupling degree is fixed at `K`
regardless of population, the per-step dendritic update no longer grows with
`n_cells`, so large populations stay numerically stable (and the topology is also
more biologically faithful — real olivary gap junctions are local, not all-to-all).
The neighbour list is built once per config (random positions + KD-tree)
**outside** the timed region, then a per-timestep neighbour-sum keeps the gap term
O(1) per cell — the same precompute-the-sum trick as the all-to-all path,
restricted to a fixed neighbour set (a gather + segment-sum on the GPU). `--no-knn`
restores the original all-to-all coupling. Only the `jit` and `jax_gpu` backends
implement this; on any other backend the flag is ignored (a one-line note is
printed) and coupling stays all-to-all.

```bash
py -3 sweep.py n_cells --backend jit            # kNN (default): stable past 1000 cells
py -3 sweep.py n_cells --backend jit --k 16     # denser local coupling
py -3 sweep.py n_cells --backend jit --no-knn   # all-to-all: goes unstable above ~4000 cells
```

**`--jax-x64` (jax_gpu only).** Run the JAX backend in **float64** instead of the
default **float32**. f32 is the fast GPU throughput path (≈2× faster, half the
memory); f64 is the exact, assignment-mandated path and is what the `validate.py`
equality check uses. The flag flips JAX's process-global `jax_enable_x64` before
any device array is built. Choosing between them is the plan's one measured
"precision fork" (Goal E) — decide by the f32-vs-f64 trace drift, not by
assumption. The recorded precision is stored as `jax_dtype` in the result JSON.
Like the other compiled backends, `jax_gpu` self-warms: the first run at a given
`(n_simsteps, n_cells, flags)` shape compiles the XLA graph (one-time, excluded
from timing), and `--repeats` reuses the cached compile.

> **Note on the GPU win (Goals C–E).** This backend currently runs **one** network
> per call (`S=1`). The time loop is irreducibly serial, so a single small network
> exposes only ~`n_cells` lanes — nowhere near a GPU's width, and at `S=1` the GPU
> typically *loses* to the CPU `jit` backbone. The throughput win comes from
> batching many independent networks with `vmap` (Goal C: arrays become `(S, N)`),
> which is the next layer and not yet wired into the sweep. Report **throughput**
> (`S·T / wall`) for the GPU, not single-sim latency.

```bash
py -3 sweep.py n_cells --backend jax_gpu            # f32, kNN (default)
py -3 sweep.py n_cells --backend jax_gpu --jax-x64  # f64 (exact), kNN
py -3 sweep.py n_cells --backend jax_gpu --no-knn   # all-to-all coupling
```

**Unstable configs are caught, not fatal.** With all-to-all coupling (`--no-knn`,
or any non-jit backend) the explicit-Euler model goes stiff at large `n_cells`
*with gap junctions on* — the O(N) gap current `C_gap·(N·Vd − ΣVd)` grows with
population, voltages blow up, and a gating denominator hits zero (a
`ZeroDivisionError` under Numba) near ~4000 cells. The sweep records that point
with `status = "unstable: <Error>"` in the CSV/table and **continues** to the next
value instead of aborting. (With `enable_gapjunctions` off, or under the default
`--knn` coupling, the same large populations run fine.) For `--backend mp_intra`
the same blow-up happens inside a worker process; it is surfaced as an
`ArithmeticError` to the parent (workers flag it and `abort()` the barrier so
nothing hangs), which the sweep catches the same way. The `n_cells` sweep range
goes up to 5000 specifically to exercise this threshold: under the default kNN
coupling every value is stable, while `--no-knn` flags the 5000-cell point
(and on the jit backend nothing else) as unstable.

> **Caveat — `jax_gpu` does not raise on blow-up.** XLA produces `NaN`/`inf`
> *silently* (no Python exception), so the sweep's `try/except` cannot flag an
> unstable JAX config — it reports `status = ok` with a real wall time but
> `NaN`-poisoned traces. Stay on the default `--knn` coupling (stable at large
> `n_cells`), and treat f32 with extra care: the explicit-Euler gates are stiff, so
> `--no-knn` or very large populations in f32 can go non-finite without warning.

---

## Suggested workflow

1. Profile the baseline (`profile_io.py`) to see where the time goes.
2. Validate the optimized backends against the baseline (`validate.py`) — confirm
   speedup, bounded divergence, and machine-epsilon agreement (include `jax` to
   check the GPU port in f64).
3. Pick `jit` as the single-sim CPU backbone and scale out with `mp_sims.py`.
4. For the GPU, decide the precision fork (`--jax-x64` f64 vs default f32) from the
   measured trace drift, then batch many networks (`vmap`, Goal C) for the
   throughput win — single small sims stay on the CPU backbone.
