#
# Goal C (variant): multiprocessing WITHIN a single simulation.
#
# mp_sims.py fans *independent* sims across cores -- embarrassingly parallel,
# zero coupling. This script instead parallelises ONE sim by splitting its
# n_cells across worker processes (domain decomposition / SPMD).
#
# Why this is harder: inside one sim the cells are NOT independent. Every
# timestep the dendrite of each cell reads the O(N) gap-junction term
#       dend_I_gap[i] = C_gap * (n_cells * Vd[i] - Sd),   Sd = sum_j Vd[j]
# i.e. the ONLY cross-cell coupling is the single global scalar Sd = sum(Vd).
# Everything else (soma/axon/dend dynamics + all 15 gating variables) is purely
# per-cell. So each step needs exactly one all-reduce (the partial sums of Vd)
# plus a barrier; no per-cell data crosses process boundaries.
#
# Minimising IPC -- the whole point of the assignment:
#   * The 15 state arrays (V_soma/axon/dend + gating + g_CaL) live in ONE
#     shared-memory block, allocated inside build_shared_state() (the
#     "buildInitialState" for this script). Workers attach to it BY NAME once at
#     startup and mutate it in place. State is never pickled per step.
#   * The only per-step IPC is a (n_workers,) shared partial-sum array plus a
#     process Barrier -- O(n_workers) scalars, independent of n_cells.
#
# The per-cell math is the exact scalar-njit kernel from io_model_jit.simulate
# (same Jacobi + O(N) closed-form gap numerics), just restricted to a cell range
# [start, end) and reading a pre-reduced Sd. Constants are imported from
# io_model_jit so Numba freezes the identical values.
#

from multiprocessing import shared_memory
from threading import BrokenBarrierError  # mp.Barrier subclasses threading.Barrier
import argparse
import multiprocessing as mp
import os
import time

import numpy as np
from numba import njit

import lab6_io_model.cpu.io_model_jit as io_model_jit
from sweep import build_initial_state

# Model constants -- imported so the njit kernel freezes the SAME values as the
# single-process jit backend (keeps numerics bit-identical to io_model_jit).
from lab6_io_model.cpu.io_model_jit import (
    g_int, p1, p2, g_h, g_K_Ca, g_ld, g_la, g_ls, S,
    g_Na_s, g_Kdr_s, g_K_s, g_CaH, g_Na_a, g_K_a,
    V_Na, V_K, V_Ca, V_h, V_l, C_gap,
)

SPIKE_THRESHOLD = -20.0  # mV; soma upward crossings counted as spikes

# ---------------------------------------------------------------------------
# Layout of the single shared state block: shape (N_FIELDS, n_cells) float64,
# one row per state array. Row indices are module-level ints so the njit kernel
# can freeze them. The field order matches build_initial_state's dict.
# ---------------------------------------------------------------------------
FIELDS = (
    "V_soma", "V_axon", "V_dend",
    "soma_k", "soma_l", "soma_h", "soma_n", "soma_x",
    "axon_Sodium_h", "axon_Potassium_x",
    "dend_Ca2Plus", "dend_Calcium_r", "dend_Potassium_s", "dend_Hcurrent_q",
    "g_CaL",
)
N_FIELDS = len(FIELDS)
(I_V_SOMA, I_V_AXON, I_V_DEND,
 I_SOMA_K, I_SOMA_L, I_SOMA_H, I_SOMA_N, I_SOMA_X,
 I_AXON_H, I_AXON_X,
 I_DEND_CA, I_DEND_R, I_DEND_S, I_DEND_Q,
 I_G_CAL) = range(N_FIELDS)


# ---------------------------------------------------------------------------
# njit kernels operating on the shared (N_FIELDS, n_cells) block.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _partial_sum_vd(st, start, end):
    """Sum of V_dend over the cell range [start, end) -- this worker's share of
    the global Sd reduction."""
    s = 0.0
    for i in range(start, end):
        s += st[I_V_DEND, i]
    return s


@njit(cache=True)
def _step_chunk(st, start, end, Sd, n_cells, delta, pulse, enable_gj, I_app):
    """Advance cells [start, end) by ONE timestep, in place on the shared block.

    Identical per-cell numerics to io_model_jit.simulate's inner loop (Jacobi:
    each cell reads its own start-of-step V from `st`, which it overwrites only
    at the very end, so reads see start-of-step values). `Sd` is the global
    sum(V_dend) reduced across ALL workers before this call -- the only piece of
    cross-cell information needed.
    """
    for i in range(start, end):
        vs = st[I_V_SOMA, i]
        va = st[I_V_AXON, i]
        vd = st[I_V_DEND, i]

        # ---------------- SOMA (reads vs, va, vd) ----------------
        sk = st[I_SOMA_K, i]; sl = st[I_SOMA_L, i]; sh = st[I_SOMA_H, i]
        sn = st[I_SOMA_N, i]; sx = st[I_SOMA_X, i]
        soma_I_leak = g_ls * (vs - V_l)
        soma_I_interact = (g_int / p1) * (vs - vd) + (g_int / (1 - p2)) * (vs - va)
        soma_Ical = st[I_G_CAL, i] * sk * sk * sk * sl * (vs - V_Ca)
        soma_m_inf = 1 / (1 + np.exp(-(vs + 30) / 5.5))
        soma_Ina = g_Na_s * soma_m_inf ** 3 * sh * (vs - V_Na)
        soma_Ikdr = g_Kdr_s * sn ** 4 * (vs - V_K)
        soma_Ik = g_K_s * sx ** 4 * (vs - V_K)
        soma_dv_dt = S * (-(soma_I_leak + soma_I_interact
                            + soma_Ik + soma_Ikdr + soma_Ina + soma_Ical))

        soma_k_inf = 1 / (1 + np.exp(-(vs + 61) / 4.2))
        soma_l_inf = 1 / (1 + np.exp((vs + 85) / 8.5))
        soma_tau_l = (20 * np.exp((vs + 160) / 30) / (1 + np.exp((vs + 84) / 7.3))) + 35
        soma_h_inf = 1 / (1 + np.exp((vs + 70) / 5.8))
        soma_tau_h = 3 * np.exp(-(vs + 40) / 33)
        soma_n_inf = 1 / (1 + np.exp(-(vs + 3) / 10))
        soma_tau_n = 5 + (47 * np.exp((vs + 50) / 900))
        soma_alpha_x = 0.13 * (vs + 25) / (1 - np.exp(-(vs + 25) / 10))
        soma_beta_x = 1.69 * np.exp(-(vs + 35) / 80)
        soma_tau_x_inv = soma_alpha_x + soma_beta_x
        soma_x_inf = soma_alpha_x / soma_tau_x_inv
        st[I_SOMA_K, i] = delta * (soma_k_inf - sk) + sk
        st[I_SOMA_L, i] = delta * (soma_l_inf - sl) / soma_tau_l + sl
        st[I_SOMA_H, i] = sh + delta * (soma_h_inf - sh) / soma_tau_h
        st[I_SOMA_N, i] = delta * (soma_n_inf - sn) / soma_tau_n + sn
        st[I_SOMA_X, i] = delta * (soma_x_inf - sx) * soma_tau_x_inv + sx

        # ---------------- AXON (reads va, vs) ----------------
        ah = st[I_AXON_H, i]; ax = st[I_AXON_X, i]
        axon_I_leak = g_la * (va - V_l)
        I_sa = (g_int / p2) * (va - vs)
        axon_m_inf = 1 / (1 + np.exp(-(va + 30) / 5.5))
        axon_h_inf = 1 / (1 + np.exp((va + 60) / 5.8))
        axon_Ina = g_Na_a * axon_m_inf ** 3 * ah * (va - V_Na)
        axon_tau_h = 1.5 * np.exp(-(va + 40) / 33)
        axon_Ik = g_K_a * ax ** 4 * (va - V_K)
        axon_alpha_x = 0.13 * (va + 25) / (1 - np.exp(-(va + 25) / 10))
        axon_beta_x = 1.69 * np.exp(-(va + 35) / 80)
        axon_tau_x_inv = axon_alpha_x + axon_beta_x
        axon_x_inf = axon_alpha_x / axon_tau_x_inv
        axon_dv_dt = S * (-(axon_I_leak + I_sa + axon_Ina + axon_Ik))
        st[I_AXON_H, i] = ah + delta * (axon_h_inf - ah) / axon_tau_h
        st[I_AXON_X, i] = delta * (axon_x_inf - ax) * axon_tau_x_inv + ax

        # ---------------- DEND (reads vd, vs, + global Sd) ----------------
        dr = st[I_DEND_R, i]; ds = st[I_DEND_S, i]
        dq = st[I_DEND_Q, i]; ca = st[I_DEND_CA, i]
        dend_I_application = -I_app + pulse
        dend_I_leak = g_ld * (vd - V_l)
        dend_I_interact = (g_int / (1 - p1)) * (vd - vs)
        dend_Icah = g_CaH * dr * dr * (vd - V_Ca)
        dend_Ikca = g_K_Ca * ds * (vd - V_K)
        dend_Ih = g_h * dq * (vd - V_h)
        if enable_gj:
            dend_I_gap = C_gap * (n_cells * vd - Sd)  # O(N) closed form, Sd reduced
        else:
            dend_I_gap = 0.0

        dend_alpha_r = 1.7 / (1 + np.exp(-(vd - 5) / 13.9))
        dend_beta_r = 0.02 * (vd + 8.5) / (np.exp((vd + 8.5) / 5) - 1.0)
        dend_tau_r_inv5 = dend_alpha_r + dend_beta_r
        dend_r_inf = dend_alpha_r / dend_tau_r_inv5
        dend_dr_dt = (dend_r_inf - dr) * dend_tau_r_inv5 * 0.2
        dend_alpha_s = ((0.00002 * ca) * (1.0 if 0.00002 * ca < 0.01 else 0.0)
                        + 0.01 * (1.0 if 0.00002 * ca > 0.01 else 0.0))
        dend_tau_s_inv = dend_alpha_s + 0.015
        dend_s_inf = dend_alpha_s / dend_tau_s_inv
        dend_ds_dt = (dend_s_inf - ds) * dend_tau_s_inv
        q_inf = 1 / (1 + np.exp((vd + 80) / 4))
        tau_q_inv = np.exp(-0.086 * vd - 14.6) + np.exp(0.070 * vd - 1.87)
        dq_dt = (q_inf - dq) * tau_q_inv
        dCa_dt = -3 * dend_Icah - 0.075 * ca
        st[I_DEND_R, i] = delta * dend_dr_dt + dr
        st[I_DEND_S, i] = delta * dend_ds_dt + ds
        st[I_DEND_Q, i] = delta * dq_dt + dq
        st[I_DEND_CA, i] = delta * dCa_dt + ca
        dend_dv_dt = S * (-(dend_I_leak + dend_I_gap + dend_I_interact
                            + dend_I_application + dend_Icah + dend_Ikca + dend_Ih))

        # --- double-buffer write: three coupled V from the snapshot locals ---
        st[I_V_SOMA, i] = vs + soma_dv_dt * delta
        st[I_V_AXON, i] = va + axon_dv_dt * delta
        st[I_V_DEND, i] = vd + dend_dv_dt * delta


# ---------------------------------------------------------------------------
# Shared "buildInitialState": allocate the per-sim state in shared memory ONCE,
# so the cross-process variables never have to be re-sent each timestep.
# ---------------------------------------------------------------------------
def build_shared_state(n_cells, g_CaL, seed):
    """Build the seeded initial state (identical numbers to
    sweep.build_initial_state) inside a single shared-memory block.

    Returns (shm, st_view): `shm` is the SharedMemory handle the parent must keep
    alive and later close()/unlink(); `st_view` is a (N_FIELDS, n_cells) float64
    ndarray backed by it. Workers re-attach to the same block by `shm.name`.
    """
    src = build_initial_state(n_cells, g_CaL, seed)
    shm = shared_memory.SharedMemory(create=True, size=N_FIELDS * n_cells * 8)
    st = np.ndarray((N_FIELDS, n_cells), dtype=np.float64, buffer=shm.buf)
    for idx, name in enumerate(FIELDS):
        st[idx, :] = src[name]
    return shm, st


def _attach(name, shape):
    """Attach to an existing shared-memory block as a float64 ndarray of `shape`."""
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    return shm, arr


def _worker(wid, n_workers, start, end, st_name, red_name, trace_name,
            n_cells, n_simsteps, delta, sim_seconds, enable_gj, I_app,
            I_pulse10ms, record_every, barrier_timeout, barrier):
    """SPMD worker: own the cell range [start, end) for the whole sim.

    Per step (gap junctions ON): write our partial sum of Vd -> barrier ->
    everyone reads the reduced Sd -> advance our cells -> barrier. With gap
    junctions OFF the cells are fully independent, so we drop the barriers and
    just run our slice to completion (intra-sim embarrassingly parallel).

    If `trace_name` is non-empty we log this worker's cells' start-of-step V into
    the shared (n_rec, n_cells, 4) trace block every `record_every` steps (state
    still advances every step) -- same record/record_every convention as the
    other backends: validate.py uses stride 1 for step-aligned divergence, sweep
    uses 40 to bound the buffer. Workers own disjoint cell columns, so the writes
    never conflict.

    Failure handling: the model can go stiff (large n_cells + gap junctions) and
    raise inside _step_chunk. To avoid leaving the other workers blocked at the
    barrier forever, every wait() has a `barrier_timeout`, and a worker that
    raises records a status code in red[2, wid] and calls barrier.abort() so its
    peers unblock immediately (BrokenBarrierError) instead of waiting it out. The
    parent reads red[2] after join and raises.
    """
    st_shm, st = _attach(st_name, (N_FIELDS, n_cells))
    # red rows: 0 = partial sums, 1 = spike counts, 2 = status (0 ok / 1 numeric
    # instability / 2 saw a broken barrier / 3 other error).
    red_shm, red = _attach(red_name, (3, n_workers))
    record = bool(trace_name)
    if record:
        n_rec = (n_simsteps + record_every - 1) // record_every
        trace_shm = shared_memory.SharedMemory(name=trace_name)
        trace = np.ndarray((n_rec, n_cells, 4), dtype=np.float64, buffer=trace_shm.buf)
    try:
        prev = None
        spikes = 0
        t = 0.0
        for i_epoch in range(n_simsteps):
            # Spike count on start-of-step soma snapshots (matches the recorded
            # baseline trace, which logs start-of-step vs each step).
            cur = st[I_V_SOMA, start:end].copy()
            if prev is not None:
                spikes += int(((prev < SPIKE_THRESHOLD)
                               & (cur >= SPIKE_THRESHOLD)).sum())
            prev = cur

            # Log start-of-step V (snapshot, before _step_chunk overwrites it) --
            # same convention as io_model_jit, which records vs/va/vd then updates.
            if record and (i_epoch % record_every == 0):
                rec = i_epoch // record_every
                trace[rec, start:end, 0] = st[I_V_SOMA, start:end]
                trace[rec, start:end, 1] = st[I_V_AXON, start:end]
                trace[rec, start:end, 2] = st[I_V_DEND, start:end]
                trace[rec, start:end, 3] = t

            pulse = (-I_pulse10ms
                     if (200 * sim_seconds < t and t < 210 * sim_seconds) else 0.0)

            if enable_gj:
                red[0, wid] = _partial_sum_vd(st, start, end)
                barrier.wait(barrier_timeout)  # all partial sums written
                Sd = float(red[0].sum())       # every worker reduces locally
            else:
                Sd = 0.0

            _step_chunk(st, start, end, Sd, n_cells, delta, pulse, enable_gj, I_app)

            if enable_gj:
                barrier.wait(barrier_timeout)  # all Vd updated before next sum
            t += delta

        red[1, wid] = spikes
    except BrokenBarrierError:
        # A peer failed first (it aborted the barrier) or a wait timed out; bail
        # out cleanly -- the parent will surface the root failure via red[2].
        red[2, wid] = 2
    except (ArithmeticError, FloatingPointError):
        # Stiff config: a gating denominator hit zero. Flag it and break the
        # barrier so peers stop waiting on us right away.
        red[2, wid] = 1
        if enable_gj:
            barrier.abort()
    except Exception:
        red[2, wid] = 3
        if enable_gj:
            barrier.abort()
    finally:
        del st, red
        st_shm.close()
        red_shm.close()
        if record:
            del trace
            trace_shm.close()


def simulate_intra(n_cells, sim_seconds, delta, enable_gj=True, I_pulse10ms=2.0,
                   seed=1981, n_workers=None, g_CaL=None, record=False,
                   record_every=1, barrier_timeout=30.0):
    """Run ONE sim with its n_cells split across worker processes.

    `record`/`record_every` mirror the other backends (io_model_jit.simulate):
    record=False skips tracing (the perf path); record=True logs start-of-step V
    every `record_every` steps into a dense (n_rec, n_cells, 4) buffer (validate
    uses stride 1, sweep 40).

    `barrier_timeout` (seconds) bounds every per-step barrier wait so a worker
    that dies on a stiff config cannot hang its peers; on any worker failure the
    parent raises (ArithmeticError for numeric instability, else RuntimeError)
    instead of returning garbage.

    Returns (elapsed, throughput_cellsteps_per_s, total_spikes, n_workers,
    v_trace); v_trace is the (n_rec, n_cells, 4) trace when record=True, else None.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n_cells))  # no empty workers
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)

    # Compile the njit kernels once in the parent (excluded from timing); fork
    # children inherit the compiled code, cache=True covers spawn.
    _warm = np.zeros((N_FIELDS, 2))
    _warm[I_V_SOMA:I_V_DEND + 1, :] = -60.0
    _partial_sum_vd(_warm, 0, 2)
    _step_chunk(_warm, 0, 2, 0.0, 2, delta, 0.0, True, 0.0)

    # Allocate the shared state + reduction scratch ONCE (the IPC-minimising step).
    # red rows: 0 = partial sums, 1 = spike counts, 2 = per-worker status flag.
    st_shm, st = build_shared_state(n_cells, g_CaL, seed)
    red_shm = shared_memory.SharedMemory(create=True, size=3 * n_workers * 8)
    red = np.ndarray((3, n_workers), dtype=np.float64, buffer=red_shm.buf)
    red[:] = 0.0

    # Shared trace block (only when recording); workers write disjoint cell cols.
    if record:
        n_rec = (n_simsteps + record_every - 1) // record_every
        trace_shm = shared_memory.SharedMemory(create=True, size=n_rec * n_cells * 4 * 8)
        trace = np.ndarray((n_rec, n_cells, 4), dtype=np.float64, buffer=trace_shm.buf)
        trace_name = trace_shm.name
    else:
        trace_name = ""

    # Contiguous cell partition: worker w owns [bounds[w], bounds[w+1]).
    bounds = [round(w * n_cells / n_workers) for w in range(n_workers + 1)]

    ctx = mp.get_context()
    barrier = ctx.Barrier(n_workers)
    procs = [
        ctx.Process(
            target=_worker,
            args=(w, n_workers, bounds[w], bounds[w + 1], st_shm.name,
                  red_shm.name, trace_name, n_cells, n_simsteps, delta,
                  sim_seconds, enable_gj, 0.0, I_pulse10ms, record_every,
                  barrier_timeout, barrier),
        )
        for w in range(n_workers)
    ]

    start = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = time.time() - start

    # Snapshot results + failure status BEFORE tearing down shared memory. A
    # worker that died without flagging (e.g. SIGKILL) still shows a nonzero
    # exitcode, which we also treat as a failure.
    status = red[2].copy()
    exitcodes = [p.exitcode for p in procs]
    total_spikes = int(red[1].sum())
    throughput = n_simsteps * n_cells / elapsed  # cell-steps/s
    v_trace = trace.copy() if record else None  # copy out before unlinking shm

    # Drop ndarray views before closing the buffers (else BufferError).
    del st, red
    st_shm.close(); st_shm.unlink()
    red_shm.close(); red_shm.unlink()
    if record:
        del trace
        trace_shm.close(); trace_shm.unlink()

    # Surface worker failures as an exception (after cleanup, so nothing leaks).
    numeric_fail = bool((status == 1).any())
    other_fail = (bool((status == 3).any())
                  or any(ec not in (0, None) for ec in exitcodes))
    if numeric_fail:
        # Same class sweep.py catches -> recorded as "unstable", sweep continues.
        raise ArithmeticError(
            "intra-sim worker went unstable (gating denominator hit zero -- the "
            "explicit-Euler model is stiff at large n_cells with gap junctions on; "
            "try --no-gj or fewer cells)")
    if other_fail:
        raise RuntimeError(
            f"intra-sim worker(s) failed (status={status.tolist()}, "
            f"exitcodes={exitcodes})")

    return elapsed, throughput, total_spikes, n_workers, v_trace


def _serial_reference(n_cells, sim_seconds, delta, enable_gj, I_pulse10ms,
                      seed, record):
    """One single-process jit run for the speedup baseline / spike validation.

    record=True returns (elapsed, spikes) from the dense trace; record=False
    returns (elapsed, None) and avoids allocating the big trace.
    """
    st = build_initial_state(n_cells, None, seed)
    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)
    t0 = time.time()
    v_trace, _ = io_model_jit.simulate(
        st["V_soma"], st["V_axon"], st["V_dend"],
        st["soma_k"], st["soma_l"], st["soma_h"], st["soma_n"], st["soma_x"],
        st["axon_Sodium_h"], st["axon_Potassium_x"],
        st["dend_Ca2Plus"], st["dend_Calcium_r"], st["dend_Potassium_s"], st["dend_Hcurrent_q"],
        st["g_CaL"], n_cells, n_simsteps, delta, sim_seconds,
        enable_gj, 0.0, I_pulse10ms, record, 1,
    )
    elapsed = time.time() - t0
    if not record:
        return elapsed, None
    soma = v_trace[:, :, 0]
    spikes = int(((soma[:-1] < SPIKE_THRESHOLD)
                  & (soma[1:] >= SPIKE_THRESHOLD)).sum())
    return elapsed, spikes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallelise ONE IO-model sim by splitting its cells across "
                    "processes (shared-memory state, per-step Sd all-reduce).")
    parser.add_argument("--n-cells", type=int, default=2000,
                        help="Cells in the single sim (split across workers).")
    parser.add_argument("--sim-seconds", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes (default: os.cpu_count()).")
    parser.add_argument("--no-gj", action="store_true", help="Disable gap junctions.")
    parser.add_argument("--seed", type=int, default=1981)
    parser.add_argument("--validate", action="store_true",
                        help="Cross-check total spikes against the single-process "
                             "jit backend (adds a recorded serial run).")
    args = parser.parse_args()

    enable_gj = not args.no_gj
    n_simsteps = int(args.sim_seconds * 1000 / args.delta + 0.5)
    print(f"=== 1 sim | {args.n_cells} cells x {n_simsteps} steps, "
          f"split across processes (gj={'on' if enable_gj else 'off'}) ===")

    # Single-process reference (warmed): same problem on one core.
    _serial_reference(2, 0.00002, args.delta, enable_gj, 2.0, args.seed, False)  # warm jit
    serial, serial_spikes = _serial_reference(
        args.n_cells, args.sim_seconds, args.delta, enable_gj, 2.0,
        args.seed, args.validate)

    # Intra-sim parallel run.
    elapsed, throughput, spikes, n_workers, _ = simulate_intra(
        args.n_cells, args.sim_seconds, args.delta, enable_gj,
        seed=args.seed, n_workers=args.workers)

    print(f"workers         : {n_workers}")
    print(f"serial (1 core) : {serial:.3f}s")
    print(f"parallel        : {elapsed:.3f}s   ({throughput:,.0f} cell-steps/s)")
    print(f"speedup         : {serial / elapsed:.2f}x")
    print(f"total spikes    : {spikes}")
    if args.validate:
        ok = "OK" if spikes == serial_spikes else "MISMATCH"
        print(f"validate spikes : parallel={spikes}  serial={serial_spikes}  [{ok}]")
