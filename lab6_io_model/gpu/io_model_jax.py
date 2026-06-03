#
# GPU backend (Goals A + B) for the Inferior-Olive (de Gruijl) model: JAX + XLA.
#
# Unlike the CPU page where "vectorized" (Goal A) and "njit" (Goal B) are a fork,
# the GPU path is a single STACKED solution -- each goal is a prerequisite for the
# next. This file implements the first two layers; Goals C (vmap batch), D
# (recording strategy) and E (f32/f64 precision fork) build on top of it.
#
#   Goal A -- functional vectorized per-network step. Same math as the CPU
#     io_model_vec.simulate (Jacobi double-buffer) and io_model_jit (local kNN
#     gap junction), but PURE: JAX arrays are immutable, so the step reads the
#     start-of-step snapshot and RETURNS new arrays. The Jacobi double-buffer is
#     therefore structural, not a choice -- every d*/dt reads only the old carry.
#
#   Goal B -- roll the entire T-step time loop into `lax.scan` under `jax.jit`.
#     The time axis is irreducibly serial (forward-Euler: state[t+1] depends on
#     state[t]), so it cannot be parallelized; but a Python for-loop calling a
#     jitted step pays T launches + dispatch. `lax.scan` fuses the whole loop into
#     ONE XLA graph -> one launch for the entire sim. Compile once, run many.
#
# The single genuine fork (Goal E: f32 vs f64) is left to the caller: every
# function here is dtype-driven by its input arrays, so passing f64 state (with
# jax.config jax_enable_x64=True) gives the assignment-mandated exact path, while
# f32 state gives the fast path. Validation against the CPU `io_model_jit` (f64)
# therefore just feeds f64 arrays in.
#
# Numerics are intentionally identical to lab6_io_model/cpu/io_model_vec.py
# (vectorized Jacobi) + io_model_jit.py (local kNN gap), so validate_jax.py can
# assert jax-vs-jit equality to f32/f64 tolerance.
#

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# ---> Model constants (de Gruijl Inferior-Olive); identical to io_model_vec.py.
# Module-level Python floats: XLA freezes them into the compiled graph as
# constants at trace time, exactly like Numba freezes them into the njit code.
g_int   = 0.13     # Cell internal conductance
p1      = 0.25     # Cell surface ratio soma/dendrite
p2      = 0.15     # Cell surface ratio axon(hillock)/soma
g_h     = 0.12     # H current (HCN)
g_K_Ca  = 35.0     # Potassium (KCa v1.1 - BK)
g_ld    = 0.01532  # Leak dendrite
g_la    = 0.016    # Leak axon
g_ls    = 0.004    # Leak soma
S       = 1.0      # 1/C_m, cm^2/uF
g_Na_s  = 150.0    # Sodium (Na v1.6)
g_Kdr_s = 9.0      # Potassium (K v4.3)
g_K_s   = 5.0      # Potassium (K v3.4)
g_CaH   = 4.5      # High-threshold calcium (Ca V2.1)
g_Na_a  = 240.0    # Sodium
g_K_a   = 240.0    # Potassium
V_Na    = 55.0     # Sodium
V_K     = -75.0    # Potassium
V_Ca    = 120.0    # Low-threshold calcium channel
V_h     = -43.0    # H current
V_l     = 10.0     # Leak
C_gap   = 0.05     # Gap conductance


class IOState(NamedTuple):
    """The 14 evolving per-cell state arrays carried through the time loop.

    A NamedTuple is a JAX pytree, so `lax.scan` threads it as the carry and (in
    Goal C) `jax.vmap` maps over its leading batch axis leaf-by-leaf. g_CaL is
    NOT here: it is a per-cell constant (never updated), so it is passed as a
    separate argument and merely closed over by the step -- which keeps it out of
    the scan carry while still letting vmap batch it per network.

    Every leaf is a (n_cells,) array (or (S, n_cells) once vmapped). The dtype of
    these arrays is what selects the Goal-E precision fork (f32 vs f64).
    """
    V_soma: jax.Array
    V_axon: jax.Array
    V_dend: jax.Array
    soma_k: jax.Array
    soma_l: jax.Array
    soma_h: jax.Array
    soma_n: jax.Array
    soma_x: jax.Array
    axon_Sodium_h: jax.Array
    axon_Potassium_x: jax.Array
    dend_Ca2Plus: jax.Array
    dend_Calcium_r: jax.Array
    dend_Potassium_s: jax.Array
    dend_Hcurrent_q: jax.Array


def state_from_dict(st, dtype=jnp.float64):
    """Convert a sweep.build_initial_state(...) dict into (IOState, g_CaL) on the
    device, cast to `dtype`.

    `st` is the NumPy dict the CPU backends already consume, so validation can
    build ONE seeded state and feed it to both the CPU `io_model_jit` and this
    backend -- apples-to-apples, same initial condition. dtype picks the Goal-E
    precision (jnp.float32 fast / jnp.float64 exact; the latter needs
    jax.config.update('jax_enable_x64', True)).
    """
    def arr(name):
        return jnp.asarray(st[name], dtype=dtype)

    state = IOState(
        V_soma=arr("V_soma"), V_axon=arr("V_axon"), V_dend=arr("V_dend"),
        soma_k=arr("soma_k"), soma_l=arr("soma_l"), soma_h=arr("soma_h"),
        soma_n=arr("soma_n"), soma_x=arr("soma_x"),
        axon_Sodium_h=arr("axon_Sodium_h"), axon_Potassium_x=arr("axon_Potassium_x"),
        dend_Ca2Plus=arr("dend_Ca2Plus"), dend_Calcium_r=arr("dend_Calcium_r"),
        dend_Potassium_s=arr("dend_Potassium_s"), dend_Hcurrent_q=arr("dend_Hcurrent_q"),
    )
    return state, arr("g_CaL")


def _make_step(g_CaL, neighbours, delta, sim_seconds,
               enable_gapjunctions, I_app, I_pulse10ms, use_knn):
    """Build the pure functional step closure (Goal A) -- the per-network advance
    used as the body of every (inner) lax.scan in simulate (Goal B).

    Closed-over (per-call constant) values:
      * g_CaL    -- (n_cells,) per-cell Ca conductance; under vmap this becomes a
                    batched tracer, so the step batches automatically.
      * neighbours -- (n_cells, k) int adjacency for local kNN gap, or None for
                    all-to-all. Same topology shared across a vmap batch.
      * delta, sim_seconds, I_app, I_pulse10ms -- scalars (may be traced).
      * use_knn, enable_gapjunctions -- Python bools, static at trace time so the
                    gap branch is chosen once, not per step.

    Returns step(carry, t_idx) -> new_carry, both IOState. It emits NO per-step
    output: recording is done at the block boundaries by simulate's outer scan, so
    the recorded buffer is bounded at (n_rec, ...) instead of (n_simsteps, ...).
    The start-of-step snapshot a recorder needs is just the incoming `carry`.
    """
    k_deg = neighbours.shape[1] if (use_knn and neighbours is not None) else 0

    def step(carry, t_idx):
        st = carry
        # --- start-of-step snapshot: every d*/dt below reads ONLY these arrays.
        # In JAX this is automatic (immutable arrays) -> the scheme is Jacobi by
        # construction, never Gauss-Seidel.
        Vs = st.V_soma
        Va = st.V_axon
        Vd = st.V_dend

        # ================= SOMA (reads Vs, Va, Vd) =================
        soma_I_leak = g_ls * (Vs - V_l)
        I_ds = (g_int / p1) * (Vs - Vd)
        I_as = (g_int / (1 - p2)) * (Vs - Va)
        soma_I_interact = I_ds + I_as

        soma_Ical = g_CaL * st.soma_k * st.soma_k * st.soma_k * st.soma_l * (Vs - V_Ca)
        soma_m_inf = 1 / (1 + jnp.exp(-(Vs + 30) / 5.5))
        soma_Ina = g_Na_s * soma_m_inf ** 3 * st.soma_h * (Vs - V_Na)
        soma_Ikdr = g_Kdr_s * st.soma_n ** 4 * (Vs - V_K)
        soma_Ik = g_K_s * st.soma_x ** 4 * (Vs - V_K)

        soma_I_Channels = soma_Ik + soma_Ikdr + soma_Ina + soma_Ical
        soma_dv_dt = S * (-(soma_I_leak + soma_I_interact + soma_I_Channels))

        soma_k_inf = 1 / (1 + jnp.exp(-(Vs + 61) / 4.2))
        soma_l_inf = 1 / (1 + jnp.exp((Vs + 85) / 8.5))
        soma_tau_l = (20 * jnp.exp((Vs + 160) / 30) /
                      (1 + jnp.exp((Vs + 84) / 7.3))) + 35
        soma_h_inf = 1 / (1 + jnp.exp((Vs + 70) / 5.8))
        soma_tau_h = 3 * jnp.exp(-(Vs + 40) / 33)
        soma_n_inf = 1 / (1 + jnp.exp(-(Vs + 3) / 10))
        soma_tau_n = 5 + (47 * jnp.exp((Vs + 50) / 900))
        soma_alpha_x = 0.13 * (Vs + 25) / (1 - jnp.exp(-(Vs + 25) / 10))
        soma_beta_x = 1.69 * jnp.exp(-(Vs + 35) / 80)
        soma_tau_x_inv = soma_alpha_x + soma_beta_x
        soma_x_inf = soma_alpha_x / soma_tau_x_inv

        soma_k_new = delta * (soma_k_inf - st.soma_k) + st.soma_k
        soma_l_new = delta * (soma_l_inf - st.soma_l) / soma_tau_l + st.soma_l
        soma_h_new = st.soma_h + delta * (soma_h_inf - st.soma_h) / soma_tau_h
        soma_n_new = delta * (soma_n_inf - st.soma_n) / soma_tau_n + st.soma_n
        soma_x_new = delta * (soma_x_inf - st.soma_x) * soma_tau_x_inv + st.soma_x

        # ================= AXON (reads Va, Vs) =================
        axon_I_leak = g_la * (Va - V_l)
        # Jacobi: I_sa uses the start-of-step soma Vs, NOT a freshly-updated soma.
        I_sa = (g_int / p2) * (Va - Vs)
        axon_I_interact = I_sa

        axon_m_inf = 1 / (1 + jnp.exp(-(Va + 30) / 5.5))
        axon_h_inf = 1 / (1 + jnp.exp((Va + 60) / 5.8))
        axon_Ina = g_Na_a * axon_m_inf ** 3 * st.axon_Sodium_h * (Va - V_Na)
        axon_tau_h = 1.5 * jnp.exp(-(Va + 40) / 33)
        axon_Ik = g_K_a * st.axon_Potassium_x ** 4 * (Va - V_K)
        axon_alpha_x = 0.13 * (Va + 25) / (1 - jnp.exp(-(Va + 25) / 10))
        axon_beta_x = 1.69 * jnp.exp(-(Va + 35) / 80)
        axon_tau_x_inv = axon_alpha_x + axon_beta_x
        axon_x_inf = axon_alpha_x / axon_tau_x_inv

        axon_I_Channels = axon_Ina + axon_Ik
        axon_dv_dt = S * (-(axon_I_leak + axon_I_interact + axon_I_Channels))

        axon_Sodium_h_new = st.axon_Sodium_h + delta * (axon_h_inf - st.axon_Sodium_h) / axon_tau_h
        axon_Potassium_x_new = delta * (axon_x_inf - st.axon_Potassium_x) * axon_tau_x_inv + st.axon_Potassium_x

        # ================= DEND (reads Vd, Vs) =================
        # Pulse window is data-dependent on the scan index -> jnp.where, NOT a
        # Python `if` (the value t = t_idx*delta is a tracer under scan). This is
        # the Lab3 "replace if with lax.select/jnp.where" rule.
        t = t_idx * delta
        pulse = jnp.where((200 * sim_seconds < t) & (t < 210 * sim_seconds),
                          -I_pulse10ms, 0.0)
        dend_I_application = -I_app + pulse
        dend_I_leak = g_ld * (Vd - V_l)
        # Jacobi: interaction uses start-of-step soma Vs.
        dend_I_interact = (g_int / (1 - p1)) * (Vd - Vs)

        dend_Icah = g_CaH * st.dend_Calcium_r * st.dend_Calcium_r * (Vd - V_Ca)
        dend_Ikca = g_K_Ca * st.dend_Potassium_s * (Vd - V_K)
        dend_Ih = g_h * st.dend_Hcurrent_q * (Vd - V_h)

        # Gap junction. Local kNN (default, like io_model_jit): each cell couples
        # only to its k nearest neighbours -> stable at large n_cells. The
        # per-cell neighbour-sum is one gather+reduce: Vd[neighbours] is
        # (n_cells, k), summed over axis 1. All-to-all is the O(N) closed form
        # C_gap*(N*Vd - sum(Vd)). Both branches are chosen at trace time (static
        # use_knn), so only one is compiled.
        if enable_gapjunctions:
            if use_knn:
                nbr_sum = jnp.sum(Vd[neighbours], axis=1)        # (n_cells,)
                dend_I_gap = C_gap * (k_deg * Vd - nbr_sum)      # local, O(1)/cell
            else:
                dend_I_gap = C_gap * (Vd.shape[0] * Vd - jnp.sum(Vd))  # all-to-all
        else:
            dend_I_gap = 0.0

        dend_alpha_r = 1.7 / (1 + jnp.exp(-(Vd - 5) / 13.9))
        dend_beta_r = 0.02 * (Vd + 8.5) / (jnp.exp((Vd + 8.5) / 5) - 1.0)
        dend_tau_r_inv5 = dend_alpha_r + dend_beta_r
        dend_r_inf = dend_alpha_r / dend_tau_r_inv5
        dend_dr_dt = (dend_r_inf - st.dend_Calcium_r) * dend_tau_r_inv5 * 0.2

        # Piecewise alpha_s replicated EXACTLY as the multiply-by-boolean form
        # used in io_model_vec.py (not jnp.where), so the f64 trace matches the
        # CPU backend to machine epsilon including the >/< 0.01 edges.
        ca2 = 0.00002 * st.dend_Ca2Plus
        dend_alpha_s = ca2 * (ca2 < 0.01) + 0.01 * (ca2 > 0.01)
        dend_tau_s_inv = dend_alpha_s + 0.015
        dend_s_inf = dend_alpha_s / dend_tau_s_inv
        dend_ds_dt = (dend_s_inf - st.dend_Potassium_s) * dend_tau_s_inv

        q_inf = 1 / (1 + jnp.exp((Vd + 80) / 4))
        tau_q_inv = jnp.exp(-0.086 * Vd - 14.6) + jnp.exp(0.070 * Vd - 1.87)
        dq_dt = (q_inf - st.dend_Hcurrent_q) * tau_q_inv

        dCa_dt = -3 * dend_Icah - 0.075 * st.dend_Ca2Plus

        dend_Calcium_r_new = delta * dend_dr_dt + st.dend_Calcium_r
        dend_Potassium_s_new = delta * dend_ds_dt + st.dend_Potassium_s
        dend_Hcurrent_q_new = delta * dq_dt + st.dend_Hcurrent_q
        dend_Ca2Plus_new = delta * dCa_dt + st.dend_Ca2Plus

        dend_I_Channels = dend_Icah + dend_Ikca + dend_Ih
        dend_dv_dt = S * (-(dend_I_leak + dend_I_gap + dend_I_interact
                            + dend_I_application + dend_I_Channels))

        # --- double-buffer write: the three coupled V update together from the
        # snapshot; immutability means we just build a fresh IOState.
        new_carry = IOState(
            V_soma=Vs + soma_dv_dt * delta,
            V_axon=Va + axon_dv_dt * delta,
            V_dend=Vd + dend_dv_dt * delta,
            soma_k=soma_k_new, soma_l=soma_l_new, soma_h=soma_h_new,
            soma_n=soma_n_new, soma_x=soma_x_new,
            axon_Sodium_h=axon_Sodium_h_new, axon_Potassium_x=axon_Potassium_x_new,
            dend_Ca2Plus=dend_Ca2Plus_new, dend_Calcium_r=dend_Calcium_r_new,
            dend_Potassium_s=dend_Potassium_s_new, dend_Hcurrent_q=dend_Hcurrent_q_new,
        )

        return new_carry

    return step


def _advance(step, carry, start_idx, length):
    """Advance `carry` exactly `length` steps with an inner lax.scan that emits
    NOTHING (ys=None) -- this is what keeps device memory bounded: no per-step
    buffer is ever materialized. Global step indices start at `start_idx` (a
    tracer) so the pulse-window timing inside `step` stays correct; `length` is a
    Python int (static), which fixes the scan trip count."""
    xs = start_idx + jnp.arange(length)
    final, _ = lax.scan(lambda c, i: (step(c, i), None), carry, xs)
    return final


@partial(jax.jit, static_argnames=(
    "n_simsteps", "enable_gapjunctions", "use_knn", "record", "record_every"))
def simulate(state, g_CaL, neighbours,
             n_simsteps, delta, sim_seconds,
             enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
             use_knn=True, record=True, record_every=1):
    """Goal B: jit + lax.scan time loop -> one fused XLA graph, one launch.

    Args mirror the CPU backends' simulate(...): `state` is an IOState (the 14
    evolving (n_cells,) arrays), `g_CaL` the per-cell Ca conductance, `neighbours`
    the (n_cells, k) int kNN adjacency (or None / any array if use_knn=False).
    Structural args (n_simsteps, the bool flags, record_every) are static so XLA
    specializes the graph; delta/sim_seconds/I_* are traced so changing a current
    amplitude does NOT force a recompile.

    Recording is STRIDED ON-DEVICE in ONE uniform scan (the OOM fix). Rather than
    emitting one sample per step into a (n_simsteps, n_cells, 3) buffer and slicing
    after -- which materializes the full dense history and OOMs at large n_cells --
    an OUTER scan over the n_rec = ceil(n_simsteps / record_every) blocks emits
    exactly one strided sample per block (the block's start-of-step snapshot WITH
    its time column already folded in -> (n_cells, 4)), and a per-block inner loop
    advances that block's steps emitting nothing. So the ONLY trace buffer ever
    materialized is the scan's own (n_rec, n_cells, 4) stack.

    This is the residual-OOM fix at n_cells=100k under the sweep: the previous
    version stacked a (n_rec, n_cells, 3) buffer, then concatenated a separate tail
    block into a second copy, then _attach_time broadcast a (n_rec, n_cells, 1)
    time column and concatenated AGAIN into the final (n_rec, n_cells, 4) -- three
    full-size buffers live at once (~3x the trace, ~15 GiB at f64), which is what
    blew past device memory even though each individual buffer fit. Peak is now ~1x
    the trace. The last block can be shorter than record_every (when it does not
    divide n_simsteps), so its length is a tracer and the inner advance is a
    dynamic-trip lax.fori_loop -- this keeps the whole recorder in ONE outer scan
    (no concatenated tail) while still advancing EXACTLY n_simsteps steps, so
    `final` stays exact.

    Returns (v_trace, final, n_simsteps):
      * v_trace -- record=True: (n_rec, n_cells, 4), columns (V_soma, V_axon,
        V_dend, t). record=False: an empty (0,0,0) array (throughput path).
      * final   -- the IOState after the last step. Always real (it depends on the
        whole loop), so callers can `block_until_ready(final)` for honest timing
        even when record=False and v_trace is empty; it is also the throughput
        result (Goal D: carry the final state, return nothing per step).

    Compile is one-time and real (~seconds), keyed on the static args (n_simsteps,
    the flags, record_every) and the input SHAPES -- a different n_simsteps or
    n_cells recompiles. Warm up once per shape BEFORE any timed region (see
    __main__ / the sweep driver), then this amortizes over long sims / repeats.
    """
    step = _make_step(g_CaL, neighbours, delta, sim_seconds,
                      enable_gapjunctions, I_app, I_pulse10ms, use_knn)

    if not record:
        # Throughput path: advance the whole sim emitting nothing per step. Only
        # the final carry crosses PCIe back (near-zero DtoH -- Goal D).
        final = _advance(step, state, 0, n_simsteps)
        return jnp.empty((0, 0, 0), dtype=state.V_soma.dtype), final, n_simsteps

    # Strided recording. n_full full record_every-length blocks + one tail block
    # (tail in [1, record_every]); all three counts are static Python ints because
    # n_simsteps and record_every are static.
    n_rec = (n_simsteps + record_every - 1) // record_every
    n_full = n_rec - 1          # full-length blocks
    tail = n_simsteps - n_full * record_every       # steps left for the last block

    def outer_step(carry, block):
        # Start-of-step snapshot with the time column folded straight in: every
        # cell in this block shares t = block*record_every*delta, so build the
        # (n_cells, 4) row here instead of stacking 3 cols and concatenating a
        # broadcast time column afterwards (that post-hoc concat was a second
        # full-size copy of the whole trace -- part of the OOM).
        t = ((block * record_every) * delta).astype(carry.V_soma.dtype)
        t_col = jnp.broadcast_to(t, carry.V_soma.shape)             # (n_cells,)
        sample = jnp.stack([carry.V_soma, carry.V_axon, carry.V_dend, t_col],
                           axis=-1)                                 # (n_cells, 4)
        # Advance this block: record_every steps for every full block, `tail` for
        # the last one, so the loop covers EXACTLY n_simsteps steps and `final`
        # stays exact. The length is a tracer (it differs on the last block when
        # record_every does not divide n_simsteps), so the inner advance is a
        # dynamic-trip fori_loop rather than a static lax.scan -- which lets the
        # whole recorder stay in this ONE outer scan (no separate concatenated
        # tail block, hence one trace buffer instead of two).
        length = jnp.where(block == n_full, tail, record_every)
        start = block * record_every
        new_carry = lax.fori_loop(0, length, lambda k, c: step(c, start + k), carry)
        return new_carry, sample

    # One scan -> the sole (n_rec, n_cells, 4) device buffer; `final` is the carry
    # after the last block, i.e. exactly n_simsteps steps in.
    final, v_trace = lax.scan(outer_step, state, jnp.arange(n_rec))
    return v_trace, final, n_simsteps


def build_neighbours(n_cells, k=8, seed=1981, dims=3):
    """Precompute each cell's k nearest neighbours for LOCAL gap-junction coupling
    (the same scheme as io_model_jit.build_neighbours, kept here so the GPU
    backend has no import dependency on the CPU package).

    The model has no geometry, so cells are scattered at random positions in a
    unit `dims`-cube and the k closest others (Euclidean) are kept. A fixed k
    makes each cell's coupling degree independent of n_cells, which removes the
    explicit-Euler instability at large populations and matches the biology
    (olivary gap junctions connect a handful of touching dendrites). Returns an
    int32 (n_cells, k_eff) array, k_eff = min(k, n_cells-1); int32 keeps the gather
    index cheap on GPU.
    """
    import numpy as np
    k_eff = min(k, n_cells - 1)
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n_cells, dims))
    neighbours = np.empty((n_cells, k_eff), dtype=np.int32)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        _dist, idx = tree.query(pos, k=k_eff + 1)  # +1: nearest point is self
        idx = np.atleast_2d(idx)
        for i in range(n_cells):
            row = idx[i]
            neighbours[i] = row[row != i][:k_eff]
    except ImportError:
        for i in range(n_cells):
            d2 = np.sum((pos - pos[i]) ** 2, axis=1)
            d2[i] = np.inf  # exclude self
            nn = np.argpartition(d2, k_eff)[:k_eff]
            neighbours[i] = nn

    return jnp.asarray(neighbours)


if __name__ == "__main__":
    # Self-contained demo / smoke test. Runs on CPU too (jax defaults to CPU when
    # no GPU is visible), so this file can be exercised anywhere; the GPU win is
    # measured later via the batch sweep (Goal C+). Mirrors io_model_vec.__main__.
    import time

    import numpy as np
    import matplotlib.pyplot as plt

    # Goal E note: validation/exact path needs f64. Enable x64 BEFORE any jnp use.
    jax.config.update("jax_enable_x64", True)
    DTYPE = jnp.float64

    sim_seconds = 1.0
    delta = 0.01
    n_cells = 30
    enable_gapjunctions = True
    I_app = 0.0
    I_pulse10ms = 2.0
    USE_KNN = True
    K = 8

    np.random.seed(1981)  # reproducible, same distributions as the CPU backends
    g_CaL_np = np.random.normal(0.7, 0.1, n_cells)
    st_dict = {
        "g_CaL": g_CaL_np,
        "V_soma": np.random.uniform(-70, -40, size=(n_cells,)),
        "soma_k": np.full(n_cells, 0.7423159),
        "soma_l": np.full(n_cells, 0.0321349),
        "soma_h": np.full(n_cells, 0.3596066),
        "soma_n": np.full(n_cells, 0.2369847),
        "soma_x": np.full(n_cells, 0.1),
        "V_axon": np.random.uniform(-70, -40, size=(n_cells,)),
        "axon_Sodium_h": np.full(n_cells, 0.9),
        "axon_Potassium_x": np.full(n_cells, 0.2369847),
        "V_dend": np.random.uniform(-70, -40, size=(n_cells,)),
        "dend_Ca2Plus": np.full(n_cells, 3.715),
        "dend_Calcium_r": np.full(n_cells, 0.0113),
        "dend_Potassium_s": np.full(n_cells, 0.0049291),
        "dend_Hcurrent_q": np.full(n_cells, 0.0337836),
    }

    state, g_CaL = state_from_dict(st_dict, dtype=DTYPE)
    neighbours = build_neighbours(n_cells, k=K, seed=1981) if USE_KNN else jnp.zeros((n_cells, 1), jnp.int32)

    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)

    # Warm up (compile) at the REAL shape + flags so XLA compilation is OUT of the
    # timed region. Compile is keyed on shapes + static args, so the warm-up must
    # use the same n_simsteps/record as the timed call to be a cache hit below.
    tic = time.perf_counter()
    jax.block_until_ready(simulate(
        state, g_CaL, neighbours, n_simsteps, delta, sim_seconds,
        enable_gapjunctions, I_app, I_pulse10ms,
        use_knn=USE_KNN, record=True, record_every=1))
    print(f"Compile (warm-up) time: {time.perf_counter() - tic:.3f} sec.")

    tic = time.perf_counter()
    v_trace, _final, _ = simulate(
        state, g_CaL, neighbours, n_simsteps, delta, sim_seconds,
        enable_gapjunctions, I_app, I_pulse10ms,
        use_knn=USE_KNN, record=True, record_every=1)
    v_trace.block_until_ready()  # force completion before stopping the clock
    print(f"Simulation execution time: {time.perf_counter() - tic:.3f} sec.")

    v_trace = np.asarray(v_trace)
    for i in range(n_cells):
        v = v_trace[:, i, 0]
        v = (v - np.nanmean(v)) / (np.nanmax(v) - np.nanmin(v)) / 2
        plt.plot(v_trace[:, i, 3], i + v, color="gray")
    plt.show()
