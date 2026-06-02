#
# Goal A: vectorized NumPy backend for the Inferior-Olive (de Gruijl) model.
#
# Three changes vs the io_model.py reference, all documented in the CPU Backend
# Optimization plan:
#   1. Vectorized over cells: every per-cell/per-compartment scalar update becomes
#      one elementwise NumPy expression on (n_cells,) arrays -> removes the Python
#      call + per-cell-loop overhead.
#   2. Jacobi double-buffer: all d*/dt are computed from the *start-of-step*
#      membrane potentials (Vs, Va, Vd), then the three coupled V arrays are
#      written together. This differs from the reference's Gauss-Seidel ordering
#      (where axon/dend see the freshly-updated soma V), so traces will diverge
#      from baseline over time -- that divergence is measured, not asserted, in
#      validate.py.
#   3. O(N) closed-form gap junction: the all-to-all O(N^2) inner loop collapses
#      to I_gap[i] = C_gap * (N * Vd[i] - sum(Vd)). Exact algebra, no numerical
#      change vs an all-to-all Jacobi loop.
#
# Mirrors the lab1 tvb_vec_jit.py convention: a self-contained simulate(...)
# taking explicit arrays/params (no module globals in the hot loop), so the same
# signature can later be wrapped/ported to Numba @njit (Goal B).
#

import time

import numpy as np

# ---> Model constants (de Gruijl Inferior-Olive); identical to io_model.py.
# These are fixed physical parameters, not swept, so they stay module-level.
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


def simulate(
    V_soma, V_axon, V_dend,
    soma_k, soma_l, soma_h, soma_n, soma_x,
    axon_Sodium_h, axon_Potassium_x,
    dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q,
    g_CaL,
    n_cells, n_simsteps, delta, sim_seconds,
    enable_gapjunctions=True, I_app=0.0, I_pulse10ms=2.0,
    record=True, record_every=1,
):
    """Vectorized Jacobi IO simulation over all cells at once.

    All state args are (n_cells,) float64 arrays (see sweep.build_initial_state).
    Returns (v_trace, n_simsteps) where v_trace is a (n_rec, n_cells, 4) array
    (columns: V_soma, V_axon, V_dend, t) when record=True, else None.

    record_every decouples trace sampling from the (dense) integration step: the
    state still advances every step, but a row is logged only every Nth step, so
    n_rec = ceil(n_simsteps / record_every). This bounds the trace buffer at
    large n_cells (the dense buffer is what OOMs) while keeping the numerics
    identical. record_every=1 reproduces the original step-by-step trace.
    """
    # Work on copies so the caller's initial-state arrays are not mutated
    # (validate.py reuses the same seeded state across backends).
    V_soma = V_soma.astype(np.float64).copy()
    V_axon = V_axon.astype(np.float64).copy()
    V_dend = V_dend.astype(np.float64).copy()
    soma_k = soma_k.astype(np.float64).copy()
    soma_l = soma_l.astype(np.float64).copy()
    soma_h = soma_h.astype(np.float64).copy()
    soma_n = soma_n.astype(np.float64).copy()
    soma_x = soma_x.astype(np.float64).copy()
    axon_Sodium_h = axon_Sodium_h.astype(np.float64).copy()
    axon_Potassium_x = axon_Potassium_x.astype(np.float64).copy()
    dend_Ca2Plus = dend_Ca2Plus.astype(np.float64).copy()
    dend_Calcium_r = dend_Calcium_r.astype(np.float64).copy()
    dend_Potassium_s = dend_Potassium_s.astype(np.float64).copy()
    dend_Hcurrent_q = dend_Hcurrent_q.astype(np.float64).copy()
    g_CaL = g_CaL.astype(np.float64)

    # (n_rec, n_cells, 4) trace: one slice-assignment every record_every steps.
    n_rec = (n_simsteps + record_every - 1) // record_every if record else 0
    v_trace = np.empty((n_rec, n_cells, 4)) if record else None
    t = 0.0

    for i_epoch in range(n_simsteps):
        # --- start-of-step snapshot: every d*/dt this step reads ONLY these ---
        # (this is what makes the scheme Jacobi rather than Gauss-Seidel).
        Vs, Va, Vd = V_soma, V_axon, V_dend

        # Record the start-of-step potentials (matches the reference, which logs
        # each compartment's V before writing it) -- but only every record_every
        # steps, so the trace buffer stays bounded.
        if record and i_epoch % record_every == 0:
            rec = i_epoch // record_every
            v_trace[rec, :, 0] = Vs
            v_trace[rec, :, 1] = Va
            v_trace[rec, :, 2] = Vd
            v_trace[rec, :, 3] = t

        # ================= SOMA (reads Vs, Va, Vd) =================
        soma_I_leak = g_ls * (Vs - V_l)
        I_ds = (g_int / p1) * (Vs - Vd)
        I_as = (g_int / (1 - p2)) * (Vs - Va)
        soma_I_interact = I_ds + I_as

        # Channel currents use the *old* gating values (computed before updates).
        soma_Ical = g_CaL * soma_k * soma_k * soma_k * soma_l * (Vs - V_Ca)
        soma_m_inf = 1 / (1 + np.exp(-(Vs + 30) / 5.5))
        soma_Ina = g_Na_s * soma_m_inf ** 3 * soma_h * (Vs - V_Na)
        soma_Ikdr = g_Kdr_s * soma_n ** 4 * (Vs - V_K)
        soma_Ik = g_K_s * soma_x ** 4 * (Vs - V_K)

        soma_I_Channels = soma_Ik + soma_Ikdr + soma_Ina + soma_Ical
        soma_dv_dt = S * (-(soma_I_leak + soma_I_interact + soma_I_Channels))

        # Gating updates (local to the soma; use Vs and old gate values).
        soma_k_inf = 1 / (1 + np.exp(-(Vs + 61) / 4.2))
        soma_l_inf = 1 / (1 + np.exp((Vs + 85) / 8.5))
        soma_tau_l = (20 * np.exp((Vs + 160) / 30) /
                      (1 + np.exp((Vs + 84) / 7.3))) + 35
        soma_h_inf = 1 / (1 + np.exp((Vs + 70) / 5.8))
        soma_tau_h = 3 * np.exp(-(Vs + 40) / 33)
        soma_n_inf = 1 / (1 + np.exp(-(Vs + 3) / 10))
        soma_tau_n = 5 + (47 * np.exp((Vs + 50) / 900))
        soma_alpha_x = 0.13 * (Vs + 25) / (1 - np.exp(-(Vs + 25) / 10))
        soma_beta_x = 1.69 * np.exp(-(Vs + 35) / 80)
        soma_tau_x_inv = soma_alpha_x + soma_beta_x
        soma_x_inf = soma_alpha_x / soma_tau_x_inv

        soma_k = delta * (soma_k_inf - soma_k) + soma_k
        soma_l = delta * (soma_l_inf - soma_l) / soma_tau_l + soma_l
        soma_h = soma_h + delta * (soma_h_inf - soma_h) / soma_tau_h
        soma_n = delta * (soma_n_inf - soma_n) / soma_tau_n + soma_n
        soma_x = delta * (soma_x_inf - soma_x) * soma_tau_x_inv + soma_x

        # ================= AXON (reads Va, Vs) =================
        axon_I_leak = g_la * (Va - V_l)
        # Jacobi change: I_sa uses the start-of-step soma Vs, NOT a new soma V.
        I_sa = (g_int / p2) * (Va - Vs)
        axon_I_interact = I_sa

        axon_m_inf = 1 / (1 + np.exp(-(Va + 30) / 5.5))
        axon_h_inf = 1 / (1 + np.exp((Va + 60) / 5.8))
        axon_Ina = g_Na_a * axon_m_inf ** 3 * axon_Sodium_h * (Va - V_Na)
        axon_tau_h = 1.5 * np.exp(-(Va + 40) / 33)
        axon_Ik = g_K_a * axon_Potassium_x ** 4 * (Va - V_K)
        axon_alpha_x = 0.13 * (Va + 25) / (1 - np.exp(-(Va + 25) / 10))
        axon_beta_x = 1.69 * np.exp(-(Va + 35) / 80)
        axon_tau_x_inv = axon_alpha_x + axon_beta_x
        axon_x_inf = axon_alpha_x / axon_tau_x_inv

        axon_I_Channels = axon_Ina + axon_Ik
        axon_dv_dt = S * (-(axon_I_leak + axon_I_interact + axon_I_Channels))

        axon_Sodium_h = axon_Sodium_h + delta * (axon_h_inf - axon_Sodium_h) / axon_tau_h
        axon_Potassium_x = delta * (axon_x_inf - axon_Potassium_x) * axon_tau_x_inv + axon_Potassium_x

        # ================= DEND (reads Vd, Vs) =================
        # Current pulse window is a scalar per step, broadcast over all cells.
        dend_I_application = -I_app + (-I_pulse10ms if 200 * sim_seconds < t < 210 * sim_seconds else 0)
        dend_I_leak = g_ld * (Vd - V_l)
        # Jacobi change: interaction uses start-of-step soma Vs.
        dend_I_interact = (g_int / (1 - p1)) * (Vd - Vs)

        dend_Icah = g_CaH * dend_Calcium_r * dend_Calcium_r * (Vd - V_Ca)
        dend_Ikca = g_K_Ca * dend_Potassium_s * (Vd - V_K)
        dend_Ih = g_h * dend_Hcurrent_q * (Vd - V_h)

        # O(N) closed-form gap junction (replaces the O(N^2) j-loop). The j==i
        # term is zero so the self-exclusion falls out automatically.
        if enable_gapjunctions:
            dend_I_gap = C_gap * (n_cells * Vd - Vd.sum())
        else:
            dend_I_gap = 0.0

        # Gating updates (use Vd and old gate / old Ca values).
        dend_alpha_r = 1.7 / (1 + np.exp(-(Vd - 5) / 13.9))
        dend_beta_r = 0.02 * (Vd + 8.5) / (np.exp((Vd + 8.5) / 5) - 1.0)
        dend_tau_r_inv5 = dend_alpha_r + dend_beta_r
        dend_r_inf = dend_alpha_r / dend_tau_r_inv5
        dend_dr_dt = (dend_r_inf - dend_Calcium_r) * dend_tau_r_inv5 * 0.2

        dend_alpha_s = ((0.00002 * dend_Ca2Plus) * (0.00002 * dend_Ca2Plus < 0.01)
                        + 0.01 * (0.00002 * dend_Ca2Plus > 0.01))
        dend_tau_s_inv = dend_alpha_s + 0.015
        dend_s_inf = dend_alpha_s / dend_tau_s_inv
        dend_ds_dt = (dend_s_inf - dend_Potassium_s) * dend_tau_s_inv

        q_inf = 1 / (1 + np.exp((Vd + 80) / 4))
        tau_q_inv = np.exp(-0.086 * Vd - 14.6) + np.exp(0.070 * Vd - 1.87)
        dq_dt = (q_inf - dend_Hcurrent_q) * tau_q_inv

        # Ca concentration update uses dend_Icah (computed above) and old Ca.
        dCa_dt = -3 * dend_Icah - 0.075 * dend_Ca2Plus

        dend_Calcium_r = delta * dend_dr_dt + dend_Calcium_r
        dend_Potassium_s = delta * dend_ds_dt + dend_Potassium_s
        dend_Hcurrent_q = delta * dq_dt + dend_Hcurrent_q
        dend_Ca2Plus = delta * dCa_dt + dend_Ca2Plus

        dend_I_Channels = dend_Icah + dend_Ikca + dend_Ih
        dend_dv_dt = S * (-(dend_I_leak + dend_I_gap + dend_I_interact
                            + dend_I_application + dend_I_Channels))

        # --- double-buffer write: the three coupled V arrays update together ---
        V_soma = Vs + soma_dv_dt * delta
        V_axon = Va + axon_dv_dt * delta
        V_dend = Vd + dend_dv_dt * delta

        t += delta

    return v_trace, n_simsteps


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---> Demo / bench config (mirrors io_model.py defaults).
    sim_seconds = 1.0
    delta = 0.01
    n_cells = 30
    enable_gapjunctions = True
    I_pulse10ms = 2.0
    np.random.seed(1981)  # reproducible run

    # Per-cell variation in low-threshold Ca conductance (as in io_model.py).
    g_CaL = np.random.normal(0.7, 0.1, n_cells)

    # Initial state (same distributions as io_model.py / sweep.build_initial_state).
    V_soma = np.random.uniform(-70, -40, size=(n_cells,))
    soma_k = np.full(n_cells, 0.7423159)
    soma_l = np.full(n_cells, 0.0321349)
    soma_h = np.full(n_cells, 0.3596066)
    soma_n = np.full(n_cells, 0.2369847)
    soma_x = np.full(n_cells, 0.1)
    V_axon = np.random.uniform(-70, -40, size=(n_cells,))
    axon_Sodium_h = np.full(n_cells, 0.9)
    axon_Potassium_x = np.full(n_cells, 0.2369847)
    V_dend = np.random.uniform(-70, -40, size=(n_cells,))
    dend_Ca2Plus = np.full(n_cells, 3.715)
    dend_Calcium_r = np.full(n_cells, 0.0113)
    dend_Potassium_s = np.full(n_cells, 0.0049291)
    dend_Hcurrent_q = np.full(n_cells, 0.0337836)

    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)

    tic = time.process_time()
    v_trace, n_simsteps = simulate(
        V_soma, V_axon, V_dend,
        soma_k, soma_l, soma_h, soma_n, soma_x,
        axon_Sodium_h, axon_Potassium_x,
        dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q,
        g_CaL, n_cells, n_simsteps, delta, sim_seconds,
        enable_gapjunctions=enable_gapjunctions, I_pulse10ms=I_pulse10ms,
    )
    print(f"Simulation execution time: {time.process_time() - tic :.3f} sec.")

    # ---> Plot soma traces (same normalized-stack layout as io_model.py).
    for i in range(n_cells):
        v = v_trace[:, i, 0]
        v = (v - np.nanmean(v)) / (np.nanmax(v) - np.nanmin(v)) / 2
        plt.plot(v_trace[:, i, 3], i + v, color="gray")
    plt.show()
