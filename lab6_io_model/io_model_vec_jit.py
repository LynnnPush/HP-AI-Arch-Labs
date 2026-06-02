#
# Goal B, variant (i): Numba @njit applied to the VECTORIZED timestep.
#
# This is the "JIT the vec version" option: same array-based Jacobi double-buffer
# + O(N) closed-form gap as io_model_vec.py, but the whole loop is compiled with
# @njit(cache=True). It still builds (n_cells,) NumPy temporaries every step, so
# at small N those allocations dominate and the speedup over plain vec is modest
# -- contrast io_model_jit.py, whose scalar loops avoid the temporaries entirely.
# All three (vec, vec_jit, jit) share identical numerics, so validate.py can
# assert equality between them at machine epsilon.
#
# Numba-compat tweaks vs io_model_vec.simulate:
#   * v_trace is always an ndarray (empty((0,0,0)) when record=False) -- njit
#     can't unify None with an array.
#   * the dend_alpha_s clamp uses np.where instead of bool-array * float.
#   * params are all positional (matches io_model_jit.simulate) for uniform calls.
#

import time

import numpy as np
from numba import njit

# ---> Model constants (de Gruijl Inferior-Olive); identical to io_model.py.
g_int   = 0.13
p1      = 0.25
p2      = 0.15
g_h     = 0.12
g_K_Ca  = 35.0
g_ld    = 0.01532
g_la    = 0.016
g_ls    = 0.004
S       = 1.0
g_Na_s  = 150.0
g_Kdr_s = 9.0
g_K_s   = 5.0
g_CaH   = 4.5
g_Na_a  = 240.0
g_K_a   = 240.0
V_Na    = 55.0
V_K     = -75.0
V_Ca    = 120.0
V_h     = -43.0
V_l     = 10.0
C_gap   = 0.05


@njit(cache=True)
def simulate(
    V_soma, V_axon, V_dend,
    soma_k, soma_l, soma_h, soma_n, soma_x,
    axon_Sodium_h, axon_Potassium_x,
    dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q,
    g_CaL, n_cells, n_simsteps, delta, sim_seconds,
    enable_gapjunctions, I_app, I_pulse10ms, record,
):
    """njit-compiled VECTORIZED Jacobi backend (same numerics as io_model_vec).

    All state args are (n_cells,) float64 arrays. Returns (v_trace, n_simsteps);
    v_trace is (n_simsteps, n_cells, 4) (V_soma, V_axon, V_dend, t) if record,
    else an empty array.
    """
    # Copy so the caller's seeded state is not mutated (validate reuses it).
    V_soma = V_soma.copy(); V_axon = V_axon.copy(); V_dend = V_dend.copy()
    soma_k = soma_k.copy(); soma_l = soma_l.copy(); soma_h = soma_h.copy()
    soma_n = soma_n.copy(); soma_x = soma_x.copy()
    axon_Sodium_h = axon_Sodium_h.copy(); axon_Potassium_x = axon_Potassium_x.copy()
    dend_Ca2Plus = dend_Ca2Plus.copy(); dend_Calcium_r = dend_Calcium_r.copy()
    dend_Potassium_s = dend_Potassium_s.copy(); dend_Hcurrent_q = dend_Hcurrent_q.copy()

    if record:
        v_trace = np.empty((n_simsteps, n_cells, 4))
    else:
        v_trace = np.empty((0, 0, 0))
    t = 0.0

    for i_epoch in range(n_simsteps):
        # --- start-of-step snapshot: all d*/dt this step read ONLY these (Jacobi).
        Vs, Va, Vd = V_soma, V_axon, V_dend

        if record:
            v_trace[i_epoch, :, 0] = Vs
            v_trace[i_epoch, :, 1] = Va
            v_trace[i_epoch, :, 2] = Vd
            v_trace[i_epoch, :, 3] = t

        # ================= SOMA (reads Vs, Va, Vd) =================
        soma_I_leak = g_ls * (Vs - V_l)
        soma_I_interact = (g_int / p1) * (Vs - Vd) + (g_int / (1 - p2)) * (Vs - Va)
        soma_Ical = g_CaL * soma_k * soma_k * soma_k * soma_l * (Vs - V_Ca)
        soma_m_inf = 1 / (1 + np.exp(-(Vs + 30) / 5.5))
        soma_Ina = g_Na_s * soma_m_inf ** 3 * soma_h * (Vs - V_Na)
        soma_Ikdr = g_Kdr_s * soma_n ** 4 * (Vs - V_K)
        soma_Ik = g_K_s * soma_x ** 4 * (Vs - V_K)
        soma_dv_dt = S * (-(soma_I_leak + soma_I_interact
                            + soma_Ik + soma_Ikdr + soma_Ina + soma_Ical))

        soma_k_inf = 1 / (1 + np.exp(-(Vs + 61) / 4.2))
        soma_l_inf = 1 / (1 + np.exp((Vs + 85) / 8.5))
        soma_tau_l = (20 * np.exp((Vs + 160) / 30) / (1 + np.exp((Vs + 84) / 7.3))) + 35
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
        I_sa = (g_int / p2) * (Va - Vs)  # Jacobi: snapshot soma Vs, not new V
        axon_m_inf = 1 / (1 + np.exp(-(Va + 30) / 5.5))
        axon_h_inf = 1 / (1 + np.exp((Va + 60) / 5.8))
        axon_Ina = g_Na_a * axon_m_inf ** 3 * axon_Sodium_h * (Va - V_Na)
        axon_tau_h = 1.5 * np.exp(-(Va + 40) / 33)
        axon_Ik = g_K_a * axon_Potassium_x ** 4 * (Va - V_K)
        axon_alpha_x = 0.13 * (Va + 25) / (1 - np.exp(-(Va + 25) / 10))
        axon_beta_x = 1.69 * np.exp(-(Va + 35) / 80)
        axon_tau_x_inv = axon_alpha_x + axon_beta_x
        axon_x_inf = axon_alpha_x / axon_tau_x_inv
        axon_dv_dt = S * (-(axon_I_leak + I_sa + axon_Ina + axon_Ik))

        axon_Sodium_h = axon_Sodium_h + delta * (axon_h_inf - axon_Sodium_h) / axon_tau_h
        axon_Potassium_x = delta * (axon_x_inf - axon_Potassium_x) * axon_tau_x_inv + axon_Potassium_x

        # ================= DEND (reads Vd, Vs) =================
        pulse = -I_pulse10ms if (200 * sim_seconds < t and t < 210 * sim_seconds) else 0.0
        dend_I_application = -I_app + pulse
        dend_I_leak = g_ld * (Vd - V_l)
        dend_I_interact = (g_int / (1 - p1)) * (Vd - Vs)  # Jacobi: snapshot Vs
        dend_Icah = g_CaH * dend_Calcium_r * dend_Calcium_r * (Vd - V_Ca)
        dend_Ikca = g_K_Ca * dend_Potassium_s * (Vd - V_K)
        dend_Ih = g_h * dend_Hcurrent_q * (Vd - V_h)

        # O(N) closed-form gap junction (j==i term is zero -> self-exclusion).
        if enable_gapjunctions:
            dend_I_gap = C_gap * (n_cells * Vd - Vd.sum())
        else:
            dend_I_gap = np.zeros(n_cells)

        dend_alpha_r = 1.7 / (1 + np.exp(-(Vd - 5) / 13.9))
        dend_beta_r = 0.02 * (Vd + 8.5) / (np.exp((Vd + 8.5) / 5) - 1.0)
        dend_tau_r_inv5 = dend_alpha_r + dend_beta_r
        dend_r_inf = dend_alpha_r / dend_tau_r_inv5
        dend_dr_dt = (dend_r_inf - dend_Calcium_r) * dend_tau_r_inv5 * 0.2

        # alpha_s clamp via np.where (njit-friendly vs bool-array * float).
        tmp = 0.00002 * dend_Ca2Plus
        dend_alpha_s = np.where(tmp < 0.01, tmp, 0.0) + np.where(tmp > 0.01, 0.01, 0.0)
        dend_tau_s_inv = dend_alpha_s + 0.015
        dend_s_inf = dend_alpha_s / dend_tau_s_inv
        dend_ds_dt = (dend_s_inf - dend_Potassium_s) * dend_tau_s_inv

        q_inf = 1 / (1 + np.exp((Vd + 80) / 4))
        tau_q_inv = np.exp(-0.086 * Vd - 14.6) + np.exp(0.070 * Vd - 1.87)
        dq_dt = (q_inf - dend_Hcurrent_q) * tau_q_inv
        dCa_dt = -3 * dend_Icah - 0.075 * dend_Ca2Plus

        dend_Calcium_r = delta * dend_dr_dt + dend_Calcium_r
        dend_Potassium_s = delta * dend_ds_dt + dend_Potassium_s
        dend_Hcurrent_q = delta * dq_dt + dend_Hcurrent_q
        dend_Ca2Plus = delta * dCa_dt + dend_Ca2Plus

        dend_dv_dt = S * (-(dend_I_leak + dend_I_gap + dend_I_interact
                            + dend_I_application + dend_Icah + dend_Ikca + dend_Ih))

        # --- double-buffer write: the three coupled V arrays update together ---
        V_soma = Vs + soma_dv_dt * delta
        V_axon = Va + axon_dv_dt * delta
        V_dend = Vd + dend_dv_dt * delta

        t += delta

    return v_trace, n_simsteps


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim_seconds = 1.0
    delta = 0.01
    n_cells = 30
    enable_gapjunctions = True
    I_app = 0.0
    I_pulse10ms = 2.0
    np.random.seed(1981)

    g_CaL = np.random.normal(0.7, 0.1, n_cells)
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

    args = (V_soma, V_axon, V_dend, soma_k, soma_l, soma_h, soma_n, soma_x,
            axon_Sodium_h, axon_Potassium_x, dend_Ca2Plus, dend_Calcium_r,
            dend_Potassium_s, dend_Hcurrent_q, g_CaL, n_cells)

    # Warm-up on a tiny problem (n_simsteps=2) to move JIT compilation OUT of the
    # timed region; cache=True amortizes it across future runs (lab1 convention).
    simulate(*args, 2, delta, sim_seconds, enable_gapjunctions, I_app, I_pulse10ms, False)

    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)

    tic = time.process_time()
    v_trace, _ = simulate(*args, n_simsteps, delta, sim_seconds,
                          enable_gapjunctions, I_app, I_pulse10ms, True)
    print(f"Simulation execution time: {time.process_time() - tic :.3f} sec.")

    for i in range(n_cells):
        v = v_trace[:, i, 0]
        v = (v - np.nanmean(v)) / (np.nanmax(v) - np.nanmin(v)) / 2
        plt.plot(v_trace[:, i, 3], i + v, color="gray")
    plt.show()
