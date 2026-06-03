#
# Goal B: Numba @njit backend for the Inferior-Olive (de Gruijl) model.
#
# Scalar per-cell loops compiled to machine code (plan recommendation B-(ii)):
# no NumPy array temporaries, so this wins over io_model_vec.py at small N where
# the vec backend is dominated by per-op overhead on tiny (n_cells,) arrays.
#
# Same numerics as io_model_vec.py: Jacobi double-buffer + O(N) closed-form gap.
# validate.py asserts jit-vs-vec equality at machine epsilon ("exact numerics
# preserved" = exact vs the vec backend, both diverging from the GS baseline).
#
# Numba convention (from lab1): @njit(cache=True), all arrays + scalar params
# passed as arguments so the JIT gets a clean typed signature. The model
# constants are module-level floats; Numba freezes them into the compiled code at
# compile time, so they need not be passed as arguments.
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


# Sentinel passed to the njit core when gap junctions are all-to-all (use_knn
# False). The core never indexes it in that path, so an empty (0,0) int array is
# enough to give Numba a concrete int64[:, :] type to compile against.
_NO_NEIGHBOURS = np.empty((0, 0), dtype=np.int64)


def build_neighbours(n_cells, k=8, seed=1981, dims=3):
    """Precompute each cell's k nearest neighbours for *local* gap-junction
    coupling (replaces the all-to-all sum, which both blows up numerically and
    grows O(N) as the population scales -- see the dend gap term below).

    The model has no geometry, so we first scatter the cells at random positions
    in a unit `dims`-cube, then keep the k closest others by Euclidean distance.
    A fixed k means each cell's coupling degree is independent of n_cells, which
    is what removes the explicit-Euler instability AND matches the biology (real
    olivary gap junctions connect a handful of touching dendrites, not the whole
    network).

    Returns an int64 array of shape (n_cells, k_eff) where k_eff = min(k,
    n_cells-1); row i holds the indices of cell i's neighbours. Note the relation
    can be asymmetric (i in nbrs[j] does not imply j in nbrs[i]); symmetrise the
    adjacency if you need strictly bidirectional junctions.
    """
    k_eff = min(k, n_cells - 1)
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n_cells, dims))
    neighbours = np.empty((n_cells, k_eff), dtype=np.int64)

    try:
        # O(N log N) via a KD-tree when SciPy is available.
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        _dist, idx = tree.query(pos, k=k_eff + 1)  # +1: nearest point is self
        idx = np.atleast_2d(idx)
        for i in range(n_cells):
            row = idx[i]
            neighbours[i] = row[row != i][:k_eff]
    except ImportError:
        # NumPy fallback: per-cell argpartition. O(N^2) time but O(N) memory
        # (no full pairwise matrix), so it still runs at large N, just slower.
        for i in range(n_cells):
            d2 = np.sum((pos - pos[i]) ** 2, axis=1)
            d2[i] = np.inf  # exclude self
            nn = np.argpartition(d2, k_eff)[:k_eff]
            neighbours[i] = nn

    return neighbours


@njit(cache=True)
def simulate(
    V_soma, V_axon, V_dend,
    soma_k, soma_l, soma_h, soma_n, soma_x,
    axon_Sodium_h, axon_Potassium_x,
    dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q,
    g_CaL, n_cells, n_simsteps, delta, sim_seconds,
    enable_gapjunctions, I_app, I_pulse10ms, record, record_every=1,
    use_knn=False, neighbours=_NO_NEIGHBOURS,
):
    """Scalar-njit Jacobi backend (same numerics as io_model_vec.simulate).

    All state args are (n_cells,) float64 arrays. Returns (v_trace, n_simsteps);
    v_trace is (n_rec, n_cells, 4) (V_soma, V_axon, V_dend, t) if record, else an
    empty array. record_every logs a row only every Nth step (state still
    advances every step), so n_rec = ceil(n_simsteps / record_every); this keeps
    the buffer bounded at large n_cells. record_every=1 is the original trace.

    Gap junctions: with use_knn False the coupling is all-to-all via the O(N)
    closed form C_gap*(N*vd - Sd). With use_knn True each cell couples only to its
    `neighbours` rows; we precompute a per-cell neighbour-sum once per timestep
    (one O(N*k) pass) so the per-cell gap term is still O(1) at point of use --
    the same precompute-the-sum idea as the all-to-all Sd, restricted to a fixed
    neighbour set. The local args default to use_knn=False / the empty
    _NO_NEIGHBOURS sentinel, so existing callers that pass everything up to
    record_every positionally reproduce the original all-to-all numerics exactly;
    pass use_knn=True with a build_neighbours(...) array to enable local coupling.
    """
    # Copy so the caller's seeded state is not mutated (validate reuses it).
    V_soma = V_soma.copy(); V_axon = V_axon.copy(); V_dend = V_dend.copy()
    soma_k = soma_k.copy(); soma_l = soma_l.copy(); soma_h = soma_h.copy()
    soma_n = soma_n.copy(); soma_x = soma_x.copy()
    axon_Sodium_h = axon_Sodium_h.copy(); axon_Potassium_x = axon_Potassium_x.copy()
    dend_Ca2Plus = dend_Ca2Plus.copy(); dend_Calcium_r = dend_Calcium_r.copy()
    dend_Potassium_s = dend_Potassium_s.copy(); dend_Hcurrent_q = dend_Hcurrent_q.copy()

    if record:
        n_rec = (n_simsteps + record_every - 1) // record_every
        v_trace = np.empty((n_rec, n_cells, 4))
    else:
        v_trace = np.empty((0, 0, 0))
    t = 0.0

    # Per-cell neighbour-sum buffer for the local (kNN) gap path. Allocated once
    # and refilled each timestep; unused (but kept typed) in the all-to-all path.
    nbr_sum = np.zeros(n_cells)
    k_deg = neighbours.shape[1]  # fixed coupling degree per cell (kNN path)

    for i_epoch in range(n_simsteps):
        # --- start-of-step snapshot: all d*/dt this step read ONLY these (Jacobi).
        Vs = V_soma.copy()
        Va = V_axon.copy()
        Vd = V_dend.copy()

        # Log this step only every record_every steps (keeps the buffer bounded).
        do_record = record and (i_epoch % record_every == 0)
        rec = i_epoch // record_every

        # Precompute the gap-coupling sum(s) once per step so the per-cell gap
        # term below is O(1). All-to-all: a single scalar Sd = sum(Vd), then
        # I_gap[i] = C_gap*(N*Vd[i] - Sd). Local (kNN): a per-cell neighbour-sum
        # nbr_sum[i] = sum over i's k neighbours, then I_gap[i] =
        # C_gap*(k*Vd[i] - nbr_sum[i]) -- same idea, fixed neighbour set.
        Sd = 0.0
        if enable_gapjunctions:
            if use_knn:
                for i in range(n_cells):
                    s = 0.0
                    for jj in range(k_deg):
                        s += Vd[neighbours[i, jj]]
                    nbr_sum[i] = s
            else:
                for j in range(n_cells):
                    Sd += Vd[j]

        # Current pulse window is a scalar per step (broadcast to all cells).
        pulse = -I_pulse10ms if (200 * sim_seconds < t and t < 210 * sim_seconds) else 0.0

        for i in range(n_cells):
            vs = Vs[i]; va = Va[i]; vd = Vd[i]  # this cell's start-of-step V

            if do_record:
                v_trace[rec, i, 0] = vs
                v_trace[rec, i, 1] = va
                v_trace[rec, i, 2] = vd
                v_trace[rec, i, 3] = t

            # ---------------- SOMA (reads snapshot vs, va, vd) ----------------
            soma_I_leak = g_ls * (vs - V_l)
            soma_I_interact = (g_int / p1) * (vs - vd) + (g_int / (1 - p2)) * (vs - va)
            soma_Ical = g_CaL[i] * soma_k[i] * soma_k[i] * soma_k[i] * soma_l[i] * (vs - V_Ca)
            soma_m_inf = 1 / (1 + np.exp(-(vs + 30) / 5.5))
            soma_Ina = g_Na_s * soma_m_inf ** 3 * soma_h[i] * (vs - V_Na)
            soma_Ikdr = g_Kdr_s * soma_n[i] ** 4 * (vs - V_K)
            soma_Ik = g_K_s * soma_x[i] ** 4 * (vs - V_K)
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
            soma_k[i] = delta * (soma_k_inf - soma_k[i]) + soma_k[i]
            soma_l[i] = delta * (soma_l_inf - soma_l[i]) / soma_tau_l + soma_l[i]
            soma_h[i] = soma_h[i] + delta * (soma_h_inf - soma_h[i]) / soma_tau_h
            soma_n[i] = delta * (soma_n_inf - soma_n[i]) / soma_tau_n + soma_n[i]
            soma_x[i] = delta * (soma_x_inf - soma_x[i]) * soma_tau_x_inv + soma_x[i]

            # ---------------- AXON (reads snapshot va, vs) ----------------
            axon_I_leak = g_la * (va - V_l)
            I_sa = (g_int / p2) * (va - vs)  # Jacobi: snapshot soma vs, not new V
            axon_m_inf = 1 / (1 + np.exp(-(va + 30) / 5.5))
            axon_h_inf = 1 / (1 + np.exp((va + 60) / 5.8))
            axon_Ina = g_Na_a * axon_m_inf ** 3 * axon_Sodium_h[i] * (va - V_Na)
            axon_tau_h = 1.5 * np.exp(-(va + 40) / 33)
            axon_Ik = g_K_a * axon_Potassium_x[i] ** 4 * (va - V_K)
            axon_alpha_x = 0.13 * (va + 25) / (1 - np.exp(-(va + 25) / 10))
            axon_beta_x = 1.69 * np.exp(-(va + 35) / 80)
            axon_tau_x_inv = axon_alpha_x + axon_beta_x
            axon_x_inf = axon_alpha_x / axon_tau_x_inv
            axon_dv_dt = S * (-(axon_I_leak + I_sa + axon_Ina + axon_Ik))
            axon_Sodium_h[i] = axon_Sodium_h[i] + delta * (axon_h_inf - axon_Sodium_h[i]) / axon_tau_h
            axon_Potassium_x[i] = delta * (axon_x_inf - axon_Potassium_x[i]) * axon_tau_x_inv + axon_Potassium_x[i]

            # ---------------- DEND (reads snapshot vd, vs) ----------------
            dend_I_application = -I_app + pulse
            dend_I_leak = g_ld * (vd - V_l)
            dend_I_interact = (g_int / (1 - p1)) * (vd - vs)  # Jacobi: snapshot vs
            dend_Icah = g_CaH * dend_Calcium_r[i] * dend_Calcium_r[i] * (vd - V_Ca)
            dend_Ikca = g_K_Ca * dend_Potassium_s[i] * (vd - V_K)
            dend_Ih = g_h * dend_Hcurrent_q[i] * (vd - V_h)
            if enable_gapjunctions:
                if use_knn:
                    dend_I_gap = C_gap * (k_deg * vd - nbr_sum[i])  # local, O(1)
                else:
                    dend_I_gap = C_gap * (n_cells * vd - Sd)  # all-to-all, O(1)
            else:
                dend_I_gap = 0.0

            dend_alpha_r = 1.7 / (1 + np.exp(-(vd - 5) / 13.9))
            dend_beta_r = 0.02 * (vd + 8.5) / (np.exp((vd + 8.5) / 5) - 1.0)
            dend_tau_r_inv5 = dend_alpha_r + dend_beta_r
            dend_r_inf = dend_alpha_r / dend_tau_r_inv5
            dend_dr_dt = (dend_r_inf - dend_Calcium_r[i]) * dend_tau_r_inv5 * 0.2
            ca = dend_Ca2Plus[i]
            dend_alpha_s = ((0.00002 * ca) * (1.0 if 0.00002 * ca < 0.01 else 0.0)
                            + 0.01 * (1.0 if 0.00002 * ca > 0.01 else 0.0))
            dend_tau_s_inv = dend_alpha_s + 0.015
            dend_s_inf = dend_alpha_s / dend_tau_s_inv
            dend_ds_dt = (dend_s_inf - dend_Potassium_s[i]) * dend_tau_s_inv
            q_inf = 1 / (1 + np.exp((vd + 80) / 4))
            tau_q_inv = np.exp(-0.086 * vd - 14.6) + np.exp(0.070 * vd - 1.87)
            dq_dt = (q_inf - dend_Hcurrent_q[i]) * tau_q_inv
            dCa_dt = -3 * dend_Icah - 0.075 * dend_Ca2Plus[i]
            dend_Calcium_r[i] = delta * dend_dr_dt + dend_Calcium_r[i]
            dend_Potassium_s[i] = delta * dend_ds_dt + dend_Potassium_s[i]
            dend_Hcurrent_q[i] = delta * dq_dt + dend_Hcurrent_q[i]
            dend_Ca2Plus[i] = delta * dCa_dt + dend_Ca2Plus[i]
            dend_dv_dt = S * (-(dend_I_leak + dend_I_gap + dend_I_interact
                                + dend_I_application + dend_Icah + dend_Ikca + dend_Ih))

            # --- double-buffer write (three coupled V update from the snapshot) ---
            V_soma[i] = vs + soma_dv_dt * delta
            V_axon[i] = va + axon_dv_dt * delta
            V_dend[i] = vd + dend_dv_dt * delta

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

    # Local gap-junction coupling (default, matching sweep.py): each cell couples
    # to only its K nearest neighbours, which stays stable at large n_cells. Set
    # USE_KNN=False to fall back to the original all-to-all coupling.
    USE_KNN = True
    K = 8
    neighbours = build_neighbours(n_cells, k=K, seed=1981) if USE_KNN else _NO_NEIGHBOURS

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
    simulate(*args, 2, delta, sim_seconds, enable_gapjunctions, I_app, I_pulse10ms,
             False, use_knn=USE_KNN, neighbours=neighbours)

    n_simsteps = int(sim_seconds * 1000 / delta + 0.5)

    tic = time.process_time()
    v_trace, _ = simulate(*args, n_simsteps, delta, sim_seconds,
                          enable_gapjunctions, I_app, I_pulse10ms, True,
                          use_knn=USE_KNN, neighbours=neighbours)
    print(f"Simulation execution time: {time.process_time() - tic :.3f} sec.")

    for i in range(n_cells):
        v = v_trace[:, i, 0]
        v = (v - np.nanmean(v)) / (np.nanmax(v) - np.nanmin(v)) / 2
        plt.plot(v_trace[:, i, 3], i + v, color="gray")
    plt.show()
