#
# Original Author: Lennart Landsmeer
# Adapted for the CESE5040 course by Max Engelen
#
# TODO fix plotting in main() function
#

import matplotlib.pyplot as plt
import time
import numpy as np
#np.random.seed(1981) # uncomment if you want reproducible simulation runs


#---> Parameters
sim_seconds = 1.0
delta = 0.01
n_cells = 30
enable_gapjunctions = True


#---> Constant values need the for the de Gruill Model (Inferior Olive)
g_int   = 0.13    # Cell internal conductance  -- now a parameter (0.13)
p1      = 0.25    # Cell surface ratio soma/dendrite
p2      = 0.15    # Cell surface ratio axon(hillock)/soma
g_h     = 0.12    # H current (HCN) (0.4996)
g_K_Ca  = 35.0    # Potassium  (KCa v1.1 - BK) (35)
g_ld    = 0.01532 # Leak dendrite (0.016)
g_la    = 0.016   # Leak axon (0.016)
g_ls    = 0.004   # Leak soma (0.016)
S       = 1.0     # 1/C_m, cm^2/uF
g_Na_s  = 150.0   # Sodium  - (Na v1.6)
g_Kdr_s = 9.0     # Potassium - (K v4.3)
g_K_s   = 5.0     # Potassium - (K v3.4)
g_CaH   = 4.5     # High-threshold calcium -- Ca V2.1
g_Na_a  = 240.0   # Sodium
g_K_a   = 240.0   # Potassium (20)
V_Na    = 55.0    # Sodium
V_K     = -75.0   # Potassium
V_Ca    = 120.0   # Low-threshold calcium channel
V_h     = -43.0   # H current
V_l     = 10.0    # Leak
C_gap   = 0.05    # Gap conductance
I_app   = 0.0
I_pulse10ms = 2.0

def update_soma(i, v_trace, at, V_soma, V_axon, V_dend, soma_k, soma_l, soma_h, soma_n, soma_x, g_CaL):
    'Perform a single soma timestep update'

    # CURRENT: Soma leak current (ls)
    soma_I_leak = g_ls * (V_soma[i] - V_l)

    # CURRENT: Soma interaction current (ds, as)
    I_ds = (g_int / p1) * (V_soma[i] - V_dend[i])
    I_as = (g_int / (1 - p2)) * (V_soma[i] - V_axon[i])
    soma_I_interact = I_ds + I_as

    # CHANNEL: Soma Low-threshold calcium (CaL)
    soma_Ical = g_CaL[i] * soma_k[i] * soma_k[i] * \
        soma_k[i] * soma_l[i] * (V_soma[i] - V_Ca)

    soma_k_inf = 1 / (1 + np.exp(-(V_soma[i] + 61)/4.2))
    soma_l_inf = 1 / (1 + np.exp((V_soma[i] + 85)/8.5))
    soma_tau_l = (20 * np.exp((V_soma[i] + 160)/30) /
                  (1 + np.exp((V_soma[i] + 84) / 7.3))) + 35

    soma_dk_dt = soma_k_inf - soma_k[i]
    soma_dl_dt = (soma_l_inf - soma_l[i]) / soma_tau_l
    soma_k[i] = delta * soma_dk_dt + soma_k[i]
    soma_l[i] = delta * soma_dl_dt + soma_l[i]

    # CHANNEL: Soma sodium (Na_s)
    # watch out direct gate: m = m_inf
    soma_m_inf = 1 / (1 + np.exp(-(V_soma[i] + 30)/5.5))
    soma_h_inf = 1 / (1 + np.exp((V_soma[i] + 70)/5.8))
    soma_Ina = g_Na_s * soma_m_inf**3 * soma_h[i] * (V_soma[i] - V_Na)
    soma_tau_h = 3 * np.exp(-(V_soma[i] + 40)/33)
    soma_dh_dt = (soma_h_inf - soma_h[i]) / soma_tau_h
    soma_h[i] = soma_h[i] + delta * soma_dh_dt

    # CHANNEL: Soma potassium, slow component (Kdr)
    soma_Ikdr = g_Kdr_s * soma_n[i]**4 * (V_soma[i] - V_K)
    soma_n_inf = 1 / (1 + np.exp(-(V_soma[i] + 3)/10))
    soma_tau_n = 5 + (47 * np.exp((V_soma[i] + 50)/900))
    soma_dn_dt = (soma_n_inf - soma_n[i]) / soma_tau_n
    soma_n[i] = delta * soma_dn_dt + soma_n[i]

    # CHANNEL: Soma potassium, fast component (K_s)
    soma_Ik = g_K_s * soma_x[i]**4 * (V_soma[i] - V_K)
    soma_alpha_x = 0.13 * (V_soma[i] + 25) / (1 - np.exp(-(V_soma[i] + 25)/10))
    soma_beta_x = 1.69 * np.exp(-(V_soma[i] + 35)/80)
    soma_tau_x_inv = soma_alpha_x + soma_beta_x
    soma_x_inf = soma_alpha_x / soma_tau_x_inv

    soma_dx_dt = (soma_x_inf - soma_x[i]) * soma_tau_x_inv
    soma_x[i] = delta * soma_dx_dt + soma_x[i]

    # RECORD: Soma Potential
    if at >= 0:
        v_trace[at, 0] = V_soma[i]

    # UPDATE: Soma compartment update (V_soma)
    soma_I_Channels = soma_Ik + soma_Ikdr + soma_Ina + soma_Ical
    soma_dv_dt = S * (-(soma_I_leak + soma_I_interact + soma_I_Channels))
    V_soma[i] = V_soma[i] + soma_dv_dt * delta


def update_axon(i, v_trace, at, V_soma, V_axon,  axon_Sodium_h, axon_Potassium_x):
    'Perform a single axon-hillock timestep update'

    # CURRENT: Axon leak current (la)
    axon_I_leak = g_la * (V_axon[i] - V_l)

    # CURRENT: Axon interaction current (sa)
    I_sa = (g_int / p2) * (V_axon[i] - V_soma[i])
    axon_I_interact = I_sa

    # CHANNEL: Axon sodium (Na_a)
    # watch out direct gate: m = m_inf
    axon_m_inf = 1 / (1 + np.exp(-(V_axon[i]+30)/5.5))
    axon_h_inf = 1 / (1 + np.exp((V_axon[i]+60)/5.8))
    axon_Ina = g_Na_a * axon_m_inf**3 * axon_Sodium_h[i] * (V_axon[i] - V_Na)
    axon_tau_h = 1.5 * np.exp(-(V_axon[i]+40)/33)
    axon_dh_dt = (axon_h_inf - axon_Sodium_h[i]) / axon_tau_h
    axon_Sodium_h[i] = axon_Sodium_h[i] + delta * axon_dh_dt

    # CHANNEL: Axon potassium (K_a)
    axon_Ik = g_K_a * axon_Potassium_x[i]**4 * (V_axon[i] - V_K)
    axon_alpha_x = 0.13*(V_axon[i] + 25) / (1 - np.exp(-(V_axon[i] + 25)/10))
    axon_beta_x = 1.69 * np.exp(-(V_axon[i] + 35)/80)
    axon_tau_x_inv = axon_alpha_x + axon_beta_x
    axon_x_inf = axon_alpha_x / axon_tau_x_inv
    axon_dx_dt = (axon_x_inf - axon_Potassium_x[i]) * axon_tau_x_inv
    axon_Potassium_x[i] = delta * axon_dx_dt + axon_Potassium_x[i]

    # RECORD: Axon Potential
    if at >= 0:
        v_trace[at, 1] = V_axon[i]

    # UPDATE: Axon-hillock compartment update (V_axon[i])
    axon_I_Channels = axon_Ina + axon_Ik
    dv_dt = S * (-(axon_I_leak + axon_I_interact + axon_I_Channels))
    V_axon[i] = V_axon[i] + dv_dt * delta


def update_dend(i, v_trace, at, t, V_soma,  V_dend, dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q):
    'Perform a single denrite timestep update'

    # CURRENT: Dend application current (I_app, I_pulse10ms)
    dend_I_application = -I_app + (-I_pulse10ms if 200 * sim_seconds < t < 210 * sim_seconds else 0)

    # CURRENT: Dend leak current (ld)
    dend_I_leak = g_ld * (V_dend[i] - V_l)

    # CURRENT: Dend interaction Current (sd)
    dend_I_interact = (g_int / (1 - p1)) * (V_dend[i] - V_soma[i])

    # CHANNEL: Dend high-threshold calcium (CaH)
    dend_Icah = g_CaH * dend_Calcium_r[i] * \
        dend_Calcium_r[i] * (V_dend[i] - V_Ca)
    dend_alpha_r = 1.7 / (1 + np.exp(-(V_dend[i] - 5)/13.9))
    dend_beta_r = 0.02*(V_dend[i] + 8.5) / (np.exp((V_dend[i] + 8.5)/5) - 1.0)
    dend_tau_r_inv5 = (dend_alpha_r + dend_beta_r)  # tau = 5 / (alpha + beta)
    dend_r_inf = dend_alpha_r / dend_tau_r_inv5
    dend_dr_dt = (dend_r_inf - dend_Calcium_r[i]) * dend_tau_r_inv5 * 0.2
    dend_Calcium_r[i] = delta * dend_dr_dt + dend_Calcium_r[i]

    # CHANNEL: Dend calcium-dependent potassium (KCa)
    dend_Ikca = g_K_Ca * dend_Potassium_s[i] * (V_dend[i] - V_K)
    dend_alpha_s = (0.00002 * dend_Ca2Plus[i]) * (0.00002 * dend_Ca2Plus[i] < 0.01) + 0.01*(0.00002 * dend_Ca2Plus[i] > 0.01)
    dend_tau_s_inv = dend_alpha_s + 0.015
    dend_s_inf = dend_alpha_s / dend_tau_s_inv
    dend_ds_dt = (dend_s_inf - dend_Potassium_s[i]) * dend_tau_s_inv
    dend_Potassium_s[i] = delta * dend_ds_dt + dend_Potassium_s[i]

    # CHANNEL: Dend proton (h)
    dend_Ih = g_h * dend_Hcurrent_q[i] * (V_dend[i] - V_h)
    q_inf = 1 / (1 + np.exp((V_dend[i] + 80)/4))
    tau_q_inv = np.exp(-0.086*V_dend[i] - 14.6) + np.exp(0.070*V_dend[i] - 1.87)
    dq_dt = (q_inf - dend_Hcurrent_q[i]) * tau_q_inv
    dend_Hcurrent_q[i] = delta * dq_dt + dend_Hcurrent_q[i]

    #CURRENT: gap junctions only connect to all neighbors (generally different topologies are possible)
    dend_I_gap = 0
    if enable_gapjunctions:
        for j in range(n_cells):
            if j != i:
                dend_I_gap += C_gap * ((V_dend[i] - V_dend[j]))

    # CONCENTRATION: Dend calcium concentration (CaPlus)
    dCa_dt = -3 * dend_Icah - 0.075 * dend_Ca2Plus[i]
    dend_Ca2Plus[i] = delta * dCa_dt + dend_Ca2Plus[i]

    # RECORD: Dend potential variables
    if at >= 0:
        v_trace[at, 2] = V_dend[i]

    # UPDATE: Dend compartment update (V_dend[i])
    dend_I_Channels = dend_Icah + dend_Ikca + dend_Ih
    dend_dv_dt = S * (-(dend_I_leak + dend_I_gap + dend_I_interact + dend_I_application + dend_I_Channels))
    V_dend[i] = V_dend[i] + dend_dv_dt * delta


if __name__ == '__main__':

    #---> Initial all values needed for the simulation run
    # Calcium T - (CaV 3.1) (0.7)
    # Add some per-cell variation for varying frequencies
    g_CaL   = np.random.normal(0.7, 0.1, n_cells)

    # Soma state
    #V_soma = [-60.0] * n_cells # replace with next line for randomizing cell somata
    V_soma = np.random.uniform(low=-70, high=-40, size=(n_cells,))
    soma_k = np.array([0.7423159] * n_cells)
    soma_l = np.array([0.0321349] * n_cells)
    soma_h = np.array([0.3596066] * n_cells)
    soma_n = np.array([0.2369847] * n_cells)
    soma_x = np.array([0.1] * n_cells)

    # Axon state
    #V_axon = [-60.0] * n_cells # replace with next line for randomizing cell axons
    V_axon = np.random.uniform(low=-70, high=-40, size=(n_cells,))
    axon_Sodium_h = np.array([0.9] * n_cells)
    axon_Potassium_x = np.array([0.2369847] * n_cells)

    # Dend state
    #V_dend = [-60.0] * n_cells # replace with next line for randomizing cell dendrites
    V_dend = np.random.uniform(low=-70, high=-40, size=(n_cells,))
    dend_Ca2Plus = np.array([3.715] * n_cells)
    dend_Calcium_r = np.array([0.0113] * n_cells)
    dend_Potassium_s = np.array([0.0049291] * n_cells)
    dend_Hcurrent_q = np.array([0.0337836] * n_cells)

    # Help variables
    # define the amount of sim steps
    n_simsteps = int(sim_seconds*1000 / delta + .5)
    v_trace = [np.empty((n_simsteps, 4)) for _ in range(n_cells)]  # store the voltage trace
    t = 0


    #---> Run the simulation
    tic = time.process_time()
    # Recorded simulation loop (record single cell (id = 0) for simplicity)
    for i_epoch in range(n_simsteps):

        # do the calculations for each step in the system
        for i_cell in range(n_cells):
            update_soma(i_cell, v_trace[i_cell], i_epoch, V_soma, V_axon, V_dend, soma_k, soma_l, soma_h, soma_n, soma_x, g_CaL)
            update_axon(i_cell, v_trace[i_cell], i_epoch, V_soma, V_axon, axon_Sodium_h, axon_Potassium_x)
            update_dend(i_cell, v_trace[i_cell], i_epoch, t, V_soma, V_dend, dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q)
            v_trace[i_cell][i_epoch, -1] = t

        #increase t by delta for next sim step.
        t += delta

    print(f'Simulation execution time: {time.process_time()-tic :.3f} sec.')

    #---> Plot the traces (TODO: Can be further improved)
    for i in range(n_cells):
        v = v_trace[i][:, 0]
        print(v)
        v = (v-np.nanmean(v))/(np.nanmax(v)-np.nanmin(v))/2
        plt.plot(v_trace[i][:, 3], i + v, color='gray')
    plt.show()
