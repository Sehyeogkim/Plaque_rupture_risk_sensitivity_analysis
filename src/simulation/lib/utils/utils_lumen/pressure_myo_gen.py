#!/usr/bin/env python3
"""
Standalone script to create P_myo.dat by solving a lumped-parameter heart ODE model.

Reads:
    ./P_in.dat      — aortic pressure waveform (1001 rows: time pressure_Pa)
    ./elastance.dat — normalized elastance     (1001 rows: time E_norm)

Writes:
    ./P_myo.dat      — ventricular pressure waveform (1001 rows: time pressure_Pa)
    ./P_myo_plot.png — comparison plot of P_in and P_myo (mmHg vs seconds)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Heart model constants ────────────────────────────────────────────────────
MMHG_TO_CGS = 1333.22
E_MAX = 2.0 * MMHG_TO_CGS            # max elastance (dyne/cm^2 per mL)
CARDIAC_OUTPUT = 5 * 1000 / 60       # 5 L/min → cc/s
ATRIAL_ELASTANCE = 250.0
ATRIAL_RESISTANCE = 5.0
ATRIAL_INDUCTANCE = 0.1
VENTRICULAR_RESISTANCE = 40.0
VENTRICULAR_INDUCTANCE = 0.01
T_SYS = 0.4
HR = 60.0                               # heart rate [bpm]

def generate_elastance(norm_file: str):
    """
    Generate real Elastance.dat from normalized elastance data.

    Normalized data assumption:
    - x-axis: t / t_sys (so x=0~1 is systole, x=1~x_max is diastole)
    - y-axis: E / E_max

    Two-zone time rescaling:
    Zone 1 (systole):  x in [0, 1]       -> t in [0, t_sys]
    Zone 2 (diastole): x in [1, x_max]   -> t in [t_sys, T_total]

    """
    # --- Load normalized data ---
    data = np.loadtxt(norm_file)
    x_norm = data[:, 0]  # t / t_sys
    E_norm = data[:, 1]  # E / E_max

    # --- Compute total cardiac cycle time ---
    T_total = 60.0 / HR
    t_dias = T_total - T_SYS

    if t_dias <= 0:
        raise ValueError(
            f"T_SYS ({T_SYS}s) >= T_total ({T_total}s). "
            f"Check HR ({HR} bpm) and T_SYS."
        )

    # --- Find boundary index (x = 1.0, end of systole) ---
    x_max = x_norm[-1]
    idx_sys = np.searchsorted(x_norm, 1.0, side="right")
    # include the boundary point in both zones for continuity

    # --- Zone 1: Systole  x in [0, 1] -> t in [0, t_sys] ---
    x_sys = x_norm[:idx_sys]
    t_sys_zone = x_sys * T_SYS

    # --- Zone 2: Diastole  x in [1, x_max] -> t in [t_sys, T_total] ---
    x_dia = x_norm[idx_sys - 1 :]  # overlap at boundary for continuity
    t_dia_zone = T_SYS + (x_dia - 1.0) / (x_max - 1.0) * t_dias

    # --- Combine (remove duplicate at boundary) ---
    t_real = np.concatenate([t_sys_zone, t_dia_zone[1:]])
    E_real_norm = np.concatenate([E_norm[:idx_sys], E_norm[idx_sys:]])

    # --- Scale y-axis ---
    E_real = E_real_norm * E_MAX

    print(f"  HR = {HR} bpm -> T_total = {T_total:.4f} s")
    print(f"  T_SYS = {T_SYS:.4f} s, t_dias = {t_dias:.4f} s")
    print(f"  E_MAX = {E_MAX} mmHg/mL")
    print(f"  Data points: {len(t_real)}")
    print(f"  Time range: [{t_real[0]:.4f}, {t_real[-1]:.4f}] s")

    return t_real, E_real

def SOLVE_HEART_ODE(P_in_path: str, 
                    elastance_path: str, 
                    n_cycles: int = 20, 
                    myo_scale_factor: float = 1.0, 
                    P_myo_path: str = None, 
                    plot: bool = False) -> None:
    """Solve lumped-parameter heart ODE and return P_myo_path."""

    # --- 1. Load input data ---
    print(f"T_SYS = {T_SYS:.4f} s, HR = {HR:.1f} bpm")
    P_in_data = np.loadtxt(P_in_path)        # [t, P] pairs, 1001 points

    time_cycle = P_in_data[:, 0]
    aortic_pressure_cycle = P_in_data[:, 1]
    steps_per_cycle = len(time_cycle)
    cycle_duration = time_cycle[-1]
    del_t = cycle_duration / (steps_per_cycle - 1)

    E_time, E_values = generate_elastance(elastance_path)

    # --- 2. Interpolate elastance to cardiac cycle time base ---
    E_cycle = np.interp(time_cycle, E_time, E_values)

    # --- 3. Tile for multi-cycle convergence ---
    total_steps = steps_per_cycle * n_cycles
    aortic_pressure_all = np.tile(aortic_pressure_cycle, n_cycles)
    E_all = np.tile(E_cycle, n_cycles)

    # --- 4. Allocate arrays ---
    ventricular_volume = np.zeros(total_steps)
    ventricular_pressure = np.zeros(total_steps)
    atrial_volume = np.zeros(total_steps)
    atrial_pressure = np.zeros(total_steps)
    aortic_flow = np.zeros(total_steps)
    atrial_flow = np.zeros(total_steps)
    venous_flow = np.full(total_steps, CARDIAC_OUTPUT)

    # Initial conditions
    ventricular_volume[0] = 0.0
    atrial_volume[0] = -60.0
    ventricular_pressure[0] = E_all[0] * ventricular_volume[0]
    atrial_pressure[0] = ATRIAL_ELASTANCE * atrial_volume[0]

    # --- 5. Solve heart ODE ---
    alpha = 1.0
    for n in range(total_steps - 1):
        current_E = E_all[n]
        next_E = E_all[n + 1]

        # Aortic valve check
        if ventricular_pressure[n] >= aortic_pressure_all[n] or aortic_flow[n] > 0:
            pressure_to_flow = -(current_E * alpha * del_t / 2.0
                                 + VENTRICULAR_INDUCTANCE / del_t / alpha
                                 + VENTRICULAR_RESISTANCE)
            pressure_offset = (ventricular_pressure[n]
                               + alpha * del_t * current_E / 2.0 * (-aortic_flow[n])
                               + VENTRICULAR_INDUCTANCE / alpha / del_t * aortic_flow[n])
            flow_to_pressure = 1.0 / pressure_to_flow
            flow_offset = -pressure_offset / pressure_to_flow
            aortic_flow[n + 1] = flow_to_pressure * aortic_pressure_all[n + 1] + flow_offset
        else:
            aortic_flow[n + 1] = 0.0

        # Mitral valve check
        if atrial_pressure[n] >= ventricular_pressure[n] or atrial_flow[n] > 0:
            flow_diff = (ATRIAL_ELASTANCE * (atrial_volume[n] + alpha * del_t * (venous_flow[n] - atrial_flow[n]))
                         - current_E * (ventricular_volume[n] + alpha * del_t * atrial_flow[n])
                         - ATRIAL_RESISTANCE * atrial_flow[n])
            atrial_flow[n + 1] = atrial_flow[n] + del_t / ATRIAL_INDUCTANCE * flow_diff
            if atrial_flow[n + 1] < 0:
                atrial_flow[n + 1] = 0.0
        else:
            atrial_flow[n + 1] = 0.0

        # Update volumes (trapezoidal rule)
        ventricular_volume[n + 1] = (ventricular_volume[n]
                                     + del_t / 2.0 * (atrial_flow[n] + atrial_flow[n + 1]
                                                       - aortic_flow[n] - aortic_flow[n + 1]))
        atrial_volume[n + 1] = (atrial_volume[n]
                                + del_t / 2.0 * (venous_flow[n] + venous_flow[n + 1]
                                                  - atrial_flow[n] - atrial_flow[n + 1]))

        # Update pressures
        ventricular_pressure[n + 1] = next_E * ventricular_volume[n + 1]
        atrial_pressure[n + 1] = ATRIAL_ELASTANCE * atrial_volume[n + 1]

    # --- 6. Extract last cycle and scale ---
    start_idx = total_steps - steps_per_cycle
    P_vent_last = ventricular_pressure[start_idx:total_steps]
    P_myo = P_vent_last * myo_scale_factor

    vol_last = ventricular_volume[start_idx:total_steps]
    print(f"Volume drift: {vol_last[-1] - vol_last[0]:.4f} mL")



    # ---7. save the myocardial pressure .dat file --
    with open(P_myo_path, "w") as f:
        for i in range(len(time_cycle)):
            f.write(f"{time_cycle[i]:.6f} {P_myo[i]:.6f}\n")


    # ---8. if plot is True, plot the myocaridal pressure + Elastance + inlet pressure.
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_cycle, aortic_pressure_cycle / MMHG_TO_CGS, label="P_in (aortic)")
        ax.plot(time_cycle, P_myo / MMHG_TO_CGS, label="P_myo (ventricular)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (mmHg)")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(time_cycle, E_cycle / MMHG_TO_CGS, color="gray", linestyle="--", alpha=0.7, label="E(t)")
        ax2.set_ylabel("Elastance (mmHg/mL)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)

        ax.set_title("Aortic vs Ventricular Pressure — One Cardiac Cycle")
        fig.tight_layout()
    
        plot_path = str(P_myo_path.with_suffix(".png"))
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved → {plot_path}")

    return P_myo_path