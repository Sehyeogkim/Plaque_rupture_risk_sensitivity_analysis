#!/usr/bin/env python3
"""
Generate pressure inlet waveform P(t) from Q(t) using 3-element Windkessel (RCR)
with Rp/Rd = 1/10 ratio constraint (SimVascular coronary reference).

Self-contained: no external dependencies beyond numpy, scipy, matplotlib.

Algorithm:
    1. Given Q(t), SBP, DBP, tau → fit Pd and Rd via least_squares
    2. Rp = Rd / 10  (physiological constraint)
    3. C  = tau / Rd
    4. Solve ODE via Forward Euler for N cycles → extract converged last-cycle P(t)
    5. Scale time by HR (60/HR)

References:
    - Westerhof et al. (2009), Med Biol Eng Comput, 47:131-141
    - SimVascular coronary BC: Rp=140, Rd=1415 → Rp/Rd ≈ 1/10
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Default Rp/Rd ratio: SimVascular coronary reference (Rp=140, Rd=1415)
DEFAULT_RP_RD_RATIO = 1.0 / 10.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_inflow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load inflow waveform Q(t) from a two-column text file (time, flow)."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path} must contain exactly two columns: time and flow.")

    time = data[:, 0]
    flow = data[:, 1]
    dt = np.diff(time)

    if len(time) < 3:
        raise ValueError("Need at least three inflow samples.")
    if not np.all(dt > 0.0):
        raise ValueError("Time must be strictly increasing.")
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-10):
        raise ValueError("This implementation assumes uniform time spacing.")

    return time, flow


# ---------------------------------------------------------------------------
# Forward Euler ODE solver
# ---------------------------------------------------------------------------

def _explicit_euler_cycle(
    flow_cycle: np.ndarray,
    dt: float,
    pd: float,
    rp: float,
    rd: float,
    tau: float,
    pc0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one cardiac cycle of the RCR ODE via explicit Euler.

    ODE:  dPc/dt = -(1/tau)*(Pc - Pd) + Q/C
    Output:  P(t) = Rp*Q(t) + Pc(t)
    """
    c = tau / rd
    a = 1.0 / tau

    pc = np.empty_like(flow_cycle)
    pressure = np.empty_like(flow_cycle)

    pc[0] = pc0
    pressure[0] = rp * flow_cycle[0] + pc[0]

    for idx in range(len(flow_cycle) - 1):
        rhs = -a * (pc[idx] - pd) + flow_cycle[idx] / c
        pc[idx + 1] = pc[idx] + dt * rhs
        pressure[idx + 1] = rp * flow_cycle[idx + 1] + pc[idx + 1]

    return pc, pressure


def _solve_last_cycle(
    time: np.ndarray,
    flow: np.ndarray,
    pd: float,
    rp: float,
    rd: float,
    tau: float,
    dbp: float,
    n_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat Q(t) for n_cycles and return the converged last-cycle P(t)."""
    if n_cycles < 1:
        raise ValueError("n_cycles must be at least 1.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if rd <= 0.0:
        raise ValueError("Rd must be positive.")

    dt = float(time[1] - time[0])
    pc0 = dbp - rp * flow[0]
    pc = None
    pressure = None

    for _ in range(n_cycles):
        pc, pressure = _explicit_euler_cycle(flow, dt, pd, rp, rd, tau, pc0)
        pc0 = float(pc[-1])

    return pc, pressure


# ---------------------------------------------------------------------------
# Ratio-constrained fitting (Rp/Rd = 1/10)
# ---------------------------------------------------------------------------

def _residual(
    x: np.ndarray,
    time: np.ndarray,
    flow: np.ndarray,
    sbp: float,
    dbp: float,
    tau: float,
    n_cycles: int,
    rp_rd_ratio: float,
) -> np.ndarray:
    """Residual vector: [max(P)-SBP, min(P)-DBP], normalized.

    Decision variables:  x = [Pd, ln(Rd)]
    Derived:             Rp = rp_rd_ratio * Rd,  C = tau / Rd
    """
    pd = x[0]
    rd = math.exp(x[1])
    rp = rp_rd_ratio * rd

    _, pressure = _solve_last_cycle(
        time=time, flow=flow, pd=pd, rp=rp, rd=rd,
        tau=tau, dbp=dbp, n_cycles=n_cycles,
    )

    max_p = float(np.max(pressure))
    min_p = float(np.min(pressure))

    return np.array([
        (max_p - sbp) / max(abs(sbp), 1.0),
        (min_p - dbp) / max(abs(dbp), 1.0),
    ], dtype=float)


def _fit_rcr(
    time: np.ndarray,
    flow: np.ndarray,
    sbp: float,
    dbp: float,
    tau: float,
    n_cycles: int = 20,
    rp_rd_ratio: float = DEFAULT_RP_RD_RATIO,
    max_nfev: int = 4000,
) -> dict:
    """Fit RCR model with Rp = rp_rd_ratio * Rd constraint.

    Returns dict with keys: pd, rp, rd, c, tau, time, pressure.
    """
    if sbp <= dbp:
        raise ValueError("SBP must be greater than DBP.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    # Initial guess from mean-pressure / mean-flow
    q_mean = max(float(np.mean(flow)), 1e-6)
    mean_p = (sbp + dbp) / 2.0
    r_total = mean_p / q_mean
    rd0 = r_total / (1.0 + rp_rd_ratio)
    pd0 = dbp

    x0 = np.array([pd0, math.log(rd0)], dtype=float)
    lb = np.array([-1e6, math.log(1e-8)], dtype=float)
    ub = np.array([1e6, math.log(1e8)], dtype=float)

    result = least_squares(
        _residual, x0=x0, bounds=(lb, ub),
        args=(time, flow, sbp, dbp, tau, n_cycles, rp_rd_ratio),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev,
    )
    if not result.success:
        raise RuntimeError(f"RCR fit failed: {result.message}")

    pd = float(result.x[0])
    rd = math.exp(result.x[1])
    rp = rp_rd_ratio * rd
    c = tau / rd

    _, pressure = _solve_last_cycle(
        time=time, flow=flow, pd=pd, rp=rp, rd=rd,
        tau=tau, dbp=dbp, n_cycles=n_cycles,
    )

    return dict(pd=pd, rp=rp, rd=rd, c=c, tau=tau, time=time, pressure=pressure)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pressure_windkessel(
    SBP: float,
    DBP: float,
    tau: float,
    inflow_path: str | Path = "inflow.dat",
    HR: float = 60.0,
    rp_rd_ratio: float = DEFAULT_RP_RD_RATIO,
    n_cycles: int = 50,
    save_path: str | Path | None = None,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate aortic/coronary pressure waveform P(t) from Q(t) inflow using
    a 3-element Windkessel (RCR) model with Rp/Rd ratio constraint.

    Parameters
    ----------
    SBP : float
        Systolic blood pressure [mmHg].
    DBP : float
        Diastolic blood pressure [mmHg].
    tau : float
        Diastolic time constant tau = Rd * C [s].
    inflow_path : str or Path
        Path to inflow waveform file (two columns: time, flow).
    HR : float
        Heart rate [bpm]. Default 60 (T_cycle = 1s).
    rp_rd_ratio : float
        Rp/Rd ratio. Default 1/10 (SimVascular coronary reference).
    n_cycles : int
        Number of cardiac cycles for convergence. Default 20.
    save_path : str, Path, or None
        If provided, save the pressure waveform to this file.
    plot : bool
        If True, save a plot next to save_path (or outputs/pressure_windkessel.png).

    Returns
    -------
    t : np.ndarray
        Time array [s], scaled by HR.
    P : np.ndarray
        Pressure array [mmHg].
    """
    # --- 1. Load inflow and fit RCR parameters ---
    time, flow = load_inflow(Path(inflow_path))

    fit = _fit_rcr(
        time, flow,
        sbp=SBP, dbp=DBP, tau=tau,
        n_cycles=n_cycles, rp_rd_ratio=rp_rd_ratio,
    )

    # --- 2. Scale time by heart rate ---
    T_cycle = 60.0 / HR
    t_scaled = fit["time"] * T_cycle

    P = fit["pressure"]

    # --- 3. Print fitted parameters ---
    print(f"--- Windkessel RCR fit (Rp/Rd = 1/{round(1/rp_rd_ratio)}) ---")
    print(f"  SBP = {SBP:.1f} mmHg,  DBP = {DBP:.1f} mmHg")
    print(f"  HR  = {HR:.0f} bpm,  T_cycle = {T_cycle:.3f} s")
    print(f"  tau = {tau:.4f} s")
    print(f"  Pd  = {fit['pd']:.4f}")
    print(f"  Rp  = {fit['rp']:.6f}")
    print(f"  Rd  = {fit['rd']:.6f}")
    print(f"  C   = {fit['c']:.6f}")
    print(f"  max(P) = {np.max(P):.2f},  min(P) = {np.min(P):.2f}")

    # --- 4. Save ---
    mmHG_to_dyne_cm2 = 1333.22
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for t_val, p_val in zip(t_scaled, P):
                f.write(f"{t_val:.6f} {p_val * mmHG_to_dyne_cm2:.6f}\n")
        print(f"  Saved → {save_path} ({len(t_scaled)} points)")

    # --- 5. Plot ---
    if plot:
        plot_path = (
            Path(save_path).with_suffix(".png")
            if save_path
            else Path("outputs/pressure_windkessel.png")
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_scaled * 1000, P, "b-", linewidth=2, label="P(t)")
        ax.axhline(SBP, color="r", ls="--", lw=0.8, alpha=0.6, label=f"SBP ({SBP:.0f})")
        ax.axhline(DBP, color="orange", ls="--", lw=0.8, alpha=0.6, label=f"DBP ({DBP:.0f})")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pressure [mmHg]")
        ax.set_title(
            f"Windkessel P(t): SBP={SBP:.0f}, DBP={DBP:.0f}, "
            f"HR={HR:.0f}, tau={tau:.2f}s"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → {plot_path}")

    return t_scaled, P


if __name__ == "__main__":
    t, P = generate_pressure_windkessel(
        SBP=120, DBP=80, tau=0.2, HR=60,
        inflow_path="inflow.dat",
        save_path="P_in.dat",
        plot=True,
    )
