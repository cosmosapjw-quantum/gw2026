from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import math

from ..core.ode_rk2 import integrate_fixed_grid
from ..core.shooting import ShootingProblem, solve_shooting


@dataclass
class GRMetricParams:
    """Parameters for GR structure in metric form (PDF Eq.60-64)."""
    K: float
    n: float
    rhoc: float


@dataclass
class GRMetricSolution:
    """Solved GR background star (metric form)."""
    K: float
    n: float
    rhoc: float

    Rs: float
    Lambda_s: float
    Phi_c: float
    M: float

    rhat: np.ndarray           # (1001,)
    Lambda: np.ndarray         # (1001,)
    Phi: np.ndarray            # (1001,)
    h: np.ndarray              # (1001,)

    rho: np.ndarray            # (1001,)
    P: np.ndarray              # (1001,)

    residual_inf: float
    newton_info: dict


def h_from_rho(rho: np.ndarray, K: float, n: float) -> np.ndarray:
    """Specific enthalpy h0 = 1 + K(n+1) rho^{1/n} (GR polytrope)."""
    rho = np.asarray(rho, dtype=float)
    rho = np.maximum(rho, 0.0)
    return 1.0 + float(K) * (float(n) + 1.0) * np.power(rho, 1.0 / float(n))


def rho_from_h(h: np.ndarray, K: float, n: float) -> np.ndarray:
    """Invert h -> rho for h>=1. For h<=1, rho=0 (vacuum)."""
    h = np.asarray(h, dtype=float)
    x = (h - 1.0) / (float(K) * (float(n) + 1.0))
    x = np.maximum(x, 0.0)
    return np.power(x, float(n))


def P_from_rho(rho: np.ndarray, K: float, n: float) -> np.ndarray:
    """Pressure P = K rho^{1+1/n}."""
    rho = np.asarray(rho, dtype=float)
    rho = np.maximum(rho, 0.0)
    return float(K) * np.power(rho, 1.0 + 1.0 / float(n))


def P_from_h(h: np.ndarray, K: float, n: float) -> np.ndarray:
    rho = rho_from_h(h, K, n)
    return P_from_rho(rho, K, n)


def mass_from_Lambda_s(Rs: float, Lambda_s: float) -> float:
    """Vacuum match: e^{-2Λs} = 1 - 2M/Rs  ->  M = (Rs/2)(1 - e^{-2Λs})."""
    Rs = float(Rs)
    Ls = float(Lambda_s)
    return 0.5 * Rs * (1.0 - float(np.exp(-2.0 * Ls)))


class GRStructureMetric(ShootingProblem):
    """GR structure in metric form (PDF Eq.60-64) with unknowns (Rs, Λs, Φc).

    State vector:
      y = [Lambda, Phi, h]
    """

    def build_inner_y0(self, unknown, params: GRMetricParams) -> np.ndarray:
        Rs, Lambda_s, Phi_c = float(unknown[0]), float(unknown[1]), float(unknown[2])
        h_c = float(h_from_rho(np.array([params.rhoc]), params.K, params.n)[0])
        return np.array([0.0, Phi_c, h_c], dtype=float)

    def build_outer_y0(self, unknown, params: GRMetricParams) -> np.ndarray:
        Rs, Lambda_s, Phi_c = float(unknown[0]), float(unknown[1]), float(unknown[2])
        # surface BC: Lambda(1)=Lambda_s, Phi(1)=-Lambda_s, h(1)=1
        return np.array([Lambda_s, -Lambda_s, 1.0], dtype=float)

    def rhs_inner(self, rhat: float, y: np.ndarray, unknown, params: GRMetricParams) -> np.ndarray:
        return _rhs_gr_metric(rhat, y, unknown, params)

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown, params: GRMetricParams) -> np.ndarray:
        return _rhs_gr_metric(rhat, y, unknown, params)

    def match_residual(self, y_inner_match: np.ndarray, y_outer_match: np.ndarray, unknown, params: GRMetricParams) -> np.ndarray:
        # continuity at rhat=0.5 for (Lambda, Phi, h)
        return np.asarray(y_inner_match - y_outer_match, dtype=float)

    @staticmethod
    def center_hook_inner(fun, r0: float, y0: np.ndarray, dr: float, args: tuple):
        """Analytic rhat=0 hook: removable 1/r terms.

        At r=0 (rhat=0), regularity implies:
          Lambda(0)=0, dLambda/dr = 0
          dPhi/dr = 0
          dh/dr = 0
        so dy/drhat at 0 is exactly 0.

        This is enough for Heun:
          k1=0, y_pred=y0, then k2 is evaluated at rhat=dr (finite).
        """
        k1 = np.zeros_like(np.asarray(y0, dtype=float))
        y_pred = np.asarray(y0, dtype=float) + float(dr) * k1
        return k1, y_pred


def _rhs_gr_metric(rhat: float, y: np.ndarray, unknown, params: GRMetricParams) -> np.ndarray:
    """RHS for Eq.(60-62) in rhat coordinate (fixed grid).

    The PDF gives equations in physical r:
        dΛ/dr, dΦ/dr, dh/dr.
    With r = Rs * rhat, we integrate w.r.t rhat:
        d/drhat = Rs * d/dr

    Important:
      - rhat=0 must be handled by center_hook (no epsilon).
      - use expm1 for stability in 1 - exp(2Λ).
    """
    Rs = float(unknown[0])
    # Lambda_s, Phi_c are in unknown but not needed directly in RHS
    Lam = float(y[0])

    if not np.isfinite(Lam) or abs(Lam) > 50.0:
        raise FloatingPointError(f"Lambda blew up: {Lam}")
    Phi = float(y[1])
    h = float(y[2])

    # EOS from h
    rho = float(rho_from_h(np.array([h]), params.K, params.n)[0])
    P = float(P_from_rho(np.array([rho]), params.K, params.n)[0])

    # physical radius
    r = Rs * float(rhat)

    # exp(2Λ) and (1-exp(2Λ)) in stable form
    # one_minus_e2L = 1 - exp(2Λ) = -expm1(2Λ)
    e2L = float(np.exp(2.0 * Lam))
    one_minus_e2L = -float(np.expm1(2.0 * Lam))

    if r == 0.0:
        # should be avoided by center_hook, but keep a safe branch
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # Eq.(60-62) in physical r
    term_geom = 0.5 * one_minus_e2L / r
    dLam_dr = term_geom + 4.0 * np.pi * r * (rho + float(params.n) * P) * e2L
    dPhi_dr = -term_geom + 4.0 * np.pi * r * P * e2L
    dh_dr = -h * dPhi_dr

    # convert to rhat derivatives
    dLam_drhat = Rs * dLam_dr
    dPhi_drhat = Rs * dPhi_dr
    dh_drhat = Rs * dh_dr

    return np.array([dLam_drhat, dPhi_drhat, dh_drhat], dtype=float)


def _default_guess(K: float, n: float, rhoc: float) -> np.ndarray:
    """A scale-aware but lightweight initial guess for (Rs, Lambda_s, Phi_c)."""
    K = float(K); n = float(n); rhoc = float(rhoc)

    # crude radius scale with Newtonian polytrope scaling:
    #   Rs ~ a * xi1,  a^2 ~ (n+1)K/(4π) * rhoc^{1/n - 1}
    # We use xi1≈π as a lightweight universal guess.
    expo = 0.5 * (1.0 / n - 1.0)
    Rs0 = float(np.sqrt((n + 1.0) * K * np.pi / 4.0) * (rhoc ** expo))
    Rs0 = max(Rs0, 1e-3)

    # crude "mass" scale from effective density ~ rho + nP
    Pc = float(P_from_rho(np.array([rhoc]), K, n)[0])
    rho_eff = rhoc + n * Pc
    M0 = float((4.0 * np.pi / 3.0) * rho_eff * (Rs0 ** 3))
    M0 = max(M0, 1e-12)

    # keep compactness moderate for the initial guess
    C = 2.0 * M0 / Rs0
    if C > 0.3:
        M0 *= 0.3 / C
        C = 2.0 * M0 / Rs0

    # Lambda_s via vacuum match
    Lambda_s0 = -0.5 * math.log(max(1.0 - C, 1e-12))

    # Phi_c: more negative than Phi_s=-Lambda_s
    Phi_c0 = -1.5 * Lambda_s0

    return np.array([Rs0, Lambda_s0, Phi_c0], dtype=float)


def solve_gr_structure_problem4(
    *,
    K: float,
    n: float,
    rhoc: float,
    unknown0: Optional[np.ndarray] = None,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
    coarse: bool = True,
) -> GRMetricSolution:
    """Solve GR structure (Problem 4) for one (K,n,rhoc).

    Returns a stitched profile on the full 1001-point grid.
    """
    params = GRMetricParams(K=float(K), n=float(n), rhoc=float(rhoc))
    problem = GRStructureMetric()

    if unknown0 is None:
        unknown0 = _default_guess(K, n, rhoc)

    # projection to keep parameters physical
    def project(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != 3:
            return x
        Rs, Ls, Phic = float(x[0]), float(x[1]), float(x[2])
        Rs = max(Rs, 1e-6)
        Ls = max(Ls, 0.0)
        # keep Phi_c in a reasonable range (avoid overflow in exp(Phi))
        Phic = float(np.clip(Phic, -80.0, 80.0))
        return np.array([Rs, Ls, Phic], dtype=float)

    # two-stage option helps robustness: coarse then fine
    if coarse:
        root = solve_shooting(
            problem,
            project(unknown0),
            params,
            tol=tol,
            maxiter=maxiter,
            dx_rel=dx_rel,
            project=project,
            coarse_steps=(200, 200),
            fine_steps=(500, 500),
            coarse_tol=1e-8,
        )
    else:
        root = solve_shooting(
            problem,
            project(unknown0),
            params,
            tol=tol,
            maxiter=maxiter,
            dx_rel=dx_rel,
            project=project,
            n_steps_inner=500,
            n_steps_outer=500,
        )

    root = project(np.asarray(root, dtype=float))
    Rs, Lambda_s, Phi_c = float(root[0]), float(root[1]), float(root[2])
    M = mass_from_Lambda_s(Rs, Lambda_s)

    # Build full profile by stitching inner (0->0.5) and outer (1->0.5)
    r_inner = np.linspace(0.0, 0.5, 501, dtype=float)
    r_outer = np.linspace(1.0, 0.5, 501, dtype=float)

    y0_in = problem.build_inner_y0(root, params)
    y0_out = problem.build_outer_y0(root, params)

    def fun(rhat: float, y: np.ndarray, unknown_: np.ndarray, params_: GRMetricParams) -> np.ndarray:
        return _rhs_gr_metric(float(rhat), y, unknown_, params_)

    y_in = integrate_fixed_grid(fun, r_inner, y0_in, args=(root, params), center_hook=problem.center_hook_inner)
    y_out = integrate_fixed_grid(fun, r_outer, y0_out, args=(root, params), center_hook=None)

    # stitch
    rhat_full = np.linspace(0.0, 1.0, 1001, dtype=float)
    y_full = np.empty((1001, 3), dtype=float)
    y_full[:501, :] = y_in  # includes match point
    y_full[500:, :] = y_out[::-1, :]  # reversed so rhat increases; overlap at index 500

    Lambda = y_full[:, 0].copy()
    Phi = y_full[:, 1].copy()
    h = y_full[:, 2].copy()

    rho = rho_from_h(h, K, n)
    P = P_from_h(h, K, n)

    # residual at match
    y_in_m = y_in[-1]
    y_out_m = y_out[-1]
    R = problem.match_residual(y_in_m, y_out_m, root, params).reshape(-1)
    residual_inf = float(np.max(np.abs(R)))

    info = getattr(solve_shooting, "last_info", {}).copy()

    return GRMetricSolution(
        K=float(K),
        n=float(n),
        rhoc=float(rhoc),
        Rs=Rs,
        Lambda_s=Lambda_s,
        Phi_c=Phi_c,
        M=M,
        rhat=rhat_full,
        Lambda=Lambda,
        Phi=Phi,
        h=h,
        rho=rho,
        P=P,
        residual_inf=residual_inf,
        newton_info=info,
    )


def write_problem4_dat(filename: str, sol: GRMetricSolution) -> None:
    """Write output in the Problem 4 format (problem4.dat).

    Format:
      - first line: n, M, Rs, Lambda_s, Phi_c
      - then columns: rhat, Lambda, Phi, h
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{sol.n:.16e} {sol.M:.16e} {sol.Rs:.16e} {sol.Lambda_s:.16e} {sol.Phi_c:.16e}\n")
        for i in range(sol.rhat.size):
            f.write(f"{sol.rhat[i]:.16e} {sol.Lambda[i]:.16e} {sol.Phi[i]:.16e} {sol.h[i]:.16e}\n")