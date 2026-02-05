from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import math
import numpy as np

from ..core.ode_rk2 import integrate_fixed_grid
from ..core.shooting import ShootingProblem, solve_shooting


__all__ = [
    "NewtonStructureSolution",
    "NewtonianStructureProblem1",
    "enthalpy_from_rho_newton",
    "rho_from_enthalpy_newton",
    "solve_newton_structure_problem1",
    "write_problem1_dat",
]


def enthalpy_from_rho_newton(rho: float, K: float, n: float) -> float:
    """Newtonian polytrope specific enthalpy (PDF eq. 9):
        h = K (n+1) rho^(1/n)
    """
    rho = float(rho)
    if rho < 0.0:
        raise ValueError("rho must be non-negative.")
    K = float(K)
    n = float(n)
    if K <= 0.0 or n <= 0.0:
        raise ValueError("K and n must be positive.")
    return K * (n + 1.0) * (rho ** (1.0 / n))


def rho_from_enthalpy_newton(h: np.ndarray, K: float, n: float) -> np.ndarray:
    """Invert h = K(n+1) rho^(1/n) -> rho = (h/(K(n+1)))^n, with h clipped at 0."""
    K = float(K)
    n = float(n)
    denom = K * (n + 1.0)
    if denom <= 0.0:
        raise ValueError("K*(n+1) must be positive.")
    h_arr = np.asarray(h, dtype=float)
    h_pos = np.maximum(h_arr, 0.0)
    return (h_pos / denom) ** n


@dataclass
class NewtonStructureSolution:
    # inputs
    K: float
    n: float
    rhoc: float

    # solved
    Rs: float
    M: float

    # profiles (fixed grid: 1001 pts)
    rhat: np.ndarray
    h: np.ndarray
    u: np.ndarray
    rho: np.ndarray

    # diagnostics
    residual_inf: float
    newton_info: dict


class NewtonianStructureProblem1(ShootingProblem):
    """Problem 1 Newtonian structure shooting problem.

    Unknowns: (Rs, M)
    State: y = [h, u] where u = dPhi/drhat.

    ODE (PDF eq. 14 with u variable):
        dh/drhat = -u
        du/drhat = -(2/rhat) u + 4*pi*Rs^2 * (h/(K(n+1)))^n

    Center (rhat=0):
        u(0)=0
        du/drhat(0) = (4*pi/3)*rhoc*Rs^2   (analytic limit)
        h(0)=K(n+1)*rhoc^(1/n)

    Surface (rhat=1):
        h(1)=0
        u(1)=M/Rs
    """
    def __init__(self, *, K: float, n: float, rhoc: float):
        self.K = float(K)
        self.n = float(n)
        self.rhoc = float(rhoc)

        # attach hook as attribute (used by solve_shooting)
        self.center_hook_inner = self._center_hook_inner

    # --- boundary builders ---
    def build_inner_y0(self, unknown: np.ndarray, params: Any) -> np.ndarray:
        h_c = enthalpy_from_rho_newton(self.rhoc, self.K, self.n)
        return np.array([h_c, 0.0], dtype=float)

    def build_outer_y0(self, unknown: np.ndarray, params: Any) -> np.ndarray:
        Rs, M = float(unknown[0]), float(unknown[1])
        return np.array([0.0, M / Rs], dtype=float)

    # --- RHS ---
    def rhs_inner(self, rhat: float, y: np.ndarray, unknown: np.ndarray, params: Any) -> np.ndarray:
        return self._rhs(float(rhat), np.asarray(y, dtype=float), unknown)

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown: np.ndarray, params: Any) -> np.ndarray:
        return self._rhs(float(rhat), np.asarray(y, dtype=float), unknown)

    def _rhs(self, rhat: float, y: np.ndarray, unknown: np.ndarray) -> np.ndarray:
        Rs = float(unknown[0])
        h, u = float(y[0]), float(y[1])

        # density (clip h >= 0 to avoid complex powers for fractional n)
        rho = float(rho_from_enthalpy_newton(np.array([h]), self.K, self.n)[0])

        dh = -u
        du = -(2.0 / rhat) * u + 4.0 * np.pi * (Rs ** 2) * rho
        return np.array([dh, du], dtype=float)

    # --- matching residual at r_match=0.5 ---
    def match_residual(
        self,
        y_inner_match: np.ndarray,
        y_outer_match: np.ndarray,
        unknown: np.ndarray,
        params: Any,
    ) -> np.ndarray:
        yi = np.asarray(y_inner_match, dtype=float).reshape(-1)
        yo = np.asarray(y_outer_match, dtype=float).reshape(-1)
        if yi.size != 2 or yo.size != 2:
            raise ValueError("Expected state dimension 2: [h,u].")
        return np.array([yi[0] - yo[0], yi[1] - yo[1]], dtype=float)

    # --- analytic rhat=0 hook (NO epsilon) ---
    def _center_hook_inner(
        self,
        fun: Callable,
        r0: float,
        y0: np.ndarray,
        dr: float,
        args: tuple,
    ):
        # args = (unknown, params)
        unknown = np.asarray(args[0], dtype=float).reshape(-1)
        Rs = float(unknown[0])

        # At rhat=0:
        #   u(0)=0, dh/dr=0
        #   du/dr = d^2Phi/dr^2 = (4*pi/3) rhoc Rs^2   (PDF eq. 15)
        k1_h = 0.0
        k1_u = (4.0 * np.pi / 3.0) * self.rhoc * (Rs ** 2)
        k1 = np.array([k1_h, k1_u], dtype=float)

        y0 = np.asarray(y0, dtype=float).reshape(-1)
        y_pred = y0 + float(dr) * k1
        return k1, y_pred


def _project_positive_RsM(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 2:
        return x
    x[0] = max(float(x[0]), 1e-12)  # Rs
    x[1] = max(float(x[1]), 1e-12)  # M
    return x


def default_guess_RsM(K: float, n: float, rhoc: float) -> Tuple[float, float]:
    """Heuristic guess for (Rs, M).

    Calibrated to the example scale (K=100, n=1, rhoc=1.28e-3) where Rs~O(10), M~O(1).
    """
    K = float(K)
    n = float(n)
    rhoc = float(rhoc)

    # Lane-Emden-inspired rough scaling:
    #   R ~ sqrt(K) * rho_c^{(1-n)/(2n)}  (up to an O(1) factor)
    expo = (1.0 - n) / (2.0 * n)
    Rs0 = 9.0 * math.sqrt(max(K, 1e-30) / 100.0) * (max(rhoc, 1e-30) / 1.28e-3) ** expo
    Rs0 = float(np.clip(Rs0, 0.5, 200.0))

    # Uniform-sphere-ish mass scale with a soft factor
    M0 = 0.5 * (4.0 * np.pi / 3.0) * rhoc * (Rs0 ** 3)
    M0 = float(np.clip(M0, 1e-8, 200.0))
    return Rs0, M0


def solve_newton_structure_problem1(
    *,
    K: float,
    n: float,
    rhoc: float,
    unknown0: Optional[Sequence[float]] = None,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
    # integration settings (fixed by PDF, but kept explicit)
    n_steps_inner: int = 500,
    n_steps_outer: int = 500,
    r_match: float = 0.5,
    # robustness
    use_two_stage: bool = True,
    dx_rel_coarse: float = 1e-5,
    multi_start_fallback: bool = True,
) -> NewtonStructureSolution:
    """Solve Newtonian structure (Problem 1) via 2D shooting (Rs, M)."""
    K = float(K)
    n = float(n)
    rhoc = float(rhoc)

    if unknown0 is None:
        Rs0, M0 = default_guess_RsM(K, n, rhoc)
    else:
        Rs0, M0 = float(unknown0[0]), float(unknown0[1])

    problem = NewtonianStructureProblem1(K=K, n=n, rhoc=rhoc)

    # Two-stage: coarse -> fine (same algorithm, fewer steps / looser tol).
    coarse_steps = (max(80, n_steps_inner // 5), max(80, n_steps_outer // 5)) if use_two_stage else None
    fine_steps = (n_steps_inner, n_steps_outer) if use_two_stage else None

    # initial attempt
    root = None
    last_err: Optional[Exception] = None

    try:
        root = solve_shooting(
            problem,
            np.array([Rs0, M0], dtype=float),
            params=None,
            r_match=float(r_match),
            n_steps_inner=int(n_steps_inner),
            n_steps_outer=int(n_steps_outer),
            tol=float(tol),
            maxiter=int(maxiter),
            dx_rel=float(dx_rel),
            project=_project_positive_RsM,
            coarse_steps=coarse_steps,
            fine_steps=fine_steps,
            coarse_tol=1e-8,
        )
    except Exception as e:
        last_err = e

    # fallback multi-start (small grid around guess) if needed
    if root is None and multi_start_fallback:
        seeds: list[np.ndarray] = []
        sRs = np.array([0.6, 0.85, 1.0, 1.15, 1.4], dtype=float)
        sM = np.array([0.6, 0.85, 1.0, 1.15, 1.4], dtype=float)
        for a in sRs:
            for b in sM:
                seeds.append(np.array([Rs0 * a, M0 * b], dtype=float))

        try:
            roots = solve_shooting(
                problem,
                np.array([Rs0, M0], dtype=float),
                params=None,
                r_match=float(r_match),
                n_steps_inner=int(n_steps_inner),
                n_steps_outer=int(n_steps_outer),
                tol=float(tol),
                maxiter=int(maxiter),
                dx_rel=float(dx_rel),
                project=_project_positive_RsM,
                coarse_steps=coarse_steps,
                fine_steps=fine_steps,
                coarse_tol=1e-8,
                multi_start=seeds,
                return_all=True,
                cluster_tol=1e-6,
            )
            if roots:
                # pick the one with smallest residual_inf (re-evaluate)
                best = None
                best_res = float("inf")
                for r in roots:
                    rvec = np.asarray(r, dtype=float).reshape(-1)
                    yi, yo = _integrate_match(problem, rvec, r_match=float(r_match),
                                             n_steps_inner=int(n_steps_inner), n_steps_outer=int(n_steps_outer))
                    R = problem.match_residual(yi, yo, rvec, None)
                    res = float(np.max(np.abs(np.asarray(R, float))))
                    if res < best_res:
                        best_res = res
                        best = rvec
                root = best
        except Exception as e:
            last_err = e

    if root is None:
        msg = "solve_newton_structure_problem1: shooting failed."
        if last_err is not None:
            msg += f" Last error: {type(last_err).__name__}: {last_err}"
        raise RuntimeError(msg)

    root = np.asarray(root, dtype=float).reshape(-1)
    Rs, M = float(root[0]), float(root[1])

    # Full profile on the PDF grid: 1001 points in [0,1]
    rhat = np.linspace(0.0, 1.0, 1001, dtype=float)

    h_c = enthalpy_from_rho_newton(rhoc, K, n)
    y0 = np.array([h_c, 0.0], dtype=float)

    def rhs_full(r: float, y: np.ndarray, unknown: np.ndarray, params: Any) -> np.ndarray:
        return problem._rhs(float(r), np.asarray(y, dtype=float), unknown)

    y_grid = integrate_fixed_grid(rhs_full, rhat, y0, args=(root, None), center_hook=problem.center_hook_inner)
    h = np.asarray(y_grid[:, 0], dtype=float)
    u = np.asarray(y_grid[:, 1], dtype=float)
    rho = rho_from_enthalpy_newton(h, K, n)

    # residual at match (for diagnostics)
    yi, yo = _integrate_match(problem, root, r_match=float(r_match),
                             n_steps_inner=int(n_steps_inner), n_steps_outer=int(n_steps_outer))
    Rm = np.asarray(problem.match_residual(yi, yo, root, None), dtype=float).reshape(-1)
    residual_inf = float(np.max(np.abs(Rm))) if Rm.size else 0.0

    info = getattr(solve_shooting, "last_info", {}).copy()

    return NewtonStructureSolution(
        K=K, n=n, rhoc=rhoc,
        Rs=Rs, M=M,
        rhat=rhat, h=h, u=u, rho=rho,
        residual_inf=residual_inf,
        newton_info=info,
    )


def _integrate_match(
    problem: NewtonianStructureProblem1,
    unknown: np.ndarray,
    *,
    r_match: float,
    n_steps_inner: int,
    n_steps_outer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper: integrate both sides and return y_inner_match, y_outer_match."""
    r_inner = np.linspace(0.0, float(r_match), int(n_steps_inner) + 1, dtype=float)
    r_outer = np.linspace(1.0, float(r_match), int(n_steps_outer) + 1, dtype=float)

    y0_in = problem.build_inner_y0(unknown, None)
    y0_out = problem.build_outer_y0(unknown, None)

    def f_in(r: float, y: np.ndarray, unk: np.ndarray, params: Any) -> np.ndarray:
        return problem.rhs_inner(float(r), y, unk, params)

    def f_out(r: float, y: np.ndarray, unk: np.ndarray, params: Any) -> np.ndarray:
        return problem.rhs_outer(float(r), y, unk, params)

    y_in = integrate_fixed_grid(f_in, r_inner, y0_in, args=(unknown, None), center_hook=problem.center_hook_inner)
    y_out = integrate_fixed_grid(f_out, r_outer, y0_out, args=(unknown, None), center_hook=None)

    return np.asarray(y_in[-1], float), np.asarray(y_out[-1], float)


def write_problem1_dat(
    filename: str,
    sol: NewtonStructureSolution,
) -> None:
    """Write Problem 1 output file format:
      - first row: n, M, Rs
      - subsequent rows: rhat, h
    Uses comma+space separators to match the baseline repo style.
    """
    lines = []
    lines.append(f"{sol.n}, {sol.M}, {sol.Rs}\n")
    for r, h in zip(sol.rhat, sol.h):
        lines.append(f"{float(r)}, {float(h)}\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
