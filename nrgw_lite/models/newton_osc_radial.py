from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.ode_rk2 import integrate_fixed_grid
from ..core.shooting import ShootingProblem, solve_shooting
from .newton_structure import NewtonStructureSolution, solve_newton_structure_problem1

__all__ = [
    "NewtonBackground",
    "NewtonRadialMode",
    "build_background_from_structure",
    "RadialCowlingProblem",
    "scan_omega_seeds",
    "solve_radial_modes",
    "integrate_radial_mode",
    "count_nodes",
    "write_problem2_dat",
]


@dataclass(frozen=True)
class NewtonBackground:
    """Background Newtonian polytrope in the handout's normalization (rhat ∈ [0,1])."""
    K: float
    n: float
    rhoc: float
    Rs: float
    M: float
    rhat: np.ndarray  # (1001,)
    h0: np.ndarray  # (1001,)
    dh0_drhat: np.ndarray  # (1001,)
    rho0: np.ndarray  # (1001,)

    # surface derivatives (w.r.t rhat)
    dh_s: float
    d2h_s: float


@dataclass(frozen=True)
class NewtonRadialMode:
    """A solved radial (l=0) Cowling mode."""
    mode_index: int
    omega: float
    nodes: int
    rhat: np.ndarray
    xihat: np.ndarray
    delta_h: np.ndarray
    bg: NewtonBackground


def build_background_from_structure(sol: NewtonStructureSolution) -> NewtonBackground:
    rhat = np.asarray(sol.rhat, dtype=float)
    h0 = np.asarray(sol.h, dtype=float)
    # In Problem 1: dh/drhat = -u
    dh0 = -np.asarray(sol.u, dtype=float)
    rho0 = np.asarray(sol.rho, dtype=float)

    # Surface derivatives:
    # dh/drhat(1) = -M/Rs from the vacuum match / handout.
    dh_s = float(dh0[-1])
    # d2h/drhat2(1) = 2 u(1) = 2 M/Rs (because rho=0 at the surface).
    d2h_s = float(2.0 * sol.u[-1])

    return NewtonBackground(
        K=float(sol.K),
        n=float(sol.n),
        rhoc=float(sol.rhoc),
        Rs=float(sol.Rs),
        M=float(sol.M),
        rhat=rhat,
        h0=h0,
        dh0_drhat=dh0,
        rho0=rho0,
        dh_s=dh_s,
        d2h_s=d2h_s,
    )


def _idx_from_rhat(rhat: float, *, N: int = 1001) -> int:
    # grid is exactly 0..1 with dr=1e-3
    i = int(round(float(rhat) * (N - 1)))
    if i < 0:
        return 0
    if i >= N:
        return N - 1
    return i


class RadialCowlingProblem(ShootingProblem):
    """Problem 2: radial (l=0) Cowling oscillations on a Newtonian polytrope background.

    State: y = [xihat, delta_h] where xihat = xi / rhat and delta_h is Eulerian enthalpy perturbation.

    Unknown: omega (scalar).

    Matching at r_match uses Eq.(37)-style scalar residual:
        f = xihat_in * delta_h_out - xihat_out * delta_h_in = 0
    """

    def __init__(self, bg: NewtonBackground):
        self.bg = bg
        self.n = float(bg.n)
        self.Rs = float(bg.Rs)

        # center hook to avoid the removable 1/r singularity at rhat=0.
        self.center_hook_inner = self._center_hook
        self.center_hook_outer = None

    # ---- BC builders ----
    def build_inner_y0(self, unknown: float, params: Any) -> np.ndarray:
        # Eq.(34): xihat(0)=1, delta_h(0)=-(3 h0c)/(n Rs) * xihat(0)
        h0c = float(self.bg.h0[0])
        xihat0 = 1.0
        delta0 = -(3.0 * h0c) / (self.n * self.Rs) * xihat0
        return np.array([xihat0, delta0], dtype=float)

    def build_outer_y0(self, unknown: float, params: Any) -> np.ndarray:
        # Eq.(35): xihat(1)=1, delta_h(1)=-(1/Rs) dh0/drhat|1 * xihat(1)
        xihat1 = 1.0
        delta1 = -(1.0 / self.Rs) * float(self.bg.dh_s) * xihat1
        return np.array([xihat1, delta1], dtype=float)

    # ---- RHS ----
    def rhs_inner(self, rhat: float, y: np.ndarray, unknown: float, params: Any) -> np.ndarray:
        return self._rhs(float(rhat), np.asarray(y, dtype=float), float(unknown))

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown: float, params: Any) -> np.ndarray:
        return self._rhs(float(rhat), np.asarray(y, dtype=float), float(unknown))

    def _rhs(self, rhat: float, y: np.ndarray, omega: float) -> np.ndarray:
        xihat = float(y[0])
        deltah = float(y[1])

        # Center (should not be called if center_hook is used, but keep it safe)
        if rhat == 0.0:
            return np.array([0.0, 0.0], dtype=float)

        # Surface analytic-limit branch to avoid h0=0 division
        if rhat == 1.0:
            a = float(self.bg.dh_s)     # dh0/drhat at surface
            b = float(self.bg.d2h_s)    # d2h0/drhat2 at surface

            # Eq.(35)(iv): d(delta_h)/drhat = Rs * omega^2 * rhat * xihat
            d_delta = self.Rs * (omega * omega) * 1.0 * xihat

            # Eq.(35)(ii): L'Hospital form
            # d xihat / drhat = -(1/(n+1)) * (1/a) * [ 3*a*(xihat/rhat) + n*Rs^2*omega^2*xihat + n*b*xihat ]
            # (at rhat=1)
            denom = a
            if denom == 0.0:
                # This should not happen for a well-posed background.
                raise ZeroDivisionError("Surface dh0/drhat is zero; cannot apply surface BC for xihat'.")
            d_xi = -(xihat / (self.n + 1.0)) * (1.0 / denom) * (
                3.0 * denom / 1.0 + self.n * (self.Rs * self.Rs) * (omega * omega) + self.n * b
            )

            return np.array([d_xi, d_delta], dtype=float)

        # Interior: read background at this rhat
        i = _idx_from_rhat(rhat)
        h0 = float(self.bg.h0[i])
        dh0 = float(self.bg.dh0_drhat[i])

        if h0 <= 0.0:
            # Should only occur at the surface point, handled above
            raise ValueError(f"h0(rhat={rhat}) is non-positive inside the star: h0={h0}")

        # Eq.(29~): use a cancellation-stable grouping near the surface:
        #   xihat' = -(3/rhat) xihat - (n/h0) [ dh0*xihat + (Rs/rhat)*delta_h ]
        d_xi = -(3.0 / rhat) * xihat - (self.n / h0) * (dh0 * xihat + (self.Rs / rhat) * deltah)

        # Euler (Cowling) radial component: d(delta_h)/drhat = Rs * omega^2 * rhat * xihat
        d_delta = self.Rs * (omega * omega) * rhat * xihat

        return np.array([d_xi, d_delta], dtype=float)

    # ---- match ----
    def match_residual(self, y_inner_match: np.ndarray, y_outer_match: np.ndarray, unknown: float, params: Any) -> float:
        yi = np.asarray(y_inner_match, dtype=float).reshape(-1)
        yo = np.asarray(y_outer_match, dtype=float).reshape(-1)
        # Eq.(37): f = xihat_in * delta_out - xihat_out * delta_in
        return float(yi[0] * yo[1] - yo[0] * yi[1])

    # ---- analytic center hook ----
    @staticmethod
    def _center_hook(fun: Callable[..., Any], r0: float, y0: np.ndarray, dr: float, args: tuple):
        # Regularity at rhat=0 gives xihat'(0)=0 and delta_h'(0)=0.
        k1 = np.zeros_like(y0, dtype=float)
        y_pred = np.asarray(y0, dtype=float)
        return k1, y_pred


def _integrate_to_match(problem: RadialCowlingProblem, omega: float, *, r_match: float, n_steps_inner: int, n_steps_outer: int) -> Tuple[np.ndarray, np.ndarray]:
    r_in = np.linspace(0.0, float(r_match), int(n_steps_inner) + 1, dtype=float)
    r_out = np.linspace(1.0, float(r_match), int(n_steps_outer) + 1, dtype=float)

    y0_in = problem.build_inner_y0(omega, None)
    y0_out = problem.build_outer_y0(omega, None)

    def f_in(r: float, y: np.ndarray, w: float, params: Any) -> np.ndarray:
        return problem.rhs_inner(r, y, w, params)

    def f_out(r: float, y: np.ndarray, w: float, params: Any) -> np.ndarray:
        return problem.rhs_outer(r, y, w, params)

    y_in = integrate_fixed_grid(f_in, r_in, y0_in, args=(omega, None), center_hook=problem.center_hook_inner)
    y_out = integrate_fixed_grid(f_out, r_out, y0_out, args=(omega, None), center_hook=None)
    return np.asarray(y_in[-1], float), np.asarray(y_out[-1], float)


def scan_omega_seeds(
    problem: RadialCowlingProblem,
    omega_min: float,
    omega_max: float,
    *,
    n_scan: int = 200,
    r_match: float = 0.5,
    n_steps_inner: int = 200,
    n_steps_outer: int = 200,
    abs_f_tol: float = 1e-3,
) -> List[float]:
    """Coarse scan to find candidate omega seeds (multi-root problems).

    Strategy:
      - sample f(omega) on a coarse grid
      - collect midpoints of sign-change intervals
      - also collect local minima of |f| below abs_f_tol
    """
    wmin = float(omega_min)
    wmax = float(omega_max)
    if not (wmin > 0 and wmax > wmin):
        raise ValueError("omega_min must be >0 and omega_max > omega_min.")

    w_grid = np.linspace(wmin, wmax, int(n_scan), dtype=float)
    f_grid = np.empty_like(w_grid)

    for i, w in enumerate(w_grid):
        try:
            yi, yo = _integrate_to_match(problem, float(w), r_match=r_match, n_steps_inner=n_steps_inner, n_steps_outer=n_steps_outer)
            f_grid[i] = problem.match_residual(yi, yo, float(w), None)
        except Exception:
            f_grid[i] = np.nan

    seeds: List[float] = []

    # sign-change bracketing
    for i in range(len(w_grid) - 1):
        f0, f1 = f_grid[i], f_grid[i + 1]
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 == 0.0:
            seeds.append(float(w_grid[i]))
        if f0 * f1 < 0.0:
            seeds.append(float(0.5 * (w_grid[i] + w_grid[i + 1])))

    # local minima of |f|
    absf = np.abs(f_grid)
    for i in range(1, len(w_grid) - 1):
        if not np.isfinite(absf[i]):
            continue
        if absf[i] < abs_f_tol and absf[i] <= absf[i - 1] and absf[i] <= absf[i + 1]:
            seeds.append(float(w_grid[i]))

    # de-duplicate seeds (coarse)
    seeds_sorted = sorted(seeds)
    uniq: List[float] = []
    for s in seeds_sorted:
        if not uniq:
            uniq.append(s)
        else:
            if abs(s - uniq[-1]) > 0.25 * (wmax - wmin) / max(int(n_scan) - 1, 1):
                uniq.append(s)
    return uniq


def integrate_radial_mode(
    bg: NewtonBackground,
    omega: float,
    *,
    normalize_surface: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the mode ODE on the full PDF grid (1001 points)."""
    prob = RadialCowlingProblem(bg)
    rhat = np.asarray(bg.rhat, dtype=float)
    y0 = prob.build_inner_y0(float(omega), None)

    def rhs(r: float, y: np.ndarray, w: float, params: Any) -> np.ndarray:
        return prob.rhs_inner(r, y, w, params)

    y = integrate_fixed_grid(rhs, rhat, y0, args=(float(omega), None), center_hook=prob.center_hook_inner)
    xihat = np.asarray(y[:, 0], dtype=float)
    deltah = np.asarray(y[:, 1], dtype=float)

    if normalize_surface:
        s = float(xihat[-1])
        if s == 0.0 or not np.isfinite(s):
            raise RuntimeError("Surface normalization failed: xihat(1) is zero or non-finite.")
        xihat = xihat / s
        deltah = deltah / s

    return rhat, xihat, deltah


def count_nodes(xihat: np.ndarray, *, atol: float = 0.0) -> int:
    """Count interior nodes (zero crossings) of xihat excluding rhat=0."""
    x = np.asarray(xihat, dtype=float)
    if x.size < 3:
        return 0
    sgn = np.sign(x[1:])  # exclude center
    # replace zeros by nearest previous nonzero sign (simple, stable enough)
    for i in range(1, sgn.size):
        if sgn[i] == 0.0:
            sgn[i] = sgn[i - 1]
    # if still zero (all zero), return 0
    if np.all(sgn == 0.0):
        return 0
    return int(np.sum(sgn[1:] * sgn[:-1] < 0.0))


def solve_radial_modes(
    *,
    K: float = 3.0,
    n: float = np.sqrt(3.0),
    rhoc: float = 1.28e-3,
    omega_min: float = 1e-3,
    omega_max: float = 0.2,
    n_scan: int = 250,
    max_modes: int = 3,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
    coarse_steps: Tuple[int, int] = (200, 200),
    fine_steps: Tuple[int, int] = (500, 500),
) -> List[NewtonRadialMode]:
    """High-level helper used by scripts: build background, scan seeds, refine roots, label by nodes."""
    sol_bg = solve_newton_structure_problem1(K=float(K), n=float(n), rhoc=float(rhoc))
    bg = build_background_from_structure(sol_bg)
    prob = RadialCowlingProblem(bg)

    seeds = scan_omega_seeds(
        prob,
        float(omega_min),
        float(omega_max),
        n_scan=int(n_scan),
        n_steps_inner=int(coarse_steps[0]),
        n_steps_outer=int(coarse_steps[1]),
        r_match=0.5,
        abs_f_tol=1e-2,
    )
    if not seeds:
        # fall back: simple geometric guesses
        seeds = [0.02, 0.04, 0.06]

    roots = solve_shooting(
        prob,
        float(seeds[0]),
        params=None,
        r_match=0.5,
        n_steps_inner=int(fine_steps[0]),
        n_steps_outer=int(fine_steps[1]),
        tol=float(tol),
        maxiter=int(maxiter),
        dx_rel=float(dx_rel),
        coarse_steps=coarse_steps,
        fine_steps=fine_steps,
        multi_start=seeds,
        return_all=True,
        cluster_tol=1e-8,
    )

    # Evaluate nodes and keep unique modes
    modes: List[NewtonRadialMode] = []
    for w in sorted([float(r) for r in roots]):
        rhat, xihat, deltah = integrate_radial_mode(bg, w, normalize_surface=True)
        nodes = count_nodes(xihat)
        # We want the lowest three modes by node count: 0,1,2.
        if nodes >= max_modes:
            continue
        modes.append(
            NewtonRadialMode(
                mode_index=int(nodes),
                omega=float(w),
                nodes=int(nodes),
                rhat=rhat,
                xihat=xihat,
                delta_h=deltah,
                bg=bg,
            )
        )

    # de-duplicate by mode_index, keep smallest omega for each
    best: Dict[int, NewtonRadialMode] = {}
    for m in modes:
        k = int(m.mode_index)
        if k not in best or m.omega < best[k].omega:
            best[k] = m

    out = [best[k] for k in sorted(best.keys())]
    return out[:max_modes]


def write_problem2_dat(filename: str, mode: NewtonRadialMode) -> None:
    """Write Problem 2 output format (see spec/NRGW_SPEC.md)."""
    lines: List[str] = []
    lines.append(f"{mode.bg.M}, {mode.bg.Rs}, {mode.omega}\n")
    for r, xi, dh in zip(mode.rhat, mode.xihat, mode.delta_h):
        lines.append(f"{float(r)}, {float(xi)}, {float(dh)}\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
