from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.ode_rk2 import integrate_fixed_grid
from ..core.shooting import ShootingProblem, solve_shooting
from .newton_structure import NewtonStructureSolution, solve_newton_structure_problem1
from .newton_osc_radial import NewtonBackground, build_background_from_structure, _idx_from_rhat, count_nodes

__all__ = [
    "NewtonNonRadialL2Mode",
    "NonRadialL2CowlingProblem",
    "integrate_nonradial_l2_mode",
    "scan_omega_seeds_l2",
    "solve_nonradial_l2_modes",
    "write_problem3_dat",
]


L_DEFAULT = 2


@dataclass(frozen=True)
class NewtonNonRadialL2Mode:
    """A solved non-radial Cowling mode (l=2)."""
    mode_index: int
    omega: float
    nodes: int
    rhat: np.ndarray
    xihat: np.ndarray
    delta_h_hat: np.ndarray
    bg: NewtonBackground


class NonRadialL2CowlingProblem(ShootingProblem):
    """Problem 3: non-radial (l=2) Cowling oscillations on a Newtonian polytrope background.

    Variables (per PDF discussion):
      - xihat = xi / rhat^(l-1) = xi / rhat   (since l=2)
      - delta_h_hat = delta_h / rhat^l = delta_h / rhat^2

    State: y = [xihat, delta_h_hat]
    Unknown: omega (scalar)

    Matching at r_match uses scalar residual:
        f = xihat_in * dhhat_out - xihat_out * dhhat_in = 0
    """

    def __init__(self, bg: NewtonBackground, *, l: int = 2):
        if int(l) != 2:
            raise ValueError("This lite implementation supports only l=2.")
        self.bg = bg
        self.n = float(bg.n)
        self.Rs = float(bg.Rs)
        self.l = int(l)

        self.center_hook_inner = self._center_hook
        self.center_hook_outer = None

    # ---- BC builders ----
    def build_inner_y0(self, unknown: float, params: Any) -> np.ndarray:
        # Regularity at center: require numerator of dhhat' to vanish:
        #   Rs*omega^2*xihat(0) - l*dhhat(0) = 0  -> dhhat(0)=Rs*omega^2/l
        w = float(unknown)
        xihat0 = 1.0
        dhhat0 = (self.Rs * (w * w) / float(self.l)) * xihat0
        return np.array([xihat0, dhhat0], dtype=float)

    def build_outer_y0(self, unknown: float, params: Any) -> np.ndarray:
        # Surface: xihat(1)=1, delta_h(1)=-(1/Rs) dh0/drhat * xi(1)
        # For l=2, xi(1)=rhat^(l-1)*xihat(1)=1*xihat(1)=1.
        xihat1 = 1.0
        dhhat1 = -(1.0 / self.Rs) * float(self.bg.dh_s) * xihat1
        return np.array([xihat1, dhhat1], dtype=float)

    # ---- RHS ----
    def rhs_inner(self, rhat: float, y: np.ndarray, unknown: float, params: Any) -> np.ndarray:
        return self._rhs(float(rhat), np.asarray(y, dtype=float), float(unknown))

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown: float, params: Any) -> np.ndarray:
        return self._rhs(float(rhat), np.asarray(y, dtype=float), float(unknown))

    def _rhs(self, rhat: float, y: np.ndarray, omega: float) -> np.ndarray:
        xihat = float(y[0])
        dhhat = float(y[1])

        if rhat == 0.0:
            return np.array([0.0, 0.0], dtype=float)

        # Surface analytic-limit branch (h0=0)
        if rhat == 1.0:
            a = float(self.bg.dh_s)   # dh/drhat at surface
            b = float(self.bg.d2h_s)  # d2h/drhat2 at surface

            # dhhat' from the regular ODE is finite at rhat=1:
            #   dhhat' = (Rs*omega^2*xihat - l*dhhat)/rhat
            d_dh = (self.Rs * (omega * omega) * xihat - float(self.l) * dhhat) / 1.0

            # dxihat' needs L'Hospital to remove 0/0 in ( ... )/h0.
            # Derived (for l=2):
            #   dxihat/drhat|1 = -(xihat/(n+1))*(1/a) * [ (n+3)*a + n*(Rs^2*omega^2 + b) + 6*a^2/(omega^2*Rs^2) ]
            denom = a
            if denom == 0.0:
                raise ZeroDivisionError("Surface dh0/drhat is zero; cannot apply surface BC for xihat' (l=2).")
            if omega == 0.0:
                raise ZeroDivisionError("omega=0 not allowed for nonradial l=2 surface BC.")

            d_xi = -(xihat / (self.n + 1.0)) * (1.0 / denom) * (
                (self.n + 3.0) * denom
                + self.n * ((self.Rs * self.Rs) * (omega * omega) + b)
                + 6.0 * (denom * denom) / ((omega * omega) * (self.Rs * self.Rs))
            )

            return np.array([d_xi, d_dh], dtype=float)

        # Interior background
        i = _idx_from_rhat(rhat)
        h0 = float(self.bg.h0[i])
        dh0 = float(self.bg.dh0_drhat[i])

        if h0 <= 0.0:
            raise ValueError(f"h0(rhat={rhat}) is non-positive inside the star: h0={h0}")

        # dxihat/drhat:
        #   = -(l+1)/rhat xihat  - (n/h0)[ dh0*xihat + Rs*rhat*dhhat ] + l(l+1)/(omega^2*Rs*rhat) dhhat
        if omega == 0.0:
            raise ZeroDivisionError("omega=0 not allowed (appears in dxihat via 1/omega^2).")

        d_xi = -((self.l + 1.0) / rhat) * xihat - (self.n / h0) * (dh0 * xihat + self.Rs * rhat * dhhat)
        d_xi = d_xi + (self.l * (self.l + 1.0)) * dhhat / ((omega * omega) * self.Rs * rhat)

        # d(dhhat)/drhat:
        #   = (Rs*omega^2*xihat - l*dhhat)/rhat
        d_dh = (self.Rs * (omega * omega) * xihat - float(self.l) * dhhat) / rhat

        return np.array([d_xi, d_dh], dtype=float)

    # ---- match ----
    def match_residual(self, y_inner_match: np.ndarray, y_outer_match: np.ndarray, unknown: float, params: Any) -> float:
        yi = np.asarray(y_inner_match, dtype=float).reshape(-1)
        yo = np.asarray(y_outer_match, dtype=float).reshape(-1)
        return float(yi[0] * yo[1] - yo[0] * yi[1])

    # ---- center hook ----
    @staticmethod
    def _center_hook(fun: Callable[..., Any], r0: float, y0: np.ndarray, dr: float, args: tuple):
        # With regularity-enforced y0, both derivatives vanish at rhat=0.
        k1 = np.zeros_like(y0, dtype=float)
        y_pred = np.asarray(y0, dtype=float)
        return k1, y_pred


def _integrate_to_match(problem: NonRadialL2CowlingProblem, omega: float, *, r_match: float, n_steps_inner: int, n_steps_outer: int) -> Tuple[np.ndarray, np.ndarray]:
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


def scan_omega_seeds_l2(
    problem: NonRadialL2CowlingProblem,
    omega_min: float,
    omega_max: float,
    *,
    n_scan: int = 250,
    r_match: float = 0.5,
    n_steps_inner: int = 200,
    n_steps_outer: int = 200,
    abs_f_tol: float = 1e-3,
) -> List[float]:
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
    for i in range(len(w_grid) - 1):
        f0, f1 = f_grid[i], f_grid[i + 1]
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 == 0.0:
            seeds.append(float(w_grid[i]))
        if f0 * f1 < 0.0:
            seeds.append(float(0.5 * (w_grid[i] + w_grid[i + 1])))

    absf = np.abs(f_grid)
    for i in range(1, len(w_grid) - 1):
        if not np.isfinite(absf[i]):
            continue
        if absf[i] < abs_f_tol and absf[i] <= absf[i - 1] and absf[i] <= absf[i + 1]:
            seeds.append(float(w_grid[i]))

    # de-duplicate
    seeds_sorted = sorted(seeds)
    uniq: List[float] = []
    step = (wmax - wmin) / max(int(n_scan) - 1, 1)
    for s in seeds_sorted:
        if not uniq or abs(s - uniq[-1]) > 0.5 * step:
            uniq.append(s)
    return uniq


def integrate_nonradial_l2_mode(
    bg: NewtonBackground,
    omega: float,
    *,
    normalize_surface: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = NonRadialL2CowlingProblem(bg, l=2)
    rhat = np.asarray(bg.rhat, dtype=float)
    y0 = prob.build_inner_y0(float(omega), None)

    def rhs(r: float, y: np.ndarray, w: float, params: Any) -> np.ndarray:
        return prob.rhs_inner(r, y, w, params)

    y = integrate_fixed_grid(rhs, rhat, y0, args=(float(omega), None), center_hook=prob.center_hook_inner)
    xihat = np.asarray(y[:, 0], dtype=float)
    dhhat = np.asarray(y[:, 1], dtype=float)

    if normalize_surface:
        s = float(xihat[-1])
        if s == 0.0 or not np.isfinite(s):
            raise RuntimeError("Surface normalization failed: xihat(1) is zero or non-finite.")
        xihat = xihat / s
        dhhat = dhhat / s

    return rhat, xihat, dhhat


def solve_nonradial_l2_modes(
    *,
    K: float = 3.0,
    n: float = np.sqrt(3.0),
    rhoc: float = 1.28e-3,
    omega_min: float = 1e-3,
    omega_max: float = 0.2,
    n_scan: int = 300,
    max_modes: int = 3,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
    coarse_steps: Tuple[int, int] = (200, 200),
    fine_steps: Tuple[int, int] = (500, 500),
) -> List[NewtonNonRadialL2Mode]:
    sol_bg = solve_newton_structure_problem1(K=float(K), n=float(n), rhoc=float(rhoc))
    bg = build_background_from_structure(sol_bg)
    prob = NonRadialL2CowlingProblem(bg, l=2)

    seeds = scan_omega_seeds_l2(
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

    modes: List[NewtonNonRadialL2Mode] = []
    for w in sorted([float(r) for r in roots]):
        rhat, xihat, dhhat = integrate_nonradial_l2_mode(bg, w, normalize_surface=True)
        nodes = count_nodes(xihat)
        if nodes >= max_modes:
            continue
        modes.append(
            NewtonNonRadialL2Mode(
                mode_index=int(nodes),
                omega=float(w),
                nodes=int(nodes),
                rhat=rhat,
                xihat=xihat,
                delta_h_hat=dhhat,
                bg=bg,
            )
        )

    best: Dict[int, NewtonNonRadialL2Mode] = {}
    for m in modes:
        k = int(m.mode_index)
        if k not in best or m.omega < best[k].omega:
            best[k] = m
    out = [best[k] for k in sorted(best.keys())]
    return out[:max_modes]


def write_problem3_dat(filename: str, mode: NewtonNonRadialL2Mode) -> None:
    lines: List[str] = []
    lines.append(f"{mode.bg.M}, {mode.bg.Rs}, {mode.omega}\n")
    for r, xi, dhh in zip(mode.rhat, mode.xihat, mode.delta_h_hat):
        lines.append(f"{float(r)}, {float(xi)}, {float(dhh)}\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
