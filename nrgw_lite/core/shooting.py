from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ode_rk2 import integrate_fixed_grid
from .newton import newton1d, newton_nd

__all__ = [
    "ShootingProblem",
    "solve_shooting",
]


Unknown = Union[float, np.ndarray]


class ShootingProblem:
    """Interface for a two-sided (inner/outer) shooting + matching problem.

    Core idea:
      - integrate from rhat=0 -> rhat=r_match (inner)
      - integrate from rhat=1 -> rhat=r_match (outer)
      - build a residual vector at r_match using the two solutions
      - solve residual(unknown)=0 with Newton (1D/2D/3D)

    The solver assumes:
      - rhat is the independent coordinate on [0,1]
      - grids are fixed, uniform; no adaptive stepping.

    You implement the following methods:

      build_inner_y0(unknown, params) -> y0_inner
      build_outer_y0(unknown, params) -> y0_outer
      rhs_inner(rhat, y, unknown, params) -> dy/drhat
      rhs_outer(rhat, y, unknown, params) -> dy/drhat
      match_residual(y_inner_match, y_outer_match, unknown, params) -> R^d

    Optional:
      center_hook_inner / center_hook_outer:
        analytic-limit handler at rhat=0 for 1/r singular terms (NO epsilon tricks).
        It must follow integrate_fixed_grid's center_hook signature:
            center_hook(fun, r0, y0, dr, args) -> y1 OR (k1_at_0, y_pred_for_k2)
        where args will be (unknown, params).
    """

    # --- required user hooks ---
    def build_inner_y0(self, unknown: Unknown, params: Any) -> np.ndarray:
        raise NotImplementedError

    def build_outer_y0(self, unknown: Unknown, params: Any) -> np.ndarray:
        raise NotImplementedError

    def rhs_inner(self, rhat: float, y: np.ndarray, unknown: Unknown, params: Any) -> np.ndarray:
        raise NotImplementedError

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown: Unknown, params: Any) -> np.ndarray:
        raise NotImplementedError

    def match_residual(
        self,
        y_inner_match: np.ndarray,
        y_outer_match: np.ndarray,
        unknown: Unknown,
        params: Any,
    ) -> Union[float, np.ndarray]:
        raise NotImplementedError

    # --- optional analytic-center hooks (default: None) ---
    center_hook_inner: Optional[Callable] = None
    center_hook_outer: Optional[Callable] = None


def _as_unknown_vec(u: Any) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim != 1:
        raise ValueError("unknown must be a scalar or 1D array-like.")
    return u_arr


def _unknown_dim(unknown0: Any) -> int:
    if np.isscalar(unknown0):
        return 1
    return int(_as_unknown_vec(unknown0).size)


def _make_grids(
    *,
    r_match: float,
    n_steps_inner: int,
    n_steps_outer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r_match_f = float(r_match)
    if not (0.0 < r_match_f < 1.0):
        raise ValueError("r_match must lie strictly inside (0,1).")
    if n_steps_inner <= 0 or n_steps_outer <= 0:
        raise ValueError("n_steps_inner and n_steps_outer must be positive.")

    r_inner = np.linspace(0.0, r_match_f, int(n_steps_inner) + 1, dtype=float)
    r_outer = np.linspace(1.0, r_match_f, int(n_steps_outer) + 1, dtype=float)
    return r_inner, r_outer


def _precheck_center(fun: Callable, r0: float, y0: np.ndarray, args: tuple) -> None:
    """Guardrail: if r0==0 and no center_hook is provided, ensure fun is finite at r=0."""
    if not np.isclose(float(r0), 0.0):
        return
    try:
        k = fun(float(r0), np.asarray(y0, dtype=float), *args)
        k = np.asarray(k, dtype=float)
        if not np.all(np.isfinite(k)):
            raise FloatingPointError("non-finite RHS at r=0")
    except Exception as e:
        raise RuntimeError(
            "RHS evaluation at rhat=0 failed or returned non-finite values. "
            "Provide an analytic center_hook (no epsilon) OR implement an explicit rhat==0 branch "
            "inside rhs_inner."
        ) from e


def _integrate_both_sides(
    problem: ShootingProblem,
    unknown: Unknown,
    params: Any,
    *,
    r_match: float,
    n_steps_inner: int,
    n_steps_outer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_inner_match, y_outer_match) at rhat=r_match."""
    r_inner, r_outer = _make_grids(r_match=r_match, n_steps_inner=n_steps_inner, n_steps_outer=n_steps_outer)

    # --- inner ---
    y0_inner = np.asarray(problem.build_inner_y0(unknown, params), dtype=float)
    if y0_inner.ndim not in (1, 2):
        raise ValueError("build_inner_y0 must return (n,) or (n,batch).")

    def fun_inner(rhat: float, y: np.ndarray, unknown_: Unknown, params_: Any) -> np.ndarray:
        return problem.rhs_inner(float(rhat), y, unknown_, params_)

    args_inner = (unknown, params)
    center_hook_in = getattr(problem, "center_hook_inner", None)
    if center_hook_in is None:
        _precheck_center(fun_inner, r_inner[0], y0_inner, args_inner)

    y_inner = integrate_fixed_grid(fun_inner, r_inner, y0_inner, args=args_inner, center_hook=center_hook_in)

    # --- outer ---
    y0_outer = np.asarray(problem.build_outer_y0(unknown, params), dtype=float)
    if y0_outer.ndim not in (1, 2):
        raise ValueError("build_outer_y0 must return (n,) or (n,batch).")

    def fun_outer(rhat: float, y: np.ndarray, unknown_: Unknown, params_: Any) -> np.ndarray:
        return problem.rhs_outer(float(rhat), y, unknown_, params_)

    args_outer = (unknown, params)
    center_hook_out = getattr(problem, "center_hook_outer", None)
    if center_hook_out is None:
        _precheck_center(fun_outer, r_outer[0], y0_outer, args_outer)

    y_outer = integrate_fixed_grid(fun_outer, r_outer, y0_outer, args=args_outer, center_hook=center_hook_out)

    # Match-point states (last element: r=r_match for both grids)
    y_inner_m = y_inner[-1]
    y_outer_m = y_outer[-1]
    return np.asarray(y_inner_m, dtype=float), np.asarray(y_outer_m, dtype=float)


def _residual_vec(problem: ShootingProblem, unknown: Unknown, params: Any, *, r_match: float, n_steps_inner: int, n_steps_outer: int) -> np.ndarray:
    yi, yo = _integrate_both_sides(problem, unknown, params, r_match=r_match, n_steps_inner=n_steps_inner, n_steps_outer=n_steps_outer)
    R = problem.match_residual(yi, yo, unknown, params)
    R_arr = np.asarray(R, dtype=float).reshape(-1)
    if not np.all(np.isfinite(R_arr)):
        raise ValueError("match_residual returned non-finite values.")
    return R_arr


def _cluster_roots(
    roots: List[Unknown],
    *,
    cluster_tol: float,
) -> List[Unknown]:
    """Deduplicate roots by simple distance-based clustering."""
    if not roots:
        return []

    ct = float(cluster_tol)
    if ct <= 0.0:
        return roots

    kept: List[Unknown] = []

    def dist(a: Unknown, b: Unknown) -> float:
        if np.isscalar(a) and np.isscalar(b):
            return float(abs(float(a) - float(b)))
        aa = np.asarray(a, dtype=float).reshape(-1)
        bb = np.asarray(b, dtype=float).reshape(-1)
        return float(np.max(np.abs(aa - bb)))

    def scale(a: Unknown) -> float:
        if np.isscalar(a):
            return abs(float(a)) + 1.0
        aa = np.asarray(a, dtype=float).reshape(-1)
        return float(np.max(np.abs(aa)) + 1.0)

    for r in roots:
        ok = True
        for k in kept:
            if dist(r, k) <= ct * scale(k):
                ok = False
                break
        if ok:
            kept.append(r)

    # Sort for reproducibility
    if kept and np.isscalar(kept[0]):
        kept = sorted(kept, key=lambda x: float(x))  # type: ignore[arg-type]
    else:
        kept = sorted(kept, key=lambda x: float(np.asarray(x, dtype=float).reshape(-1)[0]))
    return kept


def solve_shooting(
    problem: ShootingProblem,
    unknown0: Unknown,
    params: Any = None,
    *,
    n_steps: int = 1000,
    n_steps_inner: Optional[int] = 500,
    n_steps_outer: Optional[int] = 500,
    r_match: float = 0.5,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
    jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    # 2-stage (optional)
    coarse_steps: Optional[Tuple[int, int]] = None,
    fine_steps: Optional[Tuple[int, int]] = None,
    coarse_tol: float = 1e-8,
    # multi-solution (optional; default OFF)
    scan: Optional[Union[Tuple[float, float, int], Sequence[Unknown]]] = None,
    multi_start: Optional[Sequence[Unknown]] = None,
    cluster_tol: float = 1e-6,
    return_all: bool = False,
) -> Union[Unknown, List[Unknown]]:
    """Solve a matching (shooting) problem via Newton-Raphson.

    Parameters
    ----------
    problem
        A ShootingProblem instance.
    unknown0
        Initial guess for the unknown(s):
          - scalar for 1D
          - (d,) array-like for dD.
    params
        Arbitrary user parameters forwarded into problem hooks.
    n_steps, n_steps_inner, n_steps_outer
        Fixed-step counts.
        If n_steps_inner/n_steps_outer are None, they are derived from n_steps.
    r_match
        Matching coordinate, default 0.5 (as in the PDF).
    tol, maxiter, dx_rel
        Newton settings. tol is on ||R||_inf (vector) or |r| (scalar).
    jac
        Optional analytic Jacobian for the Newton solve (ND only).
        If None, Newton uses central-difference FD Jacobian (see core/newton.py).
    project
        Optional projection for ND unknowns (e.g., enforce positivity).
    coarse_steps, fine_steps
        Two-stage solve:
          - run a coarse solve first (coarse_steps, coarse_tol)
          - use result as initial guess for fine solve (fine_steps, tol).
        If coarse_steps is None, only the final solve is performed.
    scan / multi_start
        Multi-solution support (OFF by default):
          - multi_start: list of initial guesses
          - scan:
              * for 1D: (umin, umax, ngrid) OR a list of initial guesses
              * for ND: list of initial guesses
        Each initial guess is solved independently, then roots are clustered.
    cluster_tol
        Relative-ish tolerance for clustering duplicates.
    return_all
        If True, always return a list of roots (even if only one).

    Returns
    -------
    root or roots
        If multi_start/scan is not used and return_all is False: returns a single root.
        Otherwise returns a list of (deduplicated) roots.

    Notes
    -----
    Diagnostics are available as:
        solve_shooting.last_info = {...}
    """
    # Step counts
    if n_steps_inner is None or n_steps_outer is None:
        n_steps_i = int(n_steps // 2)
        n_steps_o = int(n_steps - n_steps_i)
    else:
        n_steps_i = int(n_steps_inner)
        n_steps_o = int(n_steps_outer)

    def run_newton(u0: Unknown, *, n_i: int, n_o: int, tol_run: float) -> Unknown:
        d = _unknown_dim(u0)

        if d == 1 and np.isscalar(u0):
            def f_scalar(x: float) -> float:
                R = _residual_vec(problem, float(x), params, r_match=r_match, n_steps_inner=n_i, n_steps_outer=n_o)
                if R.size != 1:
                    raise ValueError(f"1D unknown requires 1D residual, got shape {R.shape}.")
                return float(R[0])

            x = newton1d(f_scalar, float(u0), tol=tol_run, maxiter=maxiter, dx_rel=dx_rel)
            info = getattr(newton1d, "last_info", {}).copy()
            solve_shooting.last_info = {"stage": "newton1d", **info}
            return float(x)

        # ND case
        x0 = _as_unknown_vec(u0)

        def F_vec(x: np.ndarray) -> np.ndarray:
            R = _residual_vec(problem, x, params, r_match=r_match, n_steps_inner=n_i, n_steps_outer=n_o)
            if R.size != x.size:
                raise ValueError(f"Residual dim {R.size} does not match unknown dim {x.size}.")
            return R

        x = newton_nd(F_vec, x0, jac=jac, tol=tol_run, maxiter=maxiter, dx_rel=dx_rel, project=project)
        info = getattr(newton_nd, "last_info", {}).copy()
        solve_shooting.last_info = {"stage": "newton_nd", **info}
        return x

    def run_two_stage(u0: Unknown) -> Unknown:
        u = u0
        if coarse_steps is not None:
            nci, nco = int(coarse_steps[0]), int(coarse_steps[1])
            u = run_newton(u, n_i=nci, n_o=nco, tol_run=float(coarse_tol))
        if fine_steps is not None:
            nfi, nfo = int(fine_steps[0]), int(fine_steps[1])
            u = run_newton(u, n_i=nfi, n_o=nfo, tol_run=float(tol))
        else:
            u = run_newton(u, n_i=n_steps_i, n_o=n_steps_o, tol_run=float(tol))
        return u

    # Determine starting guesses
    starts: List[Unknown] = []
    if multi_start is not None:
        starts = list(multi_start)
    elif scan is not None:
        if isinstance(scan, tuple) and len(scan) == 3 and _unknown_dim(unknown0) == 1:
            a, b, ngrid = float(scan[0]), float(scan[1]), int(scan[2])
            starts = [float(x) for x in np.linspace(a, b, ngrid)]
        else:
            starts = list(scan)  # type: ignore[arg-type]
    else:
        starts = [unknown0]

    roots: List[Unknown] = []
    failures: int = 0

    for s in starts:
        try:
            root = run_two_stage(s)
            roots.append(root)
        except Exception:
            failures += 1
            continue

    roots = _cluster_roots(roots, cluster_tol=cluster_tol)

    solve_shooting.last_info = {
        "n_starts": len(starts),
        "n_roots": len(roots),
        "n_failures": failures,
        **getattr(solve_shooting, "last_info", {}),
    }

    if return_all or (scan is not None) or (multi_start is not None):
        return roots
    if not roots:
        raise RuntimeError("solve_shooting: no converged solution found (all starts failed).")
    return roots[0]
