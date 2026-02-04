# numerical.py
# Generalized shooting + damped Newton built on rk2_ivp.solve_ivp_rk2_heun
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union, Any, Dict, List
import numpy as np

# rk2_ivp.py must export these:
from rk2_ivp import solve_ivp_rk2_heun, OdeResult

ArrayLike = Union[np.ndarray, Sequence[float]]


# ============================================================
# 1) IVP configuration (solve_ivp-like) for the Heun RK2 engine
# ============================================================

@dataclass
class HeunIVPConfig:
    """
    Configuration for solve_ivp_rk2_heun.

    Modes
    -----
    (A) Fixed-step mode:
        set n_steps (adaptive will be forced to False)
    (B) Adaptive mode:
        set adaptive=True and tolerances (rtol/atol)

    Notes
    -----
    - This intentionally mimics the "spirit" of solve_ivp options:
      rtol/atol, step-size control, max_step/first_step, etc.
    """
    adaptive: bool = True
    rtol: float = 1e-6
    atol: Union[float, np.ndarray] = 1e-9

    # Step controls
    max_step: float = np.inf
    first_step: Optional[float] = None
    n_steps: Optional[int] = None  # if set => fixed-step

    # Adaptive controller knobs
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0

    # Hard limit
    max_nsteps: int = 1_000_000

    def kwargs_for(self, t_span: Tuple[float, float]) -> Dict[str, Any]:
        """Convert to solve_ivp_rk2_heun kwargs."""
        t0, tf = float(t_span[0]), float(t_span[1])

        kwargs: Dict[str, Any] = dict(
            rtol=float(self.rtol),
            atol=self.atol,
            max_step=float(self.max_step),
            first_step=self.first_step,
            adaptive=bool(self.adaptive),
            safety=float(self.safety),
            min_factor=float(self.min_factor),
            max_factor=float(self.max_factor),
            max_nsteps=int(self.max_nsteps),
            dense_output=False,
            events=None,
        )

        # fixed-step override
        if self.n_steps is not None:
            n_steps = int(self.n_steps)
            if n_steps <= 0:
                raise ValueError("n_steps must be a positive integer.")
            dt = abs(tf - t0) / n_steps
            kwargs["adaptive"] = False
            kwargs["first_step"] = dt
            kwargs["max_step"] = dt

        return kwargs


# ============================================================
# 2) Small utilities: batch support wrappers
# ============================================================

def _ensure_2d_params(p: np.ndarray, d: int) -> Tuple[np.ndarray, bool]:
    """
    Normalize parameters to shape (d, k).
    Returns (P, is_batch). If p is (d,), returns (d,1) and is_batch=False.
    Accepts also (k,d) and transposes.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        if p.size != d:
            raise ValueError(f"parameter size mismatch: expected {d}, got {p.size}")
        return p.reshape(d, 1), False
    if p.ndim == 2:
        if p.shape[0] == d:
            return p, True
        if p.shape[1] == d:
            return p.T, True
        raise ValueError(f"parameter matrix must be (d,k) or (k,d) with d={d}, got {p.shape}")
    raise ValueError("parameters must be 1D or 2D array.")


def _wrap_fun_batch(fun: Callable) -> Callable:
    """
    If fun is written for y shape (n,), this wrapper enables y shape (n,batch)
    by looping over the batch dimension. Slower but general.
    """
    def fun_batched(t: float, y: np.ndarray, *args):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return np.asarray(fun(t, y, *args), dtype=float)
        if y.ndim != 2:
            raise ValueError("State y must be 1D or 2D (n,batch).")
        n, batch = y.shape
        out = np.empty((n, batch), dtype=float)
        for j in range(batch):
            out[:, j] = np.asarray(fun(t, y[:, j], *args), dtype=float)
        return out
    return fun_batched


def _wrap_map_params_to_y0(y0_from_p: Callable, d: int) -> Callable:
    """
    Ensure y0_from_p supports:
      - p: (d,)  -> y0: (n,)
      - P: (d,k) -> y0: (n,k)
    Fallback: loop over k.
    """
    def y0_builder(p_any: np.ndarray):
        p_any = np.asarray(p_any, dtype=float)
        if p_any.ndim == 1:
            return np.asarray(y0_from_p(p_any), dtype=float)
        P, _ = _ensure_2d_params(p_any, d)
        ys = [np.asarray(y0_from_p(P[:, j]), dtype=float) for j in range(P.shape[1])]
        return np.stack(ys, axis=1)  # (n,k)
    return y0_builder


def _wrap_map_terminal_to_residual(res_from_terminal: Callable, d: int) -> Callable:
    """
    Ensure residual_from_terminal supports:
      - (t_end, y_end(n,), p(d,)) -> F(m,)
      - (t_end, y_end(n,k), P(d,k)) -> F(m,k)
    Fallback: loop over k.
    """
    def res_builder(t_end: float, y_end: np.ndarray, p_any: np.ndarray):
        p_any = np.asarray(p_any, dtype=float)
        y_end = np.asarray(y_end, dtype=float)

        if p_any.ndim == 1:
            return np.asarray(res_from_terminal(t_end, y_end, p_any), dtype=float)

        P, _ = _ensure_2d_params(p_any, d)
        if y_end.ndim == 1:
            raise ValueError("batch params provided but y_end is not batched")
        k = P.shape[1]
        outs = [np.asarray(res_from_terminal(t_end, y_end[:, j], P[:, j]), dtype=float) for j in range(k)]
        return np.stack(outs, axis=1)  # (m,k)
    return res_builder


# ============================================================
# 3) ShootingIVP: "residual(p)" built on solve_ivp_rk2_heun
# ============================================================

@dataclass
class ShootingIVP:
    """
    Generic shooting residual wrapper.

    User provides:
      - fun(t, y, *args) -> dy/dt
      - y0_from_params(p) -> y0
      - residual_from_terminal(t_end, y_end, p) -> F(p)

    This class provides:
      - residual(p) for p shape (d,) or (d,k)
      - two-stage configs (coarse/fine) to mimic your pipeline

    Shape conventions:
      p: (d,) or (d,k)
      y0: (n,) or (n,k)
      F: (m,) or (m,k)
    """
    fun: Callable
    t_span: Tuple[float, float]
    y0_from_params: Callable[[np.ndarray], np.ndarray]
    residual_from_terminal: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    args: tuple = ()
    param_dim: Optional[int] = None

    # If True, fun/y0/residual must support batch (d,k)->(n,k)->(m,k).
    # If False, we wrap with loop-based fallbacks.
    vectorized: bool = True

    # Default 2-stage schedule similar to your old code
    coarse: HeunIVPConfig = field(default_factory=lambda: HeunIVPConfig(adaptive=False, n_steps=80))
    fine: HeunIVPConfig = field(default_factory=lambda: HeunIVPConfig(adaptive=False, n_steps=150))

    def _ensure_ready(self, p0: np.ndarray):
        if self.param_dim is None:
            self.param_dim = int(np.asarray(p0).size)

        d = int(self.param_dim)
        if not self.vectorized:
            self.fun = _wrap_fun_batch(self.fun)
            self.y0_from_params = _wrap_map_params_to_y0(self.y0_from_params, d)
            self.residual_from_terminal = _wrap_map_terminal_to_residual(self.residual_from_terminal, d)

    def residual(self, p: np.ndarray, stage: str = "fine") -> np.ndarray:
        """Evaluate residual for parameters p (shape (d,) or (d,k))."""
        self._ensure_ready(p)
        d = int(self.param_dim)

        P, is_batch = _ensure_2d_params(np.asarray(p, dtype=float), d)
        cfg = self.coarse if stage == "coarse" else self.fine

        # Initial condition(s)
        y0 = self.y0_from_params(P if is_batch else P[:, 0])
        y0 = np.asarray(y0, dtype=float)

        # Integrate
        res: OdeResult = solve_ivp_rk2_heun(
            self.fun,
            self.t_span,
            y0,
            args=self.args,
            **cfg.kwargs_for(self.t_span),
        )

        # Terminal state
        y_end = res.y[:, :, -1] if is_batch else res.y[:, -1]

        # Residual
        F = self.residual_from_terminal(float(res.t[-1]), y_end, P if is_batch else P[:, 0])
        return np.asarray(F, dtype=float)


# ============================================================
# 4) Damped Newton with vectorized FD Jacobian (batch evaluation)
# ============================================================

@dataclass
class RootResult:
    x: np.ndarray
    success: bool
    status: int
    message: str
    nit: int
    ncall: int              # number of residual-function calls (batch calls)
    residual_norm: float


def _norm2_cols(F: np.ndarray) -> np.ndarray:
    """F: (m,k) -> norms: (k,)"""
    return np.sqrt(np.sum(F*F, axis=0))


def jacobian_and_fx_central_vectorized(
    Ffun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    rel_step: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Central-difference Jacobian using ONE vectorized call:
      X = [x, x±h e1, x±h e2, ...]  shape (d, 2d+1)
      F = Ffun(X)                  shape (m, 2d+1)
    """
    x = np.asarray(x, dtype=float)
    d = x.size

    h = rel_step * np.maximum(1.0, np.abs(x))
    h = np.where(np.abs(x) < h, 0.5 * np.maximum(np.abs(x), 1e-12), h)

    X = np.empty((d, 2*d + 1), dtype=float)
    X[:, 0] = x
    for i in range(d):
        X[:, 2*i + 1] = x
        X[:, 2*i + 2] = x
        X[i, 2*i + 1] = x[i] + h[i]
        X[i, 2*i + 2] = x[i] - h[i]

    FX = np.asarray(Ffun(X), dtype=float)   # (m,2d+1)
    Fx = FX[:, 0]
    m = Fx.size
    J = np.empty((m, d), dtype=float)
    for i in range(d):
        J[:, i] = (FX[:, 2*i + 1] - FX[:, 2*i + 2]) / (2.0 * h[i])

    return Fx, J, 1


def newton_damped_vectorized(
    Ffun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    tol: float = 1e-10,
    xtol: float = 1e-10,
    maxiter: int = 25,
    rel_step: float = 1e-6,
    armijo_c: float = 1e-3,
    alpha_min: float = 2.0**-20,
    project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> RootResult:
    """
    Damped Newton for square systems F(x)=0.

    Ffun must accept:
      - x shape (d,)  -> F shape (d,)
      - X shape (d,k) -> F shape (d,k)

    Note: ncall counts residual-function calls (batch calls).
    """
    x = np.asarray(x0, dtype=float).copy()
    if project is not None:
        x = np.asarray(project(x), dtype=float)

    ncall = 0

    for it in range(int(maxiter)):
        Fx, J, c = jacobian_and_fx_central_vectorized(Ffun, x, rel_step=rel_step)
        ncall += c

        if (not np.all(np.isfinite(Fx))) or (not np.all(np.isfinite(J))):
            return RootResult(x=x, success=False, status=-2, message="Non-finite F or J.", nit=it, ncall=ncall, residual_norm=float("inf"))

        fn = float(np.linalg.norm(Fx))
        if fn < tol:
            return RootResult(x=x, success=True, status=0, message="Converged (residual).", nit=it, ncall=ncall, residual_norm=fn)

        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            return RootResult(x=x, success=False, status=-3, message="Singular Jacobian.", nit=it, ncall=ncall, residual_norm=fn)

        step_rel = float(np.max(np.abs(dx) / np.maximum(1.0, np.abs(x))))
        if step_rel < xtol:
            return RootResult(x=x, success=True, status=0, message="Converged (step).", nit=it, ncall=ncall, residual_norm=fn)

        # Backtracking (Armijo-like) on ||F||_2
        alpha = 1.0
        while alpha >= alpha_min:
            x_new = x - alpha * dx
            if project is not None:
                x_new = np.asarray(project(x_new), dtype=float)

            F_new = np.asarray(Ffun(x_new), dtype=float)
            ncall += 1
            fn_new = float(np.linalg.norm(F_new))

            if np.isfinite(fn_new) and (fn_new <= (1.0 - armijo_c * alpha) * fn):
                x = x_new
                break
            alpha *= 0.5

        if alpha < alpha_min:
            x = x - alpha_min * dx
            if project is not None:
                x = np.asarray(project(x), dtype=float)

    F_end = np.asarray(Ffun(x), dtype=float)
    ncall += 1
    return RootResult(
        x=x,
        success=False,
        status=1,
        message="Max iterations exceeded.",
        nit=int(maxiter),
        ncall=ncall,
        residual_norm=float(np.linalg.norm(F_end)),
    )


# ============================================================
# 5) Multi-start seeds + high-level "coarse->fine shooting solver"
# ============================================================

def lognormal_seeds(x0: np.ndarray, n_seeds: int = 64, sigma: float = 0.6, seed: int = 0) -> np.ndarray:
    """Log-space perturbations around x0. Returns (d, n_seeds)."""
    rng = np.random.default_rng(int(seed))
    x0 = np.asarray(x0, dtype=float)
    d = x0.size
    Z = rng.normal(0.0, sigma, size=(d, int(n_seeds)))
    seeds = x0[:, None] * np.exp(Z)
    seeds[:, 0] = x0
    return np.clip(seeds, 1e-300, 1e300)


@dataclass
class HeunShootingSolver:
    """
    High-level solver that mimics your previous pipeline:

      1) coarse residual check
      2) (optional) coarse Newton if far
      3) fine Newton
      4) retry: multi-start (lognormal seeds) + coarse pruning + fine solve

    The only requirement is that your residual is implemented via ShootingIVP.
    """
    problem: ShootingIVP
    project: Optional[Callable[[np.ndarray], np.ndarray]] = None

    coarse_trigger: float = 1e-2
    coarse_tol: float = 1e-4
    coarse_maxiter: int = 10

    def solve(
        self,
        p0: ArrayLike,
        *,
        tol: float = 1e-10,
        xtol: float = 1e-10,
        maxiter: int = 25,
        rel_step: float = 1e-6,
        n_retry: int = 1,
        n_seeds: int = 128,
        prune_topk: int = 10,
        sigma: float = 0.6,
        rng_seed: int = 0,
    ) -> RootResult:
        p0 = np.asarray(p0, dtype=float)
        self.problem._ensure_ready(p0)

        def F_coarse(P): return self.problem.residual(P, stage="coarse")
        def F_fine(P):   return self.problem.residual(P, stage="fine")

        # ---- coarse check ----
        F0 = np.asarray(self.problem.residual(p0, stage="coarse"), dtype=float)
        nrm0 = float(np.linalg.norm(F0))

        p_init = p0.copy()
        total_calls = 1

        if (not np.isfinite(nrm0)) or (nrm0 > self.coarse_trigger):
            rr0 = newton_damped_vectorized(
                F_coarse, p_init,
                tol=self.coarse_tol, xtol=xtol, maxiter=self.coarse_maxiter,
                rel_step=rel_step, project=self.project
            )
            p_init = rr0.x
            total_calls += rr0.ncall

        # ---- fine Newton ----
        rr = newton_damped_vectorized(
            F_fine, p_init,
            tol=tol, xtol=xtol, maxiter=maxiter,
            rel_step=rel_step, project=self.project
        )
        rr.ncall += total_calls - 1
        if rr.success:
            return rr

        # ---- retries: multi-start + pruning ----
        best = rr
        for k in range(int(n_retry)):
            seeds = lognormal_seeds(
                best.x,
                n_seeds=n_seeds,
                sigma=(sigma + 0.2*k),
                seed=(rng_seed + 1000*k),
            )  # (d, n_seeds)

            # coarse prune in ONE batch call
            F_seeds = np.asarray(self.problem.residual(seeds, stage="coarse"), dtype=float)  # (m, n_seeds)
            norms = _norm2_cols(F_seeds)
            norms = np.where(np.isfinite(norms), norms, np.inf)

            idx = np.argsort(norms)[:int(prune_topk)]
            top = seeds[:, idx].T  # (topk, d)

            candidates: List[RootResult] = []
            for j in range(top.shape[0]):
                cand = newton_damped_vectorized(
                    F_fine, top[j],
                    tol=tol, xtol=xtol, maxiter=maxiter,
                    rel_step=rel_step, project=self.project
                )
                candidates.append(cand)

            succ = [c for c in candidates if c.success]
            best_c = min(succ, key=lambda c: c.residual_norm) if succ else min(candidates, key=lambda c: c.residual_norm)

            best_c.ncall += best.ncall
            best = best_c
            if best.success:
                return best

        return best
