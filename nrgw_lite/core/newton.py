from __future__ import annotations

from typing import Callable, Optional, Tuple, Any
import numpy as np

__all__ = ["newton1d", "newton_nd"]


def _fd_step(x: float, dx_rel: float) -> float:
    """Relative finite-difference step with a floor."""
    dx = float(dx_rel) * (abs(float(x)) + 1.0)
    # avoid pathological zero/denorm steps
    if dx == 0.0:
        dx = float(dx_rel) if dx_rel != 0.0 else 1e-12
    return dx


def _norm_inf(v: np.ndarray) -> float:
    return float(np.max(np.abs(v))) if v.size else 0.0


def _as_vec(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x0 must be a 1D array-like of shape (d,).")
    return x


def _as_F(F: Any, d: int) -> np.ndarray:
    Fv = np.asarray(F, dtype=float)
    if Fv.ndim != 1 or Fv.shape[0] != d:
        raise ValueError(f"F(x) must return shape ({d},), got {Fv.shape}.")
    if not np.all(np.isfinite(Fv)):
        raise ValueError("F(x) returned non-finite values.")
    return Fv


def _fd_jacobian_2d(F: Callable[[np.ndarray], Any], x: np.ndarray, dx_rel: float) -> np.ndarray:
    x0, x1 = float(x[0]), float(x[1])
    dx0 = _fd_step(x0, dx_rel)
    dx1 = _fd_step(x1, dx_rel)

    xp = x.copy(); xm = x.copy()

    xp[0] = x0 + dx0; xm[0] = x0 - dx0
    Fp0 = _as_F(F(xp), 2); Fm0 = _as_F(F(xm), 2)
    col0 = (Fp0 - Fm0) / (2.0 * dx0)

    xp[:] = x; xm[:] = x
    xp[1] = x1 + dx1; xm[1] = x1 - dx1
    Fp1 = _as_F(F(xp), 2); Fm1 = _as_F(F(xm), 2)
    col1 = (Fp1 - Fm1) / (2.0 * dx1)

    J = np.column_stack((col0, col1))
    return J


def _fd_jacobian_3d(F: Callable[[np.ndarray], Any], x: np.ndarray, dx_rel: float) -> np.ndarray:
    x0, x1, x2 = float(x[0]), float(x[1]), float(x[2])
    dx0 = _fd_step(x0, dx_rel)
    dx1 = _fd_step(x1, dx_rel)
    dx2 = _fd_step(x2, dx_rel)

    xp = x.copy(); xm = x.copy()

    xp[0] = x0 + dx0; xm[0] = x0 - dx0
    Fp0 = _as_F(F(xp), 3); Fm0 = _as_F(F(xm), 3)
    col0 = (Fp0 - Fm0) / (2.0 * dx0)

    xp[:] = x; xm[:] = x
    xp[1] = x1 + dx1; xm[1] = x1 - dx1
    Fp1 = _as_F(F(xp), 3); Fm1 = _as_F(F(xm), 3)
    col1 = (Fp1 - Fm1) / (2.0 * dx1)

    xp[:] = x; xm[:] = x
    xp[2] = x2 + dx2; xm[2] = x2 - dx2
    Fp2 = _as_F(F(xp), 3); Fm2 = _as_F(F(xm), 3)
    col2 = (Fp2 - Fm2) / (2.0 * dx2)

    J = np.column_stack((col0, col1, col2))
    return J


def _fd_jacobian_nd(F: Callable[[np.ndarray], Any], x: np.ndarray, dx_rel: float) -> np.ndarray:
    d = x.size
    if d == 2:
        return _fd_jacobian_2d(F, x, dx_rel)
    if d == 3:
        return _fd_jacobian_3d(F, x, dx_rel)

    J = np.empty((d, d), dtype=float)
    for j in range(d):
        dxj = _fd_step(float(x[j]), dx_rel)
        xp = x.copy(); xm = x.copy()
        xp[j] += dxj
        xm[j] -= dxj
        Fp = _as_F(F(xp), d)
        Fm = _as_F(F(xm), d)
        J[:, j] = (Fp - Fm) / (2.0 * dxj)
    return J


def newton1d(
    f: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
) -> float:
    """1D Newton-Raphson with central-difference derivative (PDF eq. 90, 92).

    Returns
    -------
    x : float
        Estimated root.

    Notes
    -----
    Diagnostics are stored as:
        newton1d.last_info = {"converged": bool, "niter": int, "f_abs": float}
    """
    x = float(x0)
    tol_f = float(tol)

    fx = float(f(x))
    if not np.isfinite(fx):
        raise ValueError("f(x0) is not finite.")
    if abs(fx) < tol_f:
        newton1d.last_info = {"converged": True, "niter": 0, "f_abs": abs(fx)}
        return x

    for it in range(1, int(maxiter) + 1):
        dx = _fd_step(x, dx_rel)
        fp = float(f(x + dx))
        fm = float(f(x - dx))
        if not (np.isfinite(fp) and np.isfinite(fm)):
            raise ValueError("Non-finite values encountered during FD derivative.")

        dfdx = (fp - fm) / (2.0 * dx)
        if dfdx == 0.0 or not np.isfinite(dfdx):
            raise RuntimeError(f"Newton1D: derivative is zero/non-finite at iter={it} (x={x}).")

        step = -fx / dfdx

        # Armijo-lite backtracking to avoid blow-ups (kept intentionally tiny).
        alpha = 1.0
        x_new = x + step
        f_new = float(f(x_new))
        for _ in range(8):
            if np.isfinite(f_new) and abs(f_new) <= (1.0 - 1e-4 * alpha) * abs(fx):
                break
            alpha *= 0.5
            x_new = x + alpha * step
            f_new = float(f(x_new))

        x, fx = x_new, f_new
        if abs(fx) < tol_f:
            newton1d.last_info = {"converged": True, "niter": it, "f_abs": abs(fx)}
            return x

    newton1d.last_info = {"converged": False, "niter": int(maxiter), "f_abs": abs(float(fx))}
    raise RuntimeError(f"Newton1D did not converge in {maxiter} iterations. |f|={abs(fx):.3e}, x={x}.")


def newton_nd(
    F: Callable[[np.ndarray], Any],
    x0: Any,
    *,
    jac: Optional[Callable[[np.ndarray], Any]] = None,
    tol: float = 1e-12,
    maxiter: int = 30,
    dx_rel: float = 1e-6,
    project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Multi-dimensional Newton-Raphson (PDF eq. 98~100) with optional FD Jacobian (eq. 92 style).

    Parameters
    ----------
    F
        Vector function: F(x) -> (d,) residual.
    x0
        Initial guess, shape (d,).
    jac
        Optional Jacobian function: jac(x) -> (d,d). If None, uses central-difference FD Jacobian.
    tol
        Convergence tolerance on ||F||_inf (max abs component).
    maxiter
        Maximum number of Newton iterations.
    dx_rel
        Relative FD step size (per component): dx_j = dx_rel*(|x_j|+1).
    project
        Optional projection: x <- project(x). Useful for enforcing simple constraints.

    Returns
    -------
    x : np.ndarray
        Estimated root, shape (d,).

    Notes
    -----
    Diagnostics are stored as:
        newton_nd.last_info = {"converged": bool, "niter": int, "F_norm": float}
    """
    x = _as_vec(x0).astype(float, copy=True)
    d = int(x.size)
    if d == 0:
        raise ValueError("x0 must have at least one dimension.")
    if project is not None:
        x = _as_vec(project(x)).astype(float, copy=False)

    Fx = _as_F(F(x), d)
    fnorm = _norm_inf(Fx)
    if fnorm < float(tol):
        newton_nd.last_info = {"converged": True, "niter": 0, "F_norm": float(fnorm)}
        return x

    tol_f = float(tol)

    for it in range(1, int(maxiter) + 1):
        if jac is None:
            J = _fd_jacobian_nd(F, x, float(dx_rel))
        else:
            J = np.asarray(jac(x), dtype=float)
            if J.shape != (d, d):
                raise ValueError(f"jac(x) must return shape ({d},{d}), got {J.shape}.")
            if not np.all(np.isfinite(J)):
                raise ValueError("jac(x) returned non-finite values.")

        # Solve J * dx = -F (PDF eq. 98).
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # fallback: least-squares step (still small and often helps near singular J)
            dx, *_ = np.linalg.lstsq(J, -Fx, rcond=None)

        if not np.all(np.isfinite(dx)):
            raise RuntimeError(f"NewtonND: non-finite step at iter={it}.")

        # Armijo-lite backtracking (intentionally compact) on ||F||_inf.
        alpha = 1.0
        x_new = x + dx
        if project is not None:
            x_new = _as_vec(project(x_new))
        F_new = _as_F(F(x_new), d)
        fn_new = _norm_inf(F_new)

        for _ in range(8):
            if fn_new <= (1.0 - 1e-4 * alpha) * fnorm or fn_new < tol_f:
                break
            alpha *= 0.5
            x_new = x + alpha * dx
            if project is not None:
                x_new = _as_vec(project(x_new))
            F_new = _as_F(F(x_new), d)
            fn_new = _norm_inf(F_new)

        x, Fx, fnorm = x_new, F_new, fn_new
        if fnorm < tol_f:
            newton_nd.last_info = {"converged": True, "niter": it, "F_norm": float(fnorm)}
            return x

    newton_nd.last_info = {"converged": False, "niter": int(maxiter), "F_norm": float(fnorm)}
    raise RuntimeError(f"NewtonND did not converge in {maxiter} iterations. ||F||_inf={fnorm:.3e}, x={x}.")
