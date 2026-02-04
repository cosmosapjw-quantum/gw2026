# rk2_ivp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union, List, Tuple
import numpy as np


@dataclass
class OdeResult:
    """
    solve_ivp-like result container.

    - t: (m,)
    - y:
        * non-batch: (n, m)
        * batch:     (n, batch, m)
    - sol: callable for dense_output (piecewise linear)
    """
    t: np.ndarray
    y: np.ndarray
    nfev: int
    status: int
    message: str
    success: bool
    t_events: Optional[List[np.ndarray]] = None
    y_events: Optional[List[np.ndarray]] = None
    sol: Optional[Callable[[Union[float, np.ndarray]], np.ndarray]] = None


def _prepare_y0(y0) -> Tuple[np.ndarray, bool]:
    """
    Returns y as (n, batch) and is_batch flag.
    - if y0 is (n,), returns (n,1), is_batch=False
    - if y0 is (n,batch), returns as is, is_batch=True
    """
    y = np.asarray(y0, dtype=float)
    if y.ndim == 1:
        return y.reshape(-1, 1), False
    if y.ndim == 2:
        return y, True
    raise ValueError("y0 must be 1D (n,) or 2D (n,batch).")


def _error_norm_max_rms(err: np.ndarray, y: np.ndarray, y_new: np.ndarray,
                        atol_vec: np.ndarray, rtol: float) -> float:
    """
    err,y,y_new: (n,batch)
    Returns max over batch of RMS(err/scale) over components.
    """
    scale = atol_vec[:, None] + rtol * np.maximum(np.abs(y), np.abs(y_new))
    scaled = err / scale
    rms = np.sqrt(np.mean(scaled * scaled, axis=0))   # (batch,)
    return float(np.max(rms))


def solve_ivp_rk2_heun(
    fun: Callable,
    t_span: Tuple[float, float],
    y0,
    *,
    t_eval: Optional[np.ndarray] = None,
    dense_output: bool = False,
    events: Optional[Union[Callable, Sequence[Callable]]] = None,
    args: Optional[tuple] = None,
    rtol: float = 1e-6,
    atol: Union[float, np.ndarray] = 1e-9,
    max_step: float = np.inf,
    first_step: Optional[float] = None,
    adaptive: bool = True,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 5.0,
    max_nsteps: int = 1_000_000,
    event_tol: float = 1e-10,
) -> OdeResult:
    """
    Heun (explicit trapezoid / improved Euler) RK2 with solve_ivp-like signature.

    fun(t, y, *args) -> dy/dt
      - non-batch: y shape (n,)
      - batch:     y shape (n,batch)

    Notes:
      - Adaptive mode uses Heun–Euler embedded estimate: err = y_heun - y_euler.
      - dense_output uses piecewise linear interpolation of accepted steps.
      - events supported ONLY for non-batch mode.
        event(t,y)->float, may have attributes: terminal (bool), direction (-1,0,+1).
    """
    if args is None:
        args = ()

    t0, tf = float(t_span[0]), float(t_span[1])
    if tf == t0:
        y_mat, is_batch = _prepare_y0(y0)
        y_out = y_mat[:, 0, None] if not is_batch else y_mat[:, :, None]
        return OdeResult(
            t=np.array([t0], dtype=float),
            y=y_out,
            nfev=0,
            status=0,
            message="t_span has zero length.",
            success=True,
        )

    direction = 1.0 if tf > t0 else -1.0

    y, is_batch = _prepare_y0(y0)  # (n,batch)
    n, batch = y.shape

    atol_arr = np.asarray(atol, dtype=float)
    if atol_arr.ndim == 0:
        atol_vec = np.full(n, float(atol_arr))
    elif atol_arr.shape == (n,):
        atol_vec = atol_arr
    else:
        raise ValueError("atol must be scalar or shape (n,)")

    # events: non-batch only (for simplicity + clarity)
    if events is not None and is_batch:
        raise NotImplementedError("events are supported only for non-batch (y0 shape (n,)) mode.")

    if events is None:
        ev_list: List[Callable] = []
    elif callable(events):
        ev_list = [events]
    else:
        ev_list = list(events)

    t_events: List[List[float]] = [[] for _ in range(len(ev_list))]
    y_events: List[List[np.ndarray]] = [[] for _ in range(len(ev_list))]

    if ev_list:
        y_vec0 = y[:, 0]
        g_prev = [float(np.asarray(ev(t0, y_vec0, *args)).reshape(())) for ev in ev_list]
    else:
        g_prev = []

    # choose initial step
    if first_step is None:
        h = min(max_step, abs(tf - t0) / 100.0)
    else:
        h = float(first_step)
        if h <= 0:
            raise ValueError("first_step must be positive.")
        h = min(h, max_step)
    h *= direction

    # storage of accepted steps
    ts = [t0]
    ys = [y.copy()]
    nfev = 0

    status = 0
    message = "The solver successfully reached the end of the integration interval."

    def heun_step(t: float, y: np.ndarray, h: float):
        """
        Heun RK2:
          k1 = f(t, y)
          y_euler = y + h*k1
          k2 = f(t+h, y_euler)
          y_heun  = y + (h/2)(k1+k2)
          err = y_heun - y_euler  (Heun-Euler embedded estimate)
        """
        k1 = np.asarray(fun(t, y, *args), dtype=float)
        y_euler = y + h * k1
        k2 = np.asarray(fun(t + h, y_euler, *args), dtype=float)
        y_heun = y + 0.5 * h * (k1 + k2)
        err = y_heun - y_euler
        return y_heun, err, 2  # 2 function evals

    t = t0
    step_count = 0

    while (t - tf) * direction < 0:
        if step_count >= max_nsteps:
            status = -1
            message = "Maximum number of steps exceeded."
            break
        step_count += 1

        # don't step past tf
        if (t + h - tf) * direction > 0:
            h = tf - t

        y_new, err, fev = heun_step(t, y, h)
        nfev += fev

        if not np.all(np.isfinite(y_new)):
            if not adaptive:
                status = -1
                message = "Non-finite state encountered."
                break
            h *= 0.5
            continue

        err_nrm = 0.0
        accept = True
        if adaptive:
            err_nrm = _error_norm_max_rms(err, y, y_new, atol_vec, rtol)
            accept = (err_nrm <= 1.0)

        if accept:
            t_old = t
            y_old = y
            t = t + h
            y = y_new

            ts.append(t)
            ys.append(y.copy())

            # event handling (non-batch only)
            if ev_list:
                y_old_vec = y_old[:, 0]
                y_vec = y[:, 0]
                for i, ev in enumerate(ev_list):
                    g0 = g_prev[i]
                    g1 = float(np.asarray(ev(t, y_vec, *args)).reshape(()))

                    direction_attr = float(getattr(ev, "direction", 0.0))
                    terminal = bool(getattr(ev, "terminal", False))

                    hit = False
                    if direction_attr == 0.0:
                        hit = (g0 == 0.0) or (g1 == 0.0) or (g0 * g1 < 0.0)
                    elif direction_attr > 0.0:
                        hit = (g0 < 0.0 and g1 > 0.0)
                    else:
                        hit = (g0 > 0.0 and g1 < 0.0)

                    if hit and (g1 != g0):
                        # locate event by bisection using linear interpolation of y
                        a, bnd = t_old, t
                        ga, gb = g0, g1
                        for _ in range(60):
                            if abs(bnd - a) <= event_tol:
                                break
                            mid = 0.5 * (a + bnd)
                            w = (mid - t_old) / (t - t_old)
                            y_mid = y_old_vec + (y_vec - y_old_vec) * w
                            gm = float(np.asarray(ev(mid, y_mid, *args)).reshape(()))
                            if ga * gm <= 0:
                                bnd, gb = mid, gm
                            else:
                                a, ga = mid, gm

                        te = 0.5 * (a + bnd)
                        w = (te - t_old) / (t - t_old)
                        ye = y_old_vec + (y_vec - y_old_vec) * w

                        t_events[i].append(te)
                        y_events[i].append(ye.copy())

                        if terminal:
                            status = 1
                            message = "A termination event occurred."
                            # truncate last stored point to event
                            ts[-1] = te
                            ys[-1][:, 0] = ye
                            t = te
                            y[:, 0] = ye
                            tf = te
                            break

                    g_prev[i] = g1

                if status == 1:
                    break

            # step update
            if adaptive:
                if err_nrm == 0.0:
                    fac = max_factor
                else:
                    # embedded error is O(h^2) => exponent 1/2
                    fac = safety * (err_nrm ** (-0.5))
                    fac = min(max_factor, max(min_factor, fac))
                h = h * fac
                if abs(h) > max_step:
                    h = direction * max_step

        else:
            # reject step
            if err_nrm == 0.0:
                fac = min_factor
            else:
                fac = safety * (err_nrm ** (-0.5))
                fac = min(1.0, max(min_factor, fac))
            h = h * fac
            if abs(h) < 1e-16:
                status = -1
                message = "Step size became too small."
                break

    # pack accepted steps
    ts_arr = np.asarray(ts, dtype=float)                 # (m,)
    ys_arr = np.stack(ys, axis=-1)                       # (n,batch,m)

    # dense output: piecewise linear interpolation
    sol = None
    if dense_output or (t_eval is not None):
        def sol_fn(tq):
            tq_arr = np.asarray(tq, dtype=float)
            scalar = (tq_arr.ndim == 0)
            tq_flat = tq_arr.reshape(-1)

            idx = np.searchsorted(ts_arr, tq_flat, side="right") - 1
            idx = np.clip(idx, 0, len(ts_arr) - 2)

            t0s = ts_arr[idx]
            t1s = ts_arr[idx + 1]
            w = (tq_flat - t0s) / (t1s - t0s)

            y0s = ys_arr[:, :, idx]     # (n,batch,nt)
            y1s = ys_arr[:, :, idx + 1]
            out = y0s + (y1s - y0s) * w

            if scalar:
                return out[:, 0, 0] if not is_batch else out[:, :, 0]
            return out.reshape((n, batch) + tq_arr.shape)

        sol = sol_fn

    # output at t_eval if provided
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        y_eval = sol(t_eval)  # (n,batch,nt)
        if not is_batch:
            y_out = y_eval[:, 0, :]
        else:
            y_out = y_eval
        t_out = t_eval
    else:
        if not is_batch:
            y_out = ys_arr[:, 0, :]
        else:
            y_out = ys_arr
        t_out = ts_arr

    # finalize events arrays
    if ev_list:
        t_events_out = [np.asarray(v, dtype=float) for v in t_events]
        y_events_out = [
            (np.vstack(v) if len(v) > 0 else np.empty((0, n), dtype=float))
            for v in y_events
        ]
    else:
        t_events_out = None
        y_events_out = None

    return OdeResult(
        t=t_out,
        y=y_out,
        nfev=nfev,
        status=status,
        message=message,
        success=(status >= 0),
        t_events=t_events_out,
        y_events=y_events_out,
        sol=sol if dense_output else None,
    )
