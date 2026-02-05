from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union, Any

import numpy as np

__all__ = [
    "heun_step",
    "integrate_fixed_grid",
]


ArrayLike = Union[np.ndarray, Sequence[float]]


def _validate_y_shape(y: np.ndarray) -> None:
    if y.ndim not in (1, 2):
        raise ValueError("y must be 1D (n,) or 2D (n,batch).")
    if y.ndim == 2 and y.shape[0] == 0:
        raise ValueError("y has zero state dimension.")


def _call_fun(fun: Callable, r: float, y: np.ndarray, args: tuple) -> np.ndarray:
    out = fun(r, y, *args)
    out_arr = np.asarray(out, dtype=float)
    return out_arr


def _heun_step_single(fun: Callable, r: float, y: np.ndarray, dr: float, args: tuple) -> np.ndarray:
    """Heun step for a single (non-batch) state y with shape (n,)."""
    k1 = _call_fun(fun, r, y, args)
    if k1.shape != y.shape:
        raise ValueError(f"fun returned shape {k1.shape}, expected {y.shape}.")

    y_euler = y + dr * k1
    k2 = _call_fun(fun, r + dr, y_euler, args)
    if k2.shape != y.shape:
        raise ValueError(f"fun returned shape {k2.shape}, expected {y.shape}.")

    return y + 0.5 * dr * (k1 + k2)


def heun_step(
    fun: Callable,
    r: float,
    y: ArrayLike,
    dr: float,
    args: tuple = (),
) -> np.ndarray:
    """One fixed Heun (RK2) step.

    Parameters
    ----------
    fun
        RHS function: fun(r, y, *args) -> dy/dr.
        - If y is (n,), return must be (n,).
        - If y is (n,batch), return should be (n,batch) (vectorized) OR you may
          provide a fun that only supports (n,); in that case we fall back to a
          per-batch loop.
    r
        Current coordinate.
    y
        Current state, shape (n,) or (n,batch).
    dr
        Step size (may be negative if integrating backward).
    args
        Extra positional arguments forwarded to fun.

    Returns
    -------
    y_next
        State at r+dr, same shape as y.
    """
    if args is None:
        args = ()

    r_f = float(r)
    dr_f = float(dr)

    y_arr = np.asarray(y, dtype=float)
    _validate_y_shape(y_arr)

    if y_arr.ndim == 1:
        return _heun_step_single(fun, r_f, y_arr, dr_f, args)

    # y is (n,batch)
    # Try vectorized evaluation first; if it fails or returns wrong shape,
    # fall back to looping over batch columns.
    try:
        k1 = _call_fun(fun, r_f, y_arr, args)
        if k1.shape != y_arr.shape:
            raise ValueError("shape mismatch")
        y_euler = y_arr + dr_f * k1
        k2 = _call_fun(fun, r_f + dr_f, y_euler, args)
        if k2.shape != y_arr.shape:
            raise ValueError("shape mismatch")
        return y_arr + 0.5 * dr_f * (k1 + k2)
    except Exception as err:
        n, batch = y_arr.shape
        y_next = np.empty_like(y_arr, dtype=float)
        loop_err: Optional[BaseException] = None
        for j in range(batch):
            try:
                y_next[:, j] = _heun_step_single(fun, r_f, y_arr[:, j], dr_f, args)
            except Exception as e:
                loop_err = e
                break
        if loop_err is not None:
            raise RuntimeError(
                "heun_step failed for batch input. "
                "Vectorized call failed and per-batch loop also failed."
            ) from err
        return y_next


def _validate_uniform_grid(r_grid: np.ndarray) -> float:
    if r_grid.ndim != 1:
        raise ValueError("r_grid must be a 1D numpy array.")
    if r_grid.size < 2:
        raise ValueError("r_grid must contain at least two points.")

    diffs = np.diff(r_grid)
    if not np.all(np.isfinite(diffs)):
        raise ValueError("r_grid contains non-finite values.")

    dr0 = float(diffs[0])
    if dr0 == 0.0:
        raise ValueError("r_grid has zero spacing.")

    # Strict enough for dr=1e-3 grids, but tolerant to FP noise.
    # (Absolute tolerance scales with |dr|.)
    atol = max(1e-15, 1e-12 * abs(dr0))
    if not np.allclose(diffs, dr0, rtol=0.0, atol=atol):
        raise ValueError("r_grid must be uniformly spaced.")

    return dr0


def integrate_fixed_grid(
    fun: Callable,
    r_grid: ArrayLike,
    y0: ArrayLike,
    args: tuple = (),
    *,
    center_hook: Optional[Callable[[Callable, float, np.ndarray, float, tuple], Any]] = None,
) -> np.ndarray:
    """Integrate on a fixed, uniformly-spaced 1D grid.

    Parameters
    ----------
    fun
        RHS function fun(r, y, *args) -> dy/dr.
    r_grid
        1D array of grid points. Must be uniformly spaced.
        (Typical NRGW: 1001 points on [0,1] with dr=1e-3.)
    y0
        Initial state at r_grid[0]. Shape (n,) or (n,batch).
    args
        Extra positional arguments forwarded to fun.
    center_hook
        Optional callable invoked ONLY for the very first step when r_grid[0]==0.
        Intended for analytic handling of singular RHS terms like 1/r at r=0.

        Signature:
            center_hook(fun, r0, y0, dr, args) -> one of:
              - y1
                  Directly return y(r0+dr). Must match shape of y0.
              - (k1_at_r0, y_pred_for_k2)
                  Provide analytic/regularized k1 at r0, and a predictor state
                  at r0+dr used to evaluate k2. We then compute
                      y1 = y0 + 0.5*dr*(k1 + k2)

        Notes:
            - If you return the (k1, y_pred) form, you get a true RK2 step.
            - If you return y1 directly, you are responsible for the accuracy.

    Returns
    -------
    y_grid
        Array of states on the grid:
          - if y0 is (n,), shape is (len(r_grid), n)
          - if y0 is (n,batch), shape is (len(r_grid), n, batch)

    """
    if args is None:
        args = ()

    r_arr = np.asarray(r_grid, dtype=float)
    dr0 = _validate_uniform_grid(r_arr)

    y0_arr = np.asarray(y0, dtype=float)
    _validate_y_shape(y0_arr)

    m = r_arr.size
    if y0_arr.ndim == 1:
        y_out = np.empty((m, y0_arr.size), dtype=float)
        y_out[0, :] = y0_arr
    else:
        # (m, n, batch)
        y_out = np.empty((m,) + y0_arr.shape, dtype=float)
        y_out[0, :, :] = y0_arr

    # Main loop (grid is uniform, but we use the stored dr0 for clarity)
    for i in range(m - 1):
        r_i = float(r_arr[i])
        y_i = y_out[i]

        # Special handling only for the *first* step at r=0.
        if center_hook is not None and i == 0 and np.isclose(r_i, 0.0):
            ret = center_hook(fun, r_i, np.asarray(y_i, dtype=float), float(dr0), args)

            if isinstance(ret, tuple):
                if len(ret) == 2:
                    k1, y_pred = ret
                    k1_arr = np.asarray(k1, dtype=float)
                    y_pred_arr = np.asarray(y_pred, dtype=float)
                    if k1_arr.shape != y_i.shape:
                        raise ValueError(
                            f"center_hook returned k1 shape {k1_arr.shape}, expected {y_i.shape}."
                        )
                    if y_pred_arr.shape != y_i.shape:
                        raise ValueError(
                            f"center_hook returned y_pred shape {y_pred_arr.shape}, expected {y_i.shape}."
                        )
                    k2_arr = _call_fun(fun, r_i + dr0, y_pred_arr, args)
                    if k2_arr.shape != y_i.shape:
                        raise ValueError(
                            f"fun returned shape {k2_arr.shape}, expected {y_i.shape}."
                        )
                    y_next = y_i + 0.5 * dr0 * (k1_arr + k2_arr)
                elif len(ret) == 1:
                    y_next = np.asarray(ret[0], dtype=float)
                else:
                    raise ValueError(
                        "center_hook must return y1 or (k1_at_0, y_pred_for_k2)."
                    )
            else:
                y_next = np.asarray(ret, dtype=float)

            if y_next.shape != y_i.shape:
                raise ValueError(
                    f"center_hook returned y1 shape {y_next.shape}, expected {y_i.shape}."
                )

        else:
            y_next = heun_step(fun, r_i, y_i, dr0, args=args)

        y_out[i + 1] = y_next

    return y_out
