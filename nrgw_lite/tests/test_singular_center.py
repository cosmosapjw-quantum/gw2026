import numpy as np

from nrgw_lite.core.ode_rk2 import integrate_fixed_grid


def _singular_rhs(r: float, y: np.ndarray) -> np.ndarray:
    """RHS with an explicit 1/r singularity at r=0.

    dy/dr = y/r

    For the regular solution y = C r, the limit at r->0 is finite.
    This test ensures integrate_fixed_grid can avoid calling RHS at r=0 via center_hook.
    """
    return y / r


def _center_hook(fun, r0: float, y0: np.ndarray, dr: float, args: tuple):
    """Analytic first step for the regular solution y=r (C=1).

    For y = r, we have (dy/dr)|_{r=0} = 1.
    Provide k1_at_0=1 and the Euler predictor y_pred = y0 + dr*k1.
    """
    k1 = np.ones_like(y0, dtype=float)
    y_pred = y0 + dr * k1
    return k1, y_pred


def test_center_hook_avoids_r0_and_matches_regular_solution() -> None:
    r = np.linspace(0.0, 1.0, 1001)

    # Start exactly at the center.
    y0 = np.array([0.0], dtype=float)

    y = integrate_fixed_grid(_singular_rhs, r, y0, center_hook=_center_hook)
    assert np.all(np.isfinite(y))

    # Regular solution with C=1 is y(r) = r.
    y_exact = r
    max_err = float(np.max(np.abs(y[:, 0] - y_exact)))

    # RK2 global error ~ O(dr^2)
    assert max_err < 2.0e-6
