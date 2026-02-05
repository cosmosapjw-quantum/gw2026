import numpy as np

from nrgw_lite.core.ode_rk2 import integrate_fixed_grid


def _sho_rhs(t: float, y: np.ndarray, omega: float) -> np.ndarray:
    """Simple harmonic oscillator.

    State: y = [x, v]
      dx/dt = v
      dv/dt = -omega^2 x

    This RHS is written to be batch-friendly: if y is (2,batch), it returns (2,batch).
    """
    x = y[0]
    v = y[1]
    return np.array([v, -(omega**2) * x], dtype=float)


def test_sho_single_and_batch() -> None:
    # Fixed grid: 1001 points on [0,1]
    t = np.linspace(0.0, 1.0, 1001)
    omega = 1.0

    # --- single trajectory ---
    y0 = np.array([1.0, 0.0], dtype=float)
    y = integrate_fixed_grid(_sho_rhs, t, y0, args=(omega,))

    x_exact = np.cos(omega * t)
    v_exact = -omega * np.sin(omega * t)

    err_x = float(np.max(np.abs(y[:, 0] - x_exact)))
    err_v = float(np.max(np.abs(y[:, 1] - v_exact)))

    # Heun is 2nd order (global error ~ O(dt^2) ~ 1e-6 for dt=1e-3)
    assert err_x < 2.0e-6
    assert err_v < 2.0e-6

    # --- batch trajectories ---
    # Two different initial conditions in one batch.
    # Column 0: x0=1, v0=0
    # Column 1: x0=0, v0=1
    y0b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)  # shape (2,2)
    yb = integrate_fixed_grid(_sho_rhs, t, y0b, args=(omega,))

    # analytic: x(t) = x0 cos(ωt) + (v0/ω) sin(ωt)
    #           v(t) = -x0 ω sin(ωt) + v0 cos(ωt)
    x0 = y0b[0]
    v0 = y0b[1]
    x_ex = x0[None, :] * np.cos(omega * t)[:, None] + (v0[None, :] / omega) * np.sin(omega * t)[:, None]
    v_ex = -x0[None, :] * omega * np.sin(omega * t)[:, None] + v0[None, :] * np.cos(omega * t)[:, None]

    err_xb = float(np.max(np.abs(yb[:, 0, :] - x_ex)))
    err_vb = float(np.max(np.abs(yb[:, 1, :] - v_ex)))

    assert err_xb < 2.0e-6
    assert err_vb < 2.0e-6
