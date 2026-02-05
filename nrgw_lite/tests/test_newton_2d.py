import numpy as np
import pytest

from nrgw_lite.core.newton import newton_nd


def F_system(x: np.ndarray) -> np.ndarray:
    # A simple nonlinear 2D system with a known root at (6, 1):
    #   x^2 + y - 37 = 0
    #   x - y^2 - 5 = 0
    return np.array([x[0] ** 2 + x[1] - 37.0, x[0] - x[1] ** 2 - 5.0], dtype=float)


def J_system(x: np.ndarray) -> np.ndarray:
    # Analytic Jacobian:
    # dF1/dx = 2x, dF1/dy = 1
    # dF2/dx = 1,  dF2/dy = -2y
    return np.array([[2.0 * x[0], 1.0], [1.0, -2.0 * x[1]]], dtype=float)


def test_newton_2d_fd_jacobian() -> None:
    x0 = np.array([5.0, 2.0], dtype=float)
    x = newton_nd(F_system, x0, tol=1e-12, maxiter=30, dx_rel=1e-6)

    assert np.allclose(x, np.array([6.0, 1.0]), rtol=0.0, atol=1e-10)
    assert newton_nd.last_info["converged"] is True
    assert newton_nd.last_info["F_norm"] < 1e-12


def test_newton_2d_analytic_jacobian_matches() -> None:
    x0 = np.array([5.0, 2.0], dtype=float)
    x = newton_nd(F_system, x0, jac=J_system, tol=1e-12, maxiter=20)

    assert np.allclose(x, np.array([6.0, 1.0]), rtol=0.0, atol=1e-12)
    assert newton_nd.last_info["converged"] is True
