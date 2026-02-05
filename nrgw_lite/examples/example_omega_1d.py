"""
Example: 1D shooting to find an eigenfrequency ω.

We use a toy "spherical wave" ODE with a 2/r singular term:
    u'' + (2/r) u' + ω^2 u = 0   on rhat ∈ [0,1]
Regular center solution has u(0) finite and u'(0)=0.

We impose boundary conditions:
    inner: u(0)=1, u'(0)=0  (regular, arbitrary normalization)
    outer: u(1)=0, u'(1)=1  (arbitrary normalization)

Then match at rhat=0.5 using a proportionality condition:
    det([u_in, u_out; v_in, v_out]) = 0
i.e. u_in * v_out - u_out * v_in = 0

The true regular solution is u = sin(ω r)/(ω r), so zeros occur near ω = nπ.

Run:
  cd /mnt/data/gw2026-main
  python -m nrgw_lite.examples.example_omega_1d
"""
from __future__ import annotations

import numpy as np

from nrgw_lite.core.shooting import ShootingProblem, solve_shooting


class SphericalBesselEigen1D(ShootingProblem):
    def build_inner_y0(self, unknown, params) -> np.ndarray:
        # regular center: u(0)=1, u'(0)=0
        return np.array([1.0, 0.0], dtype=float)  # [u, v=u']

    def build_outer_y0(self, unknown, params) -> np.ndarray:
        # surface: u(1)=0, choose u'(1)=1 (scaling arbitrary)
        return np.array([0.0, 1.0], dtype=float)

    def rhs_inner(self, rhat: float, y: np.ndarray, unknown, params) -> np.ndarray:
        omega = float(unknown)
        u = float(y[0])
        v = float(y[1])
        du = v
        dv = -(2.0 / rhat) * v - (omega ** 2) * u
        return np.array([du, dv], dtype=float)

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown, params) -> np.ndarray:
        # same ODE
        return self.rhs_inner(rhat, y, unknown, params)

    def match_residual(self, y_inner_match: np.ndarray, y_outer_match: np.ndarray, unknown, params) -> float:
        ui, vi = float(y_inner_match[0]), float(y_inner_match[1])
        uo, vo = float(y_outer_match[0]), float(y_outer_match[1])
        # proportionality (cross-product) condition
        return ui * vo - uo * vi

    @staticmethod
    def center_hook_inner(fun, r0: float, y0: np.ndarray, dr: float, args: tuple):
        """Analytic center limit for dv/dr when dv contains (2/r) v.

        Regular expansion gives:
            v(r) ~ a r,  a = -(ω^2/3) u(0)
        hence
            v'(0) = a = -(ω^2/3) u0
        """
        omega, _params = args
        omega = float(omega)

        u0 = float(np.asarray(y0, dtype=float)[0])
        v0 = float(np.asarray(y0, dtype=float)[1])

        du0 = v0
        dv0 = -(omega ** 2 / 3.0) * u0
        k1 = np.array([du0, dv0], dtype=float)
        y_pred = np.asarray(y0, dtype=float) + dr * k1
        return k1, y_pred


def main() -> None:
    problem = SphericalBesselEigen1D()

    # initial guess near the first root pi
    omega0 = 3.0

    omega = solve_shooting(
        problem,
        omega0,
        params=None,
        r_match=0.5,
        n_steps_inner=500,
        n_steps_outer=500,
        tol=1e-12,
        maxiter=30,
        dx_rel=1e-6,
    )

    print("Converged ω =", float(omega))
    print("Expected near π =", float(np.pi))
    print("last_info =", solve_shooting.last_info)

    # If you want multiple roots (OFF by default), enable scan+return_all:
    # roots = solve_shooting(problem, omega0, scan=(2.0, 12.0, 21), return_all=True)
    # print("roots:", roots)


if __name__ == "__main__":
    main()
