"""
Example: 2D shooting for Newtonian stellar structure (unknowns: Rs, M).

This mirrors the PDF Sec.7 "shooting method" illustration:
  - integrate from rhat=0 -> 0.5 (inner)
  - integrate from rhat=1 -> 0.5 (outer)
  - match h and dPhi/drhat at rhat=0.5
  - solve for (Rs, M) with 2D Newton.

Run:
  cd /mnt/data/gw2026-main
  python -m nrgw_lite.examples.example_newton_structure_2d
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from nrgw_lite.core.shooting import ShootingProblem, solve_shooting


@dataclass
class Params:
    K: float
    n: float
    rho_c: float


class NewtonianStructure2D(ShootingProblem):
    """Newtonian structure in (h, phi') variables (PDF eq. 14) with unknowns (Rs, M)."""

    def build_inner_y0(self, unknown, params: Params) -> np.ndarray:
        Rs, M = float(unknown[0]), float(unknown[1])  # M unused for inner BC
        h_c = params.K * (params.n + 1.0) * (params.rho_c ** (1.0 / params.n))
        return np.array([h_c, 0.0], dtype=float)  # [h, phi']

    def build_outer_y0(self, unknown, params: Params) -> np.ndarray:
        Rs, M = float(unknown[0]), float(unknown[1])
        # surface BC: h(1)=0, phi'(1)=M/Rs
        return np.array([0.0, M / Rs], dtype=float)

    def rhs_inner(self, rhat: float, y: np.ndarray, unknown, params: Params) -> np.ndarray:
        Rs, M = float(unknown[0]), float(unknown[1])
        h, phip = float(y[0]), float(y[1])

        # rho = (h/(K(n+1)))^n ; clip h to avoid negative-power issues
        denom = params.K * (params.n + 1.0)
        hpos = max(h, 0.0)
        rho = (hpos / denom) ** params.n if hpos > 0.0 else 0.0

        dh = -phip
        dphip = -2.0 * phip / rhat + 4.0 * np.pi * (Rs ** 2) * rho
        return np.array([dh, dphip], dtype=float)

    def rhs_outer(self, rhat: float, y: np.ndarray, unknown, params: Params) -> np.ndarray:
        # same ODE
        return self.rhs_inner(rhat, y, unknown, params)

    def match_residual(self, y_inner_match: np.ndarray, y_outer_match: np.ndarray, unknown, params: Params) -> np.ndarray:
        # residual: outer - inner at rhat=0.5 for (h, phi')
        return np.array(
            [
                float(y_outer_match[0] - y_inner_match[0]),
                float(y_outer_match[1] - y_inner_match[1]),
            ],
            dtype=float,
        )

    @staticmethod
    def center_hook_inner(fun, r0: float, y0: np.ndarray, dr: float, args: tuple):
        """Analytic center limit for the 1/r term in dphi'/drhat.

        For regularity at rhat=0:
          phi'(0)=0 and phi''(0) = (4/3) pi rho_c Rs^2  (PDF eq. 15)
        """
        unknown, params = args
        Rs = float(np.asarray(unknown, dtype=float).reshape(-1)[0])
        rho_c = float(params.rho_c)

        # k1 = dy/drhat at rhat=0
        dh0 = 0.0                    # dh/dr = -phi' = 0
        dphip0 = (4.0 * np.pi / 3.0) * rho_c * (Rs ** 2)
        k1 = np.array([dh0, dphip0], dtype=float)

        # Euler predictor to r=dr, used to compute k2 at r=dr
        y_pred = np.asarray(y0, dtype=float) + dr * k1
        return k1, y_pred


def main() -> None:
    params = Params(K=100.0, n=1.0, rho_c=1.28e-3)
    problem = NewtonianStructure2D()

    # PDF Fig.5 caption suggests a decent initial guess:
    unknown0 = np.array([9.0, 1.4], dtype=float)  # (Rs, M)

    root = solve_shooting(
        problem,
        unknown0,
        params,
        r_match=0.5,
        n_steps_inner=500,
        n_steps_outer=500,
        tol=1e-12,
        maxiter=30,
        dx_rel=1e-6,
        # A gentle projection can help if Newton steps wander to Rs<=0
        project=lambda x: np.array([max(x[0], 1e-6), max(x[1], 1e-12)], dtype=float),
    )

    Rs, M = float(root[0]), float(root[1])
    info = solve_shooting.last_info
    print("Converged (Rs, M) =", (Rs, M))
    print("last_info =", info)


if __name__ == "__main__":
    main()
