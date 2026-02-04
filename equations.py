# equations.py
import numpy as np

# Polytrope parameters (Newtonian toy model)
K = 50.0
n = 1.0


class Equations:
    def __init__(self):
        # smoothing scale used by rho_from_h (set per-rhoc in Numerical.shooting)
        self.DELTA_H = 0.0
        # threshold to avoid the 2/r singular cancellation at the center
        self.CENTER_RHAT = 1e-6

    def rho_from_h(self, h):
        """
        rho(h) from your enthalpy definition:
          h = K(n+1) rho^(1/n)  -> rho = (h/(K(n+1)))^n
        Optional smoothing near h=0 for FD Jacobian stability:
          h_pos = (h + sqrt(h^2 + DELTA_H^2))/2
        """
        h = np.asarray(h, dtype=float)
        delta = float(getattr(self, "DELTA_H", 0.0))
        if delta > 0.0:
            h_pos = 0.5 * (h + np.sqrt(h*h + delta*delta))
        else:
            h_pos = np.maximum(h, 0.0)
        return (h_pos / (K * (n + 1.0)))**n

    # Newtonian hydrostatic relation in enthalpy form:
    # dh/dr = - dPhi/dr  ->  dh/drhat = - dPhi/drhat
    def hdot(self, rhat, phi, phidot, h, Rs):
        return -phidot

    # Poisson (spherical) in rhat = r/Rs:
    # d2Phi/drhat^2 = -(2/rhat) dPhi/drhat + 4π Rs^2 rho(h)
    def phiddot(self, rhat, phi, phidot, h, Rs):
        rhat = float(rhat)
        Rs = np.asarray(Rs, dtype=float)

        rho = self.rho_from_h(h)

        if rhat <= self.CENTER_RHAT:
            # regular center limit
            return (4.0/3.0) * np.pi * (Rs**2) * rho
        else:
            return -(2.0 / rhat) * phidot + 4.0 * np.pi * (Rs**2) * rho
