# equations.py
import numpy as np

# Polytrope parameters (natural units: G=c=1 assumed in TOV form)
K = 50.0
n = 1.0  # polytropic index
# Gamma = 1 + 1/n  (implicit)

class Equations:
    def __init__(self):
        # smoothing scale for rho_from_h (set per-rhoc in Numerical.shooting)
        self.DELTA_H = 0.0

    # -------------------------
    # EOS in terms of "h"
    # -------------------------
    def rho_from_h(self, h):
        """
        Treat rho(h) as rest-mass density rho0(h) defined by your enthalpy convention:
          h = K (n+1) rho0^(1/n)  ->  rho0 = (h/(K(n+1)))^n
        Optional smoothing near h=0 for FD stability:
          h_pos = (h + sqrt(h^2 + DELTA_H^2))/2
        """
        h = np.asarray(h, dtype=float)
        delta = float(getattr(self, "DELTA_H", 0.0))
        if delta > 0.0:
            h_pos = 0.5 * (h + np.sqrt(h*h + delta*delta))
        else:
            h_pos = np.maximum(h, 0.0)
        return (h_pos / (K * (n + 1.0)))**n

    def P_from_h(self, h):
        """
        Polytrope: P = K rho0^(1 + 1/n).
        Under your h-definition one can use the identity:
          P = rho0 * h / (n+1)
        """
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0)
        rho0 = self.rho_from_h(h_pos)
        return rho0 * h_pos / (n + 1.0)

    def eps_from_h(self, h):
        """
        Energy density epsilon (GR TOV 'rho' often means energy density):
          epsilon = rho0 + P/(Gamma-1) = rho0 + n*P
        """
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0)
        rho0 = self.rho_from_h(h_pos)
        P = self.P_from_h(h_pos)
        return rho0 + n * P

    # -------------------------
    # (Legacy) Newton/Poisson helpers (kept for reference)
    # -------------------------
    def hdot(self, rhat, phi, phidot, h, Rs):
        return -phidot

    def phiddot(self, rhat, phi, phidot, h, Rs):
        if rhat < 1e-9:
            return (4.0/3.0) * np.pi * (Rs**2) * self.rho_from_h(h)
        else:
            return -(2.0/rhat) * phidot + 4.0*np.pi*(Rs**2) * self.rho_from_h(h)

    # -------------------------
    # TOV in normalized radius rhat = r/Rs
    # unknowns: Rs (surface radius), Ms (total gravitational mass)
    #
    # Standard TOV (G=c=1):
    #   dm/dr   = 4π r^2 ε
    #   dP/dr   = -(ε+P) (m+4π r^3 P) / [r(r-2m)]
    # Here we evolve (m, h) using:
    #   dh/dr   = -(1+h) (m+4π r^3 P) / [r(r-2m)]
    # because with this EOS: ε+P = rho0(1+h).
    # We convert to rhat:
    #   dm/drhat = Rs * dm/dr
    #   dh/drhat = Rs * dh/dr
    # -------------------------
    def tov_mdot(self, rhat, m, h, Rs):
        """
        dm/drhat = 4π Rs^3 rhat^2 ε(h)
        """
        rhat = float(rhat)
        m = np.asarray(m, dtype=float)
        h = np.asarray(h, dtype=float)
        Rs = np.asarray(Rs, dtype=float)

        r = Rs * rhat
        eps = self.eps_from_h(h)

        denom_ok = (r > 0.0) & ((r - 2.0*m) > 0.0)
        dm = 4.0*np.pi * (Rs**3) * (rhat**2) * eps
        return np.where(denom_ok, dm, np.nan)

    def tov_hdot(self, rhat, m, h, Rs):
        """
        dh/drhat = -Rs (1+h_pos) (m+4π r^3 P) / [r(r-2m)]
        Note: we use h_pos in EOS and in (1+h) factor for robustness.
        """
        rhat = float(rhat)
        m = np.asarray(m, dtype=float)
        h = np.asarray(h, dtype=float)
        Rs = np.asarray(Rs, dtype=float)

        r = Rs * rhat
        h_pos = np.maximum(h, 0.0)
        P = self.P_from_h(h_pos)

        denom = r * (r - 2.0*m)
        num = m + 4.0*np.pi * (r**3) * P

        denom_ok = (r > 0.0) & (denom > 0.0)
        dh = -Rs * (1.0 + h_pos) * num / denom
        return np.where(denom_ok, dh, np.nan)
