import numpy as np

# 물리 상수 (다른 모듈에서 import하여 사용)
K = 100.0
n = 1.0

class Equations:
    def __init__(self):
        pass

    def rho_from_h(self, h):
        """
        Smooth positive-part to reduce nonsmooth kink at h=0 (better FD Jacobian stability).
        If self.DELTA_H <= 0, falls back to exact max(h,0).
        """
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0)
        return (h_pos / (K * (n + 1.0)))**n

    def hdot(self, rhat, phi, phidot, h, Rs):
        return -phidot

    def phiddot(self, rhat, phi, phidot, h, Rs):
        if rhat < 1e-9:
            return (4.0/3.0) * np.pi * (Rs**2) * self.rho_from_h(h)
        else:
            return -(2./rhat)*phidot + 4.*np.pi* (Rs**2) * self.rho_from_h(h)