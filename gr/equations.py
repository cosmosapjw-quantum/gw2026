import numpy as np

K = 50.0
n = 1.0

class Equations:
    def __init__(self):
        pass

    def rho_from_h(self, h):
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0)
        return (h_pos / (K * (n + 1.0)))**n

    def P_from_h(self, h):
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0)
        rho0 = self.rho_from_h(h_pos)
        return rho0 * h_pos / (n + 1.0)

    def eps_from_h(self, h):
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0)
        rho0 = self.rho_from_h(h_pos)
        P = self.P_from_h(h_pos)
        return rho0 + n * P

    # 해당 코드는 표준적인 TOV 방정식을 따라 작성함
    def tov_mdot(self, rhat, m, h, Rs):
        rhat = float(rhat)
        m = np.asarray(m, dtype=float)
        h = np.asarray(h, dtype=float)
        Rs = np.asarray(Rs, dtype=float)

        r = Rs * rhat
        eps = self.eps_from_h(h)

        denom_ok = (r > 0.0) & ((r - 2.0*m) > 0.0)
        dm = 4.0*np.pi * (Rs**3) * (rhat**2) * eps
        return np.where(denom_ok, dm, np.nan) #수치문제 해결을 위해 AI에게 추천받은 부분

    def tov_hdot(self, rhat, m, h, Rs):
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
        return np.where(denom_ok, dh, np.nan)  #수치문제 해결을 위해 AI에게 추천받은 부분
