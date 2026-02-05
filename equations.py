import numpy as np

K = 50.0
n = 1.0

# 현재 코드는 일부분 외에는 직접 작성함

class Equations:
    def __init__(self):
        # 0점 근처 phi''(0) 극한값
        self.CENTER_RHAT = 1e-6

    def rho_from_h(self, h):
        h = np.asarray(h, dtype=float)
        h_pos = np.maximum(h, 0.0) # 엔탈피 음수 방지(해당 내용은 수치적 안정성을 위해 AI에게 추천받은 부분)
        return (h_pos / (K * (n + 1.0)))**n

    def hdot(self, rhat, phi, phidot, h, Rs):
        return -phidot
    
    def phiddot(self, rhat, phi, phidot, h, Rs):
        rhat = float(rhat)
        Rs = np.asarray(Rs, dtype=float)

        rho = self.rho_from_h(h)

        if rhat <= self.CENTER_RHAT:
            return (4.0/3.0) * np.pi * (Rs**2) * rho
        else:
            return -(2.0 / rhat) * phidot + 4.0 * np.pi * (Rs**2) * rho
