import numpy as np
import time

K = 1.0*1e2
n = 0.8

class Equations:

    def __init__(self):
        pass

    def rho_from_h(self,h):
        h_pos = np.maximum(h, 0.0)
        return (h_pos / (K*(n+1.0)))**n

    def hdot(self,rhat,phi,phidot,h,Rs):
        return -phidot

    def phiddot(self,rhat,phi,phidot,h,Rs):
        if rhat < 1e-9:
            return (4.0/3.0) * np.pi * (Rs**2) * self.rho_from_h(h)
        else:
            return -(2./rhat)*phidot + 4.*np.pi* (Rs**2) * self.rho_from_h(h)

class Numerical(Equations):

    def __init__(self):
        pass

    def central_entalphy(self,rhoc):
        return K*(1+n)*rhoc**(1.0/n)

    # scipy.differentiate 관련 오류(phi0 Scope Error)를 방지합니다.
    def get_numerical_jacobian(self, f, x0, eps=1e-8):
        x0 = np.asarray(x0, dtype=float)
        n_dim = len(x0)
        fx = f(x0)
        jac = np.zeros((n_dim, n_dim))
        
        for i in range(n_dim):
            x_perturbed = x0.copy()
            x_perturbed[i] += eps
            fx_perturbed = f(x_perturbed)
            jac[:, i] = (fx_perturbed - fx) / eps
        return jac

    def nrmethod2D(self, f, x0, eps):
        x0 = np.asarray(x0, dtype=float)
        # Change: self.get_numerical_jacobian
        jac0 = self.get_numerical_jacobian(f, x0)
        fx0  = np.asarray(f(x0), dtype=float)
        x1   = x0 - np.linalg.solve(jac0, fx0)

        while True:
            x0 = x1
            # Change: self.get_numerical_jacobian
            jac1 = self.get_numerical_jacobian(f, x0)
            fx1  = np.asarray(f(x0), dtype=float)
            x1   = x0 - np.linalg.solve(jac1, fx1)
            if (np.abs(x1 - x0) < eps).all():
                break
        return x1

    def rkheunPoisson_full(self, phiddot, hdot, phi_ini, h_ini, t_ini, t_fin, point, Rs):
        step = (t_fin - t_ini) / point
        t = np.linspace(t_ini, t_fin, point+1)

        Rs_arr = np.asarray(Rs, dtype=float)
        
        # phi0 변수 초기화 명확화
        phi0 = np.asarray(phi_ini[0], dtype=float)
        phidot0 = np.asarray(phi_ini[1], dtype=float)
        h0 = np.asarray(h_ini, dtype=float)
        
        # 스칼라일 경우 1차원 배열로 변환 (브로드캐스팅 안전성 확보)
        if phi0.ndim == 0: phi0 = phi0.reshape(1)
        if phidot0.ndim == 0: phidot0 = phidot0.reshape(1)
        if h0.ndim == 0: h0 = h0.reshape(1)
        
        phi0, phidot0, h0, Rs_arr = np.broadcast_arrays(phi0, phidot0, h0, Rs_arr)
        batch_shape = phi0.shape

        phi = np.empty((point+1,)+batch_shape, dtype=float)
        phidot = np.empty((point+1,)+batch_shape, dtype=float)
        h = np.empty((point+1,)+batch_shape, dtype=float)

        phi[0] = phi0
        phidot[0] = phidot0
        h[0] = h0

        for i in range(point):
            k1phi = phiddot(t[i], phi[i], phidot[i], h[i], Rs_arr)
            k1h   = hdot(t[i], phi[i], phidot[i], h[i], Rs_arr)

            k2phi = phiddot(t[i+1], phi[i]+step*phidot[i], phidot[i]+step*k1phi, h[i]+step*k1h, Rs_arr)
            k2h   = hdot(t[i+1], phi[i]+step*phidot[i], phidot[i]+step*k1phi, h[i]+step*k1h, Rs_arr)

            phi[i+1]    = phi[i]    + step*phidot[i] + 0.5 * step**2 * k1phi
            phidot[i+1] = phidot[i] + 0.5*step*(k1phi+k2phi)
            h[i+1]      = h[i]      + 0.5*step*(k1h+k2h)

        return t, phi, phidot, h

    def rkheunPoisson_end(self, phiddot, hdot, phi_ini, h_ini, t_ini, t_fin, point, Rs):
        step = (t_fin - t_ini) / point
        t = np.linspace(t_ini, t_fin, point+1)

        Rs_arr = np.asarray(Rs, dtype=float)
        phi    = np.asarray(phi_ini[0], dtype=float)
        phidot = np.asarray(phi_ini[1], dtype=float)
        h      = np.asarray(h_ini, dtype=float)

        phi, phidot, h, Rs_arr = np.broadcast_arrays(phi, phidot, h, Rs_arr)

        for i in range(point):
            k1_phi    = phidot
            k1_phidot = phiddot(t[i], phi, phidot, h, Rs_arr)
            k1_h      = hdot(t[i], phi, phidot, h, Rs_arr)

            phi_p    = phi    + step*k1_phi
            phidot_p = phidot + step*k1_phidot
            h_p      = h      + step*k1_h

            k2_phi    = phidot_p
            k2_phidot = phiddot(t[i+1], phi_p, phidot_p, h_p, Rs_arr)
            k2_h      = hdot(t[i+1], phi_p, phidot_p, h_p, Rs_arr)

            phi    = phi    + 0.5*step*(k1_phi + k2_phi)
            phidot = phidot + 0.5*step*(k1_phidot + k2_phidot)
            h      = h      + 0.5*step*(k1_h + k2_h)

        return phi, phidot

    def bound_inter(self, rhoc, Rs):
        # NOTE: phiddot, hdot, self.central_entalphy must be defined globally or passed as self variables
        # Change: self.rkheunPoisson_end
        phi, phidot = self.rkheunPoisson_end(self.phiddot, self.hdot, [0.0, 0.0], self.central_entalphy(rhoc), 1e-6, 0.5, 500, Rs)
        return phi, phidot

    def bound_outer(self, Rs, Ms):
        # surface BC: phi(1)=-M/Rs, phidot(1)=+M/Rs (phidot = dphi/drhat)
        # Change: self.rkheunPoisson_end
        phi, phidot = self.rkheunPoisson_end(self.phiddot, self.hdot, [-Ms/Rs, Ms/Rs], 0.0, 1.0, 0.5, 500, Rs)
        return phi, phidot

    def func(self, x):
            x = np.asarray(x, dtype=float)
            Rs, Ms = x[0], x[1]
            Rs, Ms = np.abs(Rs), np.abs(Ms)
            
            h_c = self.central_entalphy(self.rhoc)

            phi_inter, phidot_inter = self.bound_inter(self.rhoc, Rs)
            phi_outer, phidot_outer = self.bound_outer(Rs, Ms)

            phi_diff = (phi_inter - phi_outer) - (h_c + Ms/Rs)
            phidot_diff = phidot_inter - phidot_outer

            return np.stack([phi_diff, phidot_diff], axis=0)

    def shooting(self, rhoc, Rs0, Ms0):
        self.rhoc = rhoc
        # Change: self.nrmethod2D and passing self.func
        Rs, Ms = self.nrmethod2D(self.func, np.array([Rs0, Ms0], dtype=float), 1e-8)
        return np.abs(Rs), np.abs(Ms)
    
def rm_generator():

    solver = Numerical()
    Rlist = np.empty(0)
    Mlist = np.empty(0)
    for rhoc in np.logspace(1e-9,1e-1,100,base=10):
        print("rhoc = ",rhoc)
        R, M = solver.shooting(rhoc,1.0,0.1)
        if R <= 2 * M:
                    break
        Rlist = np.append(Rlist,R)
        Mlist = np.append(Mlist,M)
    return Rlist, Mlist

Rlist, Mlist = rm_generator()

print(Rlist,Mlist)
