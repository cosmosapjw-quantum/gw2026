import numpy as np
from scipy.integrate import solve_ivp
from scipy.differentiate import jacobian
from scipy.optimize import fsolve

K = 1.0*1e2
n = 0.8
rhoc = 1.28*1e-3
h_c = K*(1+n)*rhoc**(1.0/n)

def rho_from_h(h):
    h_pos = np.maximum(h, 0.0)  # avoid negative -> NaN for fractional n
    return (h_pos / (K*(n+1.0)))**n

def hdot(rhat,phi,phidot,h,Rs):
    return -phidot

def phiddot(rhat,phi,phidot,h,Rs):
    return -(2./rhat)*phidot + 4.*np.pi*Rs**2 * rho_from_h(h)

def nrmethod2D(f,x0,eps):
    x0 = np.asarray(x0, dtype=float)

    jac0 = jacobian(f, x0).df
    fx0  = np.asarray(f(x0), dtype=float)
    x1   = x0 - np.linalg.solve(jac0, fx0)

    while True:
        x0 = x1
        jac1 = jacobian(f, x0, order=4).df
        fx1  = np.asarray(f(x0), dtype=float)
        x1   = x0 - np.linalg.solve(jac1, fx1)
        if (np.abs(x1 - x0) < eps).all():
            break
    return x1

def rkheunPoisson_full(phiddot,hdot,phi_ini,h_ini,t_ini,t_fin,point,Rs):
    # step consistent with linspace endpoints
    step = (t_fin - t_ini) / (point - 1)
    t = np.linspace(t_ini, t_fin, point)

    # broadcast batch shape (supports scalar or vector Rs / ICs)
    Rs_arr = np.asarray(Rs, dtype=float)
    phi0 = np.asarray(phi_ini[0], dtype=float)
    phidot0 = np.asarray(phi_ini[1], dtype=float)
    h0 = np.asarray(h_ini, dtype=float)
    phi0, phidot0, h0, Rs_arr = np.broadcast_arrays(phi0, phidot0, h0, Rs_arr)
    batch_shape = phi0.shape

    phi = np.empty((point,)+batch_shape, dtype=float)
    phidot = np.empty((point,)+batch_shape, dtype=float)
    h = np.empty((point,)+batch_shape, dtype=float)

    phi[0] = phi0
    phidot[0] = phidot0
    h[0] = h0

    for i in range(point-1):
        k1phi = phiddot(t[i],   phi[i],                phidot[i],                h[i],                Rs_arr)
        k1h   = hdot  (t[i],   phi[i],                phidot[i],                h[i],                Rs_arr)

        k2phi = phiddot(t[i+1], phi[i]+step*phidot[i], phidot[i]+step*k1phi,     h[i]+step*k1h,       Rs_arr)
        k2h   = hdot  (t[i+1], phi[i]+step*phidot[i], phidot[i]+step*k1phi,      h[i]+step*k1h,       Rs_arr)

        phi[i+1]    = phi[i]    + step*phidot[i] + 0.5 * step**2 * k1phi
        phidot[i+1] = phidot[i] + 0.5*step*(k1phi+k2phi)
        h[i+1]      = h[i]      + 0.5*step*(k1h+k2h)

    return t, phi, phidot, h

def rkheunPoisson_end(phiddot,hdot,phi_ini,h_ini,t_ini,t_fin,point,Rs):
    # step consistent with linspace endpoints
    step = (t_fin - t_ini) / (point - 1)
    t = np.linspace(t_ini, t_fin, point)

    Rs_arr = np.asarray(Rs, dtype=float)
    phi = np.asarray(phi_ini[0], dtype=float)
    phidot = np.asarray(phi_ini[1], dtype=float)
    h = np.asarray(h_ini, dtype=float)

    # vectorize via broadcasting
    phi, phidot, h, Rs_arr = np.broadcast_arrays(phi, phidot, h, Rs_arr)

    for i in range(point-1):
        k1phi = phiddot(t[i],   phi,                 phidot,                 h,                 Rs_arr)
        k1h   = hdot  (t[i],   phi,                 phidot,                 h,                 Rs_arr)

        k2phi = phiddot(t[i+1], phi+step*phidot,     phidot+step*k1phi,      h+step*k1h,         Rs_arr)
        k2h   = hdot  (t[i+1], phi+step*phidot,     phidot+step*k1phi,       h+step*k1h,         Rs_arr)  # FIX: phi[i] -> phi

        phi    = phi    + step*phidot + 0.5 * step**2 * k1phi
        phidot = phidot + 0.5*step*(k1phi+k2phi)
        h      = h      + 0.5*step*(k1h+k2h)

    return phi, phidot

def bound_inter(Rs):
    phi, phidot = rkheunPoisson_end(phiddot, hdot, [0.0, 0.0], h_c, 1e-5, 0.5, 500, Rs)
    return phi, phidot

def bound_outer(Rs, Ms):
    # surface BC: phi(1)=-M/Rs, phidot(1)=+M/Rs (phidot = dphi/drhat)
    phi, phidot = rkheunPoisson_end(phiddot, hdot, [-Ms/Rs, Ms/Rs], 0.0, 1.0, 0.5, 500, Rs)
    return phi, phidot

def func(x):
    x = np.asarray(x, dtype=float)
    Rs, Ms = x[0], x[1]

    phi_inter, phidot_inter = bound_inter(Rs)
    phi_outer, phidot_outer = bound_outer(Rs, Ms)

    return np.stack([phi_inter-phi_outer, phidot_inter-phidot_outer], axis=0)

def shooting(Rs0, Ms0):
    Rs, Ms = nrmethod2D(func, np.array([Rs0, Ms0], dtype=float), 1e-5)
    return Rs, Ms

print(shooting(1.0, 1.0))