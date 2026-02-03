import numpy as np
from scipy.differentiate import jacobian
import time

K = 5.0*1e2
n = 1.0
rhoc = 1.28*1e-3
h_c = K*(1+n)*rhoc**(1.0/n)

# --- smoothing for "max(h,0)" to reduce nonsmooth-kink impact on numerical Jacobian
#     delta=0 -> exactly max(h,0). Small delta -> C^1-ish smoothing near 0.
DELTA_H = 1e-12 * max(1.0, abs(h_c))

def rho_from_h(h):
    # smooth positive part: h_pos = (h + sqrt(h^2 + delta^2))/2
    # delta->0 gives max(h,0) exactly
    h = np.asarray(h, dtype=float)
    h_pos = 0.5*(h + np.sqrt(h*h + DELTA_H*DELTA_H))
    return (h_pos / (K*(n+1.0)))**n

def hdot(rhat,phi,phidot,h,Rs):
    return -phidot

def phiddot(rhat,phi,phidot,h,Rs):
    return -(2./rhat)*phidot + 4.*np.pi*(Rs**2) * rho_from_h(h)

def nrmethod2D(f, x0, eps):
    """
    Damped Newton with a very light Armijo-style decrease test + positivity guard.
    Keep design similar: still uses scipy.differentiate.jacobian, but stabilized.
    """
    x = np.asarray(x0, dtype=float)

    # settings: prioritize robustness over fancy high-order differencing
    max_newton = 30
    c_armijo = 1e-4
    alpha_min = 2.0**-20

    for _ in range(max_newton):
        fx = np.asarray(f(x), dtype=float)
        if not np.all(np.isfinite(fx)):
            raise RuntimeError("f(x) became non-finite. Try different initial guess or smaller step.")

        fnorm = np.linalg.norm(fx, ord=2)
        if fnorm < eps:
            return x

        # set a fixed scale-aware initial_step for FD Jacobian (absolute step)
        # scipy requires it broadcastable with x. See jacobian docs.
        init_step = 1e-3 * np.maximum(1.0, np.abs(x))

        J = jacobian(
            f, x,
            order=2,          # less cancellation / less sensitivity than high order for noisy f
            maxiter=1,        # fewer Richardson refinements (stability first)
            initial_step=init_step
        ).df

        dx = np.linalg.solve(J, fx)

        # damping / line search: ensure decrease and keep Rs,Ms positive
        alpha = 1.0
        while alpha >= alpha_min:
            x_new = x - alpha*dx

            # positivity guard for Rs, Ms
            if (x_new[0] > 0.0) and (x_new[1] > 0.0):
                fx_new = np.asarray(f(x_new), dtype=float)
                if np.all(np.isfinite(fx_new)):
                    fnorm_new = np.linalg.norm(fx_new, ord=2)
                    # Armijo-like sufficient decrease on ||f|| (cheap & effective here)
                    if fnorm_new <= (1.0 - c_armijo*alpha) * fnorm:
                        x = x_new
                        break

            alpha *= 0.5

        # if line search failed, accept a very small step instead of exploding
        if alpha < alpha_min:
            x = np.maximum(x - alpha_min*dx, 1e-12)

        # step-based stopping (prevents endless tiny updates)
        if np.max(np.abs(alpha*dx)) < eps:
            return x

    raise RuntimeError("Newton did not converge within max iterations.")

def rkheunPoisson_full(phiddot, hdot, phi_ini, h_ini, t_ini, t_fin, point, Rs):
    """
    Proper Heun on the 1st-order system:
      phi'    = phidot
      phidot' = phiddot(...)
      h'      = hdot(...)
    Vectorized via broadcasting.
    """
    step = (t_fin - t_ini) / point
    t = np.linspace(t_ini, t_fin, point+1)

    Rs_arr = np.asarray(Rs, dtype=float)
    phi0 = np.asarray(phi_ini[0], dtype=float)
    phidot0 = np.asarray(phi_ini[1], dtype=float)
    h0 = np.asarray(h_ini, dtype=float)

    phi0, phidot0, h0, Rs_arr = np.broadcast_arrays(phi0, phidot0, h0, Rs_arr)
    batch_shape = phi0.shape

    phi = np.empty((point+1,)+batch_shape, dtype=float)
    phidot = np.empty((point+1,)+batch_shape, dtype=float)
    h = np.empty((point+1,)+batch_shape, dtype=float)

    phi[0] = phi0
    phidot[0] = phidot0
    h[0] = h0

    for i in range(point):
        # k1 at (t_i, y_i)
        k1_phi    = phidot[i]
        k1_phidot = phiddot(t[i], phi[i], phidot[i], h[i], Rs_arr)
        k1_h      = hdot(t[i], phi[i], phidot[i], h[i], Rs_arr)

        # predictor (Euler)
        phi_p    = phi[i]    + step*k1_phi
        phidot_p = phidot[i] + step*k1_phidot
        h_p      = h[i]      + step*k1_h

        # k2 at (t_{i+1}, y_pred)
        k2_phi    = phidot_p
        k2_phidot = phiddot(t[i+1], phi_p, phidot_p, h_p, Rs_arr)
        k2_h      = hdot(t[i+1], phi_p, phidot_p, h_p, Rs_arr)

        # corrector (average slopes)
        phi[i+1]    = phi[i]    + 0.5*step*(k1_phi + k2_phi)
        phidot[i+1] = phidot[i] + 0.5*step*(k1_phidot + k2_phidot)
        h[i+1]      = h[i]      + 0.5*step*(k1_h + k2_h)

    return t, phi, phidot, h

def rkheunPoisson_end(phiddot, hdot, phi_ini, h_ini, t_ini, t_fin, point, Rs):
    """
    Same as full, but returns only end values.
    """
    step = (t_fin - t_ini) / point
    t = np.linspace(t_ini, t_fin, point+1)

    Rs_arr = np.asarray(Rs, dtype=float)
    phi = np.asarray(phi_ini[0], dtype=float)
    phidot = np.asarray(phi_ini[1], dtype=float)
    h = np.asarray(h_ini, dtype=float)

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

# --- Regularized center start (series) to avoid the 2/r singular sensitivity
# From spherical Poisson analysis: near center, phi'(r) ~ (4π/3) rho_c Rs^2 * r
# and phi(r) ~ (2π/3) rho_c Rs^2 * r^2.  (same logic as uniform sphere potential)
# This stabilizes integration starting at eps.
eps_center = 1e-6

def bound_inter(Rs):
    Rs_arr = np.asarray(Rs, dtype=float)

    a = (4.0*np.pi/3.0) * (Rs_arr**2) * rhoc  # phi'(rhat) ~ a * rhat near 0

    phi0    = 0.5 * a * (eps_center**2)
    phidot0 = a * eps_center
    h0      = h_c - 0.5 * a * (eps_center**2)

    phi, phidot = rkheunPoisson_end(phiddot, hdot, [phi0, phidot0], h0, eps_center, 0.5, 500, Rs_arr)
    return phi, phidot

def bound_outer(Rs, Ms):
    # surface BC: phi(1)=-M/Rs, phidot(1)=+M/Rs  (vacuum phi = -M/(Rs rhat))
    phi, phidot = rkheunPoisson_end(phiddot, hdot, [-Ms/Rs, Ms/Rs], 0.0, 1.0, 0.5, 500, Rs)
    return phi, phidot

def func(x):
    x = np.asarray(x, dtype=float)
    Rs, Ms = x[0], x[1]

    phi_inter, phidot_inter = bound_inter(Rs)
    phi_outer, phidot_outer = bound_outer(Rs, Ms)

    return np.stack([phi_inter - phi_outer, phidot_inter - phidot_outer], axis=0)

def shooting(Rs0, Ms0):
    # mild guard: require positive initial guesses
    if Rs0 <= 0 or Ms0 <= 0:
        raise ValueError("Initial guesses Rs0, Ms0 must be positive.")
    Rs, Ms = nrmethod2D(func, np.array([Rs0, Ms0], dtype=float), 1e-8)
    return Rs, Ms

# ---- run ----
time1 = time.time()
print(shooting(1.0, 1.0))
time2 = time.time()
print("time = ", time2-time1)