import os
import numpy as np
import matplotlib.pyplot as plt

from numerical_tools import ShootingIVP, HeunIVPConfig, HeunShootingSolver


# -----------------------------
# Model parameters (toy polytrope)
# -----------------------------
K = 50.0
n = 1.0

EPS_CENTER = 0.0
R_MATCH = 0.5

# -----------------------------
# EOS: rho(h)  (Newtonian toy)
# -----------------------------
def rho_from_h(h, delta_h=0.0):
    # optional smoothing near h=0 for FD stability
    h = np.asarray(h, dtype=float)
    if delta_h > 0.0:
        h_pos = 0.5 * (h + np.sqrt(h*h + delta_h*delta_h))
    else:
        h_pos = np.maximum(h, 0.0)
    return (h_pos / (K * (n + 1.0)))**n


def central_enthalpy(rhoc):
    return K * (n + 1.0) * rhoc**(1.0 / n)


# -----------------------------
# Coupled (inner+outer) IVP on s in [0,1]
# State y = [phi_i, phidot_i, h_i,  phi_o, phidot_o, h_o,  Rs_const]
# rhat_i(s) = eps + (R_MATCH-eps)*s
# rhat_o(s) = 1   + (R_MATCH-1)*s  (decreases)
# -----------------------------
def make_problem_newton(rhoc, *, eps=EPS_CENTER, r_match=R_MATCH):
    rhoc = float(rhoc)
    h_c = float(central_enthalpy(rhoc))
    delta_h = 1e-12 * max(1.0, abs(h_c))

    dr_i = (r_match - eps)
    dr_o = (r_match - 1.0)

    def fun(s, y):
        # y is always (7, batch) inside rk2_ivp
        phi_i    = y[0]
        phidot_i = y[1]
        h_i      = y[2]
        phi_o    = y[3]
        phidot_o = y[4]
        h_o      = y[5]
        Rs       = y[6]

        rhat_i = eps + dr_i * float(s)
        rhat_o = 1.0 + dr_o * float(s)

        rho_i = rho_from_h(h_i, delta_h=delta_h)
        rho_o = rho_from_h(h_o, delta_h=delta_h)

        # Poisson in rhat: phidot' = -(2/rhat) phidot + 4π Rs^2 rho
        # For very small rhat, use removable-singularity limit (regular center)
        if rhat_i < 1e-9:
            phiddot_i = (4.0*np.pi/3.0) * (Rs**2) * rho_i
        else:
            phiddot_i = -(2.0/rhat_i) * phidot_i + 4.0*np.pi * (Rs**2) * rho_i

        phiddot_o = -(2.0/rhat_o) * phidot_o + 4.0*np.pi * (Rs**2) * rho_o

        # Convert drhat -> ds
        dphi_i    = dr_i * phidot_i
        dphidot_i = dr_i * phiddot_i
        dh_i      = dr_i * (-phidot_i)

        dphi_o    = dr_o * phidot_o
        dphidot_o = dr_o * phiddot_o
        dh_o      = dr_o * (-phidot_o)

        dRs = np.zeros_like(Rs)

        return np.vstack([dphi_i, dphidot_i, dh_i, dphi_o, dphidot_o, dh_o, dRs])

    def y0_from_params(p):
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            Rs, Ms = np.abs(p[0]), np.abs(p[1])

            a = (4.0*np.pi/3.0) * (Rs**2) * rhoc  # phi''(0) in rhat variable
            phi_i0    = 0.5 * a * (eps**2)
            phidot_i0 = a * eps
            h_i0      = h_c - phi_i0

            phi_o0    = -Ms / Rs
            phidot_o0 =  Ms / Rs
            h_o0      = 0.0

            return np.array([phi_i0, phidot_i0, h_i0, phi_o0, phidot_o0, h_o0, Rs], dtype=float)

        # batch: p shape (2,k)
        Rs = np.abs(p[0])
        Ms = np.abs(p[1])

        a = (4.0*np.pi/3.0) * (Rs**2) * rhoc
        phi_i0    = 0.5 * a * (eps**2)
        phidot_i0 = a * eps
        h_i0      = h_c - phi_i0

        phi_o0    = -Ms / Rs
        phidot_o0 =  Ms / Rs
        h_o0      = np.zeros_like(Rs)

        return np.vstack([phi_i0, phidot_i0, h_i0, phi_o0, phidot_o0, h_o0, Rs])

    def residual_from_terminal(t_end, y_end, p):
        # want continuity at match: h_i = h_o and phidot_i = phidot_o
        if y_end.ndim == 1:
            h_i = y_end[2]
            h_o = y_end[5]
            dp_i = y_end[1]
            dp_o = y_end[4]
            return np.array([h_i - h_o, dp_i - dp_o], dtype=float)

        h_i = y_end[2]
        h_o = y_end[5]
        dp_i = y_end[1]
        dp_o = y_end[4]
        return np.vstack([h_i - h_o, dp_i - dp_o])

    prob = ShootingIVP(
        fun=fun,
        t_span=(0.0, 1.0),
        y0_from_params=y0_from_params,
        residual_from_terminal=residual_from_terminal,
        vectorized=True,
        coarse=HeunIVPConfig(adaptive=False, n_steps=50),
        fine=HeunIVPConfig(adaptive=False, n_steps=500),
    )

    def project(p):
        p = np.asarray(p, dtype=float).copy()
        p[0] = max(abs(p[0]), 1e-12)  # Rs > 0
        p[1] = max(abs(p[1]), 1e-12)  # Ms > 0
        return p

    solver = HeunShootingSolver(prob, project=project)
    return solver


def load_or_init_guess(rhoc0):
    # reasonable first guess
    # For Newtonian n=1 polytrope, radius tends to be ~ constant (scale ~ sqrt(pi*K/2) in G=1 units).
    Rs0 = float(np.sqrt(np.pi * K / 2.0))
    Ms0 = float((4.0*np.pi/3.0) * rhoc0 * (Rs0**3))
    Ms0 = max(Ms0, 1e-12)
    return Rs0, Ms0


# --- 단위 변환 상수 (G=c=M_sun=1) ---
# 1 length unit = GM_sun/c^2 ≈ 1.4766 km
LUNIT_KM = 1.47662512  # km per (M_sun) in geometrized units

def rm_generator():
    rhocs = np.logspace(-5, -1, 100, base=10)
    Rs0, Ms0 = load_or_init_guess(float(rhocs[0]))

    R_list, M_list = [], []

    for rhoc in rhocs:
        solver = make_problem_newton(rhoc)
        out = solver.solve([Rs0, Ms0], tol=1e-10, xtol=1e-10, maxiter=30,
                           rel_step=1e-6, n_retry=2, n_seeds=128, prune_topk=10, sigma=0.2)

        if (not out.success) or (not np.isfinite(out.residual_norm)):
            print(f"[FAIL] rhoc={rhoc:.3e}, ||F||={out.residual_norm:.3e}")
            break

        Rs, Ms = float(out.x[0]), float(out.x[1])

        # ---- NEW: compactness / horizon guard ----
        if Rs <= 2.0 * Ms:
            print(f"[STOP] hit Rs<=2Ms at rhoc={rhoc:.3e}: Rs={Rs:.6g}, Ms={Ms:.6g} (M/Msun={Ms:.6g})")
            break

        R_list.append(Rs)
        M_list.append(Ms)

        # continuation warm-start
        Rs0, Ms0 = Rs, Ms

    R_arr = np.asarray(R_list, dtype=float)
    M_arr = np.asarray(M_list, dtype=float)

    # ---- NEW: physical-unit arrays ----
    R_km = R_arr * LUNIT_KM
    M_msun = M_arr  # already in M/Msun

    return R_arr, M_arr, R_km, M_msun


if __name__ == "__main__":
    R_arr, M_arr, R_km, M_msun = rm_generator()

    plt.plot(R_km, M_msun)
    plt.xlabel("R [km]")
    plt.ylabel("M [M_sun]")
    plt.title("Newtonian toy (stop at Rs<=2Ms)")
    plt.grid(True)
    plt.show()
