import os
import numpy as np
import matplotlib.pyplot as plt

from numerical_tools import ShootingIVP, HeunIVPConfig, HeunShootingSolver


# -----------------------------
# Model parameters (polytrope in natural units G=c=1)
# -----------------------------
K = 100.0
n = 1.0

EPS_CENTER = 1e-9
R_MATCH = 0.5

# -----------------------------
# EOS in terms of "h" (same convention as your earlier TOV code)
# rho0(h) = (h/(K(n+1)))^n,  P = rho0*h/(n+1),  epsilon = rho0 + n P
# (This matches the common relativistic polytrope practice: P=K rho0^Gamma plus internal energy term.)
# -----------------------------
def rho0_from_h(h, delta_h=0.0):
    h = np.asarray(h, dtype=float)
    if delta_h > 0.0:
        h_pos = 0.5 * (h + np.sqrt(h*h + delta_h*delta_h))
    else:
        h_pos = np.maximum(h, 0.0)
    return (h_pos / (K * (n + 1.0)))**n


def P_from_h(h, delta_h=0.0):
    h = np.asarray(h, dtype=float)
    h_pos = np.maximum(h, 0.0)
    rho0 = rho0_from_h(h_pos, delta_h=delta_h)
    return rho0 * h_pos / (n + 1.0)


def eps_from_h(h, delta_h=0.0):
    h = np.asarray(h, dtype=float)
    h_pos = np.maximum(h, 0.0)
    rho0 = rho0_from_h(h_pos, delta_h=delta_h)
    P = P_from_h(h_pos, delta_h=delta_h)
    return rho0 + n * P


def central_enthalpy(rhoc):
    return K * (n + 1.0) * rhoc**(1.0 / n)


# -----------------------------
# Coupled (inner+outer) IVP on s in [0,1]
# State y = [m_i, h_i,  m_o, h_o,  Rs_const]
# rhat_i(s) = eps + (R_MATCH-eps)*s
# rhat_o(s) = 1   + (R_MATCH-1)*s
#
# TOV (G=c=1):
#   dm/dr = 4π r^2 ε
#   dP/dr = -(ε+P)(m+4π r^3 P) / (r(r-2m))
# With this enthalpy-like variable, you can use:
#   dh/dr = -(1+h) (m+4π r^3 P) / (r(r-2m))
# -----------------------------
def make_problem_tov(rhoc, *, eps=EPS_CENTER, r_match=R_MATCH):
    rhoc = float(rhoc)
    h_c = float(central_enthalpy(rhoc))
    delta_h = 1e-12 * max(1.0, abs(h_c))

    dr_i = (r_match - eps)
    dr_o = (r_match - 1.0)

    def dm_drhat(rhat, m, h, Rs):
        # dm/drhat = Rs * dm/dr = 4π Rs^3 rhat^2 ε
        eps_e = eps_from_h(h, delta_h=delta_h)
        return 4.0*np.pi * (Rs**3) * (float(rhat)**2) * eps_e

    def dh_drhat(rhat, m, h, Rs):
        rhat = float(rhat)
        h_pos = np.maximum(h, 0.0)
        P = P_from_h(h_pos, delta_h=delta_h)

        r = Rs * rhat
        denom = r * (r - 2.0*m)
        num = m + 4.0*np.pi * (r**3) * P

        bad = (r <= 0.0) | (denom <= 0.0)
        out = -Rs * (1.0 + h_pos) * num / denom
        out = np.where(h <= 0.0, 0.0, out)
        out = np.where(bad, np.nan, out)
        return out

    def fun(s, y):
        m_i = y[0]
        h_i = y[1]
        m_o = y[2]
        h_o = y[3]
        Rs  = y[4]

        rhat_i = eps + dr_i * float(s)
        rhat_o = 1.0 + dr_o * float(s)

        dm_i = dm_drhat(rhat_i, m_i, h_i, Rs)
        dh_i = dh_drhat(rhat_i, m_i, h_i, Rs)

        dm_o = dm_drhat(rhat_o, m_o, h_o, Rs)
        dh_o = dh_drhat(rhat_o, m_o, h_o, Rs)

        # d/ds = (drhat/ds) d/drhat
        dm_i_ds = dr_i * dm_i
        dh_i_ds = dr_i * dh_i

        dm_o_ds = dr_o * dm_o
        dh_o_ds = dr_o * dh_o

        dRs = np.zeros_like(Rs)

        return np.vstack([dm_i_ds, dh_i_ds, dm_o_ds, dh_o_ds, dRs])

    def y0_from_params(p):
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            Rs, Ms = np.abs(p[0]), np.abs(p[1])

            eps_c = float(eps_from_h(h_c, delta_h=delta_h))
            r0 = Rs * eps
            m_i0 = (4.0*np.pi/3.0) * eps_c * (r0**3)
            h_i0 = h_c

            m_o0 = Ms
            h_o0 = 0.0

            return np.array([m_i0, h_i0, m_o0, h_o0, Rs], dtype=float)

        Rs = np.abs(p[0])
        Ms = np.abs(p[1])

        eps_c = float(eps_from_h(h_c, delta_h=delta_h))
        r0 = Rs * eps
        m_i0 = (4.0*np.pi/3.0) * eps_c * (r0**3)
        h_i0 = np.full_like(Rs, h_c)

        m_o0 = Ms
        h_o0 = np.zeros_like(Rs)

        return np.vstack([m_i0, h_i0, m_o0, h_o0, Rs])

    def residual_from_terminal(t_end, y_end, p):
        # want continuity at match: m_i = m_o and h_i = h_o
        if y_end.ndim == 1:
            return np.array([y_end[0] - y_end[2], y_end[1] - y_end[3]], dtype=float)
        return np.vstack([y_end[0] - y_end[2], y_end[1] - y_end[3]])

    prob = ShootingIVP(
        fun=fun,
        t_span=(0.0, 1.0),
        y0_from_params=y0_from_params,
        residual_from_terminal=residual_from_terminal,
        vectorized=True,
        coarse=HeunIVPConfig(adaptive=False, n_steps=50),
        fine=HeunIVPConfig(adaptive=False, n_steps=100),  # TOV는 보통 조금 더 빡세게
    )

    def project(p):
        p = np.asarray(p, dtype=float).copy()
        Rs = max(abs(p[0]), 1e-12)
        Ms = max(abs(p[1]), 1e-12)

        # keep away from horizon: Rs > 2Ms*(1+tiny)
        if Rs <= 2.0 * Ms * (1.0 + 1e-6):
            Rs = 2.0 * Ms * (1.0 + 1e-6)

        p[0], p[1] = Rs, Ms
        return p

    solver = HeunShootingSolver(prob, project=project)
    return solver


def load_or_init_guess(rhoc0):
 # fallback guess: start from Newtonian-ish scale + mass from central epsilon
    Rs0 = float(np.sqrt(np.pi * K / 2.0))

    h_c0 = float(central_enthalpy(rhoc0))
    eps_c0 = float(eps_from_h(h_c0, delta_h=1e-12 * max(1.0, abs(h_c0))))
    Ms0 = float((4.0*np.pi/3.0) * eps_c0 * (Rs0**3))

    # enforce Rs>2Ms
    Ms0 = min(Ms0, 0.4 * Rs0)
    Ms0 = max(Ms0, 1e-12)
    return Rs0, Ms0


def rm_generator():
    # TOV는 너무 넓은 rhoc 스캔에서 쉽게 horizon 근처로 가니, 일단 안전 범위 예시
    rhocs = np.logspace(-6, -1, 50, base=10)

    Rs0, Ms0 = load_or_init_guess(float(rhocs[0]))

    R_list, M_list = [], []

    for rhoc in rhocs:
        print("rhoc = ",rhoc)
        solver = make_problem_tov(rhoc)

        out = solver.solve(
            [Rs0, Ms0],
            tol=1e-10,
            xtol=1e-10,
            maxiter=30,
            rel_step=1e-4,
            n_retry=2,
            n_seeds=196,
            prune_topk=10,
            sigma=0.6,
        )

        if (not out.success) or (not np.isfinite(out.residual_norm)):
            print(f"[FAIL] rhoc={rhoc:.3e}, ||F||={out.residual_norm:.3e}")
            break

        Rs, Ms = float(out.x[0]), float(out.x[1])

        if Rs <= 2.0 * Ms:
            print(f"[STOP] hit horizon guard at rhoc={rhoc:.3e}: Rs<=2Ms")
            break

        R_list.append(Rs)
        M_list.append(Ms)

        # continuation warm-start
        Rs0, Ms0 = Rs, Ms

    return np.asarray(R_list), np.asarray(M_list)


if __name__ == "__main__":
    R, M = rm_generator()
    plt.plot(R, M)
    ax = plt.gca()
    ax.ticklabel_format(style="plain", useOffset=False, axis="x")
    plt.xlabel("Rs")
    plt.ylabel("Ms")
    plt.title("TOV Matching Shooting — Heun(RK2) + FD Newton")
    plt.grid(True)
    plt.show()
