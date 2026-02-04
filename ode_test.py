# coupled_sho_example.py
import numpy as np
import matplotlib.pyplot as plt

# rk2_ivp.py가 같은 폴더에 있다고 가정
from rk2_ivp import solve_ivp_rk2_heun


def coupled_sho_rhs_factory(m=1.0, k=1.0, kc=0.2):
    """
    Two coupled oscillators:
      x1'' = -((k+kc)/m) x1 + (kc/m) x2
      x2'' =  (kc/m) x1 -((k+kc)/m) x2
    State y = [x1, v1, x2, v2]
    """
    a = (k + kc) / m
    b = kc / m

    def f(t, y):
        x1, v1, x2, v2 = y
        dx1 = v1
        dv1 = -a * x1 + b * x2
        dx2 = v2
        dv2 =  b * x1 - a * x2
        return np.array([dx1, dv1, dx2, dv2], dtype=float)

    return f


def total_energy(y, m=1.0, k=1.0, kc=0.2):
    """
    E = T + V
    T = 1/2 m (v1^2 + v2^2)
    V = 1/2 k (x1^2 + x2^2) + 1/2 kc (x1 - x2)^2
    """
    x1, v1, x2, v2 = y
    T = 0.5 * m * (v1*v1 + v2*v2)
    V = 0.5 * k * (x1*x1 + x2*x2) + 0.5 * kc * (x1 - x2)**2
    return T + V


if __name__ == "__main__":
    # parameters
    m = 1.0
    k = 1.0
    kc = 0.2

    f = coupled_sho_rhs_factory(m=m, k=k, kc=kc)

    # initial condition: x1 displaced, x2 at rest, both velocities zero
    y0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # integrate
    t0, tf = 0.0, 200.0
    dt = 0.01
    t_eval = np.arange(t0, tf + 0.5*dt, dt)

    res = solve_ivp_rk2_heun(
        f,
        (t0, tf),
        y0,
        t_eval=t_eval,
        adaptive=False,      # 실습 목적: "고정 스텝" Heun(RK2)
        first_step=dt,
        max_step=dt,
        dense_output=False,
    )

    if not res.success:
        print("Integration failed:", res.message)
        raise SystemExit(1)

    # unpack
    t = res.t
    y = res.y  # shape (4, len(t))
    x1, v1, x2, v2 = y

    # energy drift check (RK2 고정스텝이면 장시간에 조금씩 드리프트는 있을 수 있음)
    E = np.array([total_energy(y[:, i], m=m, k=k, kc=kc) for i in range(y.shape[1])])
    E0 = E[0]
    rel_drift = (E - E0) / max(1e-15, abs(E0))
    print("nfev:", res.nfev)
    print("final relative energy drift:", rel_drift[-1])

    # plot
    plt.figure()
    plt.plot(t, x1, label="x1(t)")
    plt.plot(t, x2, label="x2(t)")
    plt.xlabel("t")
    plt.ylabel("displacement")
    plt.title("Coupled SHO solved by custom Heun (RK2)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, rel_drift)
    plt.xlabel("t")
    plt.ylabel("(E - E0)/E0")
    plt.title("Relative energy drift (sanity check)")
    plt.grid(True)

    plt.show()
