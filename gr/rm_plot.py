# 뉴턴 버전과 차이가 거의 없음. 다만 바뀐 TOV에 해당하여 AI에게 조정을 요청함

import os
import numpy as np
import matplotlib.pyplot as plt
from numerical import Numerical
from equations import K

def initial_guess(solver, rhoc0):

    Rs0 = float(np.sqrt(np.pi * K / 2.0))
    h_c = solver.central_entalphy(float(rhoc0))
    eps_c = float(solver.eps_from_h(h_c))
    Ms0 = float((4.0*np.pi/3.0) * eps_c * (Rs0**3))
    Ms0 = max(Ms0, 1e-12)
    return Rs0, Ms0

def rm_generator():
    solver = Numerical()

    R_list = []
    M_list = []

    rhocs = np.logspace(-9, -1, 100, base=10)
    Rs0, Ms0 = initial_guess(solver, rhocs[0])

    for rhoc in rhocs:
        print("rhoc = ", rhoc)

        Rs, Ms, res, ok = solver.shooting(
            rhoc, Rs0, Ms0,
            point_prune=solver.point_prune,
            point_main=solver.point_main,
            tol_main=1e-10,
            maxiter_main=25,
            n_retry=1,
            n_seeds=128, prune_topk=10, sigma=0.6,
            parallel=False
        )

        if (not ok) or (not np.isfinite(res)):
            # more aggressive retry
            Rs, Ms, res, ok = solver.shooting(
                rhoc, Rs0, Ms0,
                point_prune=solver.point_prune,
                point_main=max(solver.point_main, 200),
                tol_main=1e-10,
                maxiter_main=35,
                n_retry=2,
                n_seeds=256, prune_topk=12, sigma=0.9,
                parallel=False
            )
            if (not ok) or (not np.isfinite(res)):
                print(f"[FAIL] rhoc={rhoc:.3e}, residual={res:.3e}")
                break

        # physical guard: outside horizon
        if Rs <= 2.0 * Ms:
            print(f"[STOP] hit compactness limit: Rs<=2Ms at rhoc={rhoc:.3e}")
            break

        R_list.append(Rs)
        M_list.append(Ms)

        # continuation warm-start (THIS is correct and should be kept)
        Rs0, Ms0 = Rs, Ms

    return np.asarray(R_list, dtype=float), np.asarray(M_list, dtype=float)

if __name__ == "__main__":
    Rlist, Mlist = rm_generator()

    plt.plot(Rlist, Mlist)
    ax = plt.gca()
    ax.ticklabel_format(style='plain', useOffset=False, axis='x')
    plt.xlabel("Rs")
    plt.ylabel("Ms")
    plt.title("TOV Mass-Radius")
    plt.grid(True)
    plt.show()
