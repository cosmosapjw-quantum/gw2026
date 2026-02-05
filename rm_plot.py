import os
import numpy as np
import matplotlib.pyplot as plt

from numerical import Numerical
from equations import K, n


# 해당 함수는 안정적인 초기 추정값을 탐색하기 위해 AI에게 생성을 부탁함
def initial_guess(solver, rhoc0):

    if abs(n - 1.0) < 1e-12:
        Rs0 = float(np.sqrt(np.pi * K / 2.0))
    else:
        Rs0 = 10.0

    # crude mass scale (order-of-magnitude)
    Ms0 = float((4.0 * np.pi / 3.0) * rhoc0 * (Rs0**3))
    Ms0 = max(Ms0, 1e-12)

    return Rs0, Ms0


def rm_generator():
    solver = Numerical()

    R_list = []
    M_list = []
    rhoc_list = []
    total_list=[]

    rhocs = np.logspace(-5, -1, 100, base=10)
    Rs0, Ms0 = initial_guess(solver, float(rhocs[0]))

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
            Rs, Ms, res, ok = solver.shooting(
                rhoc, Rs0, Ms0,
                point_prune=solver.point_prune,
                point_main=max(solver.point_main, 250),
                tol_main=1e-10,
                maxiter_main=35,
                n_retry=2,
                n_seeds=256, prune_topk=12, sigma=0.9,
                parallel=False
            )
            if (not ok) or (not np.isfinite(res)):
                print(f"[FAIL] rhoc={rhoc:.3e}, residual={res:.3e}")
                break

        R_list.append(np.round(Rs,5))
        M_list.append(np.round(Ms,5))
        rhoc_list.append(rhoc)
        total_list.append([rhoc, Rs, Ms])

        # continuation warm-start (초기값 갱신으로 더 빠른 뉴턴-랩슨 수렴을 위해 AI에게 추천받은 부분)
        Rs0, Ms0 = Rs, Ms

    return np.asarray(R_list, dtype=float), np.asarray(M_list, dtype=float)

if __name__ == "__main__":
    Rlist, Mlist = rm_generator()

    plt.plot(Rlist, Mlist)
    ax = plt.gca()
    ax.ticklabel_format(style='plain', useOffset=False, axis='x')
    plt.xlabel("Rs")
    plt.ylabel("Ms")
    plt.title("Newtonian Polytrope 예제")
    plt.grid(True)
    plt.show()
