import numpy as np
import matplotlib.pyplot as plt
from numerical import Numerical

def rm_generator():
    solver = Numerical()

    R_list = []
    M_list = []

    # continuation warm-start
    Rs0, Ms0 = 10.0, 0.1

    for rhoc in np.logspace(-9, -1, 100, base=10):
        print("rhoc = ", rhoc)

        Rs, Ms, res, ok = solver.shooting(
            rhoc, Rs0, Ms0,
            point_prune=solver.point_prune,
            point_main=solver.point_main,
            tol_main=1e-10,
            maxiter_main=25,
            n_retry=1,
            # multi-start knobs (only used on failure)
            n_seeds=128, prune_topk=10, sigma=0.6,
            parallel=False
        )

        # hard fail guard
        if (not ok) or (not np.isfinite(res)):
            # one more aggressive retry
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

        # physical compactness guard
        if Rs <= 2.0 * Ms:
            break

        R_list.append(Rs)
        M_list.append(Ms)

        # continuation
        Rs0, Ms0 = Rs, Ms

    return np.asarray(R_list, dtype=float), np.asarray(M_list, dtype=float)

if __name__ == "__main__":
    Rlist, Mlist = rm_generator()
    print("Results:", Rlist, Mlist)

    plt.plot(Rlist, Mlist)
    plt.xlabel("Rs")
    plt.ylabel("Ms")
    plt.title("Mass-Radius Relation")
    plt.grid(True)
    plt.show()