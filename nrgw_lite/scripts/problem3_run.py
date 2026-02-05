from __future__ import annotations

import argparse
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..models.newton_osc_nonradial_l2 import solve_nonradial_l2_modes, write_problem3_dat


def main() -> None:
    ap = argparse.ArgumentParser(description="NRGW Lite - Problem 3 (nonradial l=2 Cowling oscillations)")
    ap.add_argument("--K", type=float, default=3.0)
    ap.add_argument("--n", type=float, default=math.sqrt(3.0))
    ap.add_argument("--rhoc", type=float, default=1.28e-3)
    ap.add_argument("--omega-min", type=float, default=1e-3)
    ap.add_argument("--omega-max", type=float, default=0.2)
    ap.add_argument("--scan-N", type=int, default=350)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    modes = solve_nonradial_l2_modes(
        K=args.K,
        n=args.n,
        rhoc=args.rhoc,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        n_scan=args.scan_N,
        max_modes=args.modes,
    )

    if not modes:
        raise RuntimeError("No modes found. Try widening the omega scan range or increasing scan-N.")

    bg = modes[0].bg
    print(f"[background] K={bg.K} n={bg.n} rhoc={bg.rhoc}  Rs={bg.Rs:.12g}  M={bg.M:.12g}")
    for m in modes:
        print(f"[mode] nodes={m.nodes}  omega={m.omega:.12g}")

    for m in modes:
        fn = os.path.join(args.outdir, f"problem3_mode{m.nodes}.dat")
        write_problem3_dat(fn, m)

    # conventional "problem3.dat" as 0-node mode if present
    fundamental = None
    for m in modes:
        if m.nodes == 0:
            fundamental = m
            break
    if fundamental is not None:
        write_problem3_dat(os.path.join(args.outdir, "problem3.dat"), fundamental)

    # Plot xihat
    plt.figure()
    for m in modes:
        plt.plot(m.rhat, m.xihat, label=f"nodes={m.nodes}, ω={m.omega:.4g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\hat \xi$ (l=2)")
    plt.legend()
    plt.title("Problem 3: nonradial l=2 modes (Cowling)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "problem3_xihat.png"), dpi=200)
    plt.close()

    # Plot delta_h_hat
    plt.figure()
    for m in modes:
        plt.plot(m.rhat, m.delta_h_hat, label=f"nodes={m.nodes}, ω={m.omega:.4g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\delta \hat h$")
    plt.legend()
    plt.title("Problem 3: nonradial l=2 modes (Cowling)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "problem3_delta_h_hat.png"), dpi=200)
    plt.close()

    print(f"[saved] dat + plots in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
