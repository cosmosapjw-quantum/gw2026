from __future__ import annotations

import argparse
import math
import os
from typing import List

import numpy as np

# Non-interactive backend for batch runs
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..models.newton_osc_radial import solve_radial_modes, write_problem2_dat


def main() -> None:
    ap = argparse.ArgumentParser(description="NRGW Lite - Problem 2 (radial Cowling oscillations)")
    ap.add_argument("--K", type=float, default=3.0)
    ap.add_argument("--n", type=float, default=math.sqrt(3.0))
    ap.add_argument("--rhoc", type=float, default=1.28e-3)
    ap.add_argument("--omega-min", type=float, default=1e-3)
    ap.add_argument("--omega-max", type=float, default=0.2)
    ap.add_argument("--scan-N", type=int, default=300)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    modes = solve_radial_modes(
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

    # Write dat files
    for m in modes:
        fn = os.path.join(args.outdir, f"problem2_mode{m.nodes}.dat")
        write_problem2_dat(fn, m)

    # Also write the conventional "problem2.dat" as the fundamental (0-node) if present
    fundamental = None
    for m in modes:
        if m.nodes == 0:
            fundamental = m
            break
    if fundamental is not None:
        write_problem2_dat(os.path.join(args.outdir, "problem2.dat"), fundamental)

    # Plot xihat
    plt.figure()
    for m in modes:
        plt.plot(m.rhat, m.xihat, label=f"nodes={m.nodes}, ω={m.omega:.4g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\hat \xi$")
    plt.legend()
    plt.title("Problem 2: radial modes (Cowling)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "problem2_xihat.png"), dpi=200)
    plt.close()

    # Plot delta_h
    plt.figure()
    for m in modes:
        plt.plot(m.rhat, m.delta_h, label=f"nodes={m.nodes}, ω={m.omega:.4g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\delta h$")
    plt.legend()
    plt.title("Problem 2: radial modes (Cowling)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "problem2_delta_h.png"), dpi=200)
    plt.close()

    print(f"[saved] dat + plots in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
