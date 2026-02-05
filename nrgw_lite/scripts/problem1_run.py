from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Use a non-interactive backend (safe on headless machines)
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..models.newton_structure import solve_newton_structure_problem1, write_problem1_dat


def _safe_tag(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace(".", "p").replace("-", "m")


def run_single_case(outdir: Path, *, n: float, K: float, rhoc: float, dat_name: str = "problem1.dat") -> None:
    print(f"\n[Problem1-single] Solving n={n}, K={K}, rhoc={rhoc} -> writing {dat_name}")
    sol = solve_newton_structure_problem1(K=K, n=n, rhoc=rhoc)
    fn = outdir / dat_name
    write_problem1_dat(str(fn), sol)
    print(f"  -> Rs={sol.Rs:.6g}, M={sol.M:.6g}, residual_inf={sol.residual_inf:.3e}")
    print(f"  wrote: {fn}")

    # optional quick plot (rho profile)
    plt.figure()
    plt.plot(sol.rhat, sol.rho)
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\rho(\hat r)$")
    plt.title(f"Density profile (n={n:g}, K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    figpath = outdir / f"problem1_density_single_n{_safe_tag(n)}.png"
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved: {figpath}")


def run_density_profiles(outdir: Path) -> None:
    K = 1.0e2
    rhoc = 1.28e-3
    n_list = [0.8, 1.0, 1.5]

    sols = []

    print("\n[Problem1-2] Density profiles (K=100, rhoc=1.28e-3)")
    for n in n_list:
        print(f"  - solving n={n} ...")
        sol = solve_newton_structure_problem1(K=K, n=n, rhoc=rhoc)
        print(f"    -> Rs={sol.Rs:.6g}, M={sol.M:.6g}, residual_inf={sol.residual_inf:.3e}")
        sols.append(sol)

        # write per-n dat (avoid overwriting)
        fn = outdir / f"problem1_n{_safe_tag(n)}.dat"
        write_problem1_dat(str(fn), sol)
        print(f"    wrote: {fn}")

    # plot rho(rhat)
    plt.figure()
    for sol in sols:
        plt.plot(sol.rhat, sol.rho, label=f"n={sol.n:g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\rho(\hat r)$")
    plt.title("Newtonian polytrope density profiles (Problem 1)")
    plt.grid(True)
    plt.legend()
    figpath = outdir / "problem1_density_profiles.png"
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved: {figpath}")


def _sweep_one_eos(
    *,
    K: float,
    n: float,
    rhoc_grid: np.ndarray,
    outdir: Path,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rs_list: List[float] = []
    M_list: List[float] = []
    rh_list: List[float] = []

    # warm start
    unknown0 = None

    for i, rhoc in enumerate(rhoc_grid):
        print(f"    rhoc[{i+1}/{len(rhoc_grid)}] = {rhoc:.3e}")
        try:
            sol = solve_newton_structure_problem1(K=K, n=n, rhoc=float(rhoc), unknown0=unknown0)
        except Exception as e:
            print(f"    [FAIL] rhoc={rhoc:.3e}: {type(e).__name__}: {e}")
            break

        Rs, M = float(sol.Rs), float(sol.M)
        Rs_list.append(Rs)
        M_list.append(M)
        rh_list.append(float(rhoc))
        unknown0 = (Rs, M)  # continuation

        if Rs <= 2.0 * M:
            print(f"    [BREAK] Rs <= 2M reached: Rs={Rs:.6g}, 2M={2.0*M:.6g}")
            break

    Rs_arr = np.asarray(Rs_list, dtype=float)
    M_arr = np.asarray(M_list, dtype=float)
    rh_arr = np.asarray(rh_list, dtype=float)

    np.savez(outdir / f"problem1_mr_{label}.npz", rhoc=rh_arr, Rs=Rs_arr, M=M_arr)
    return rh_arr, Rs_arr, M_arr


def run_mass_radius_curves(
    outdir: Path,
    *,
    rhoc_min: float = 1e-5,
    rhoc_max: float = 1e-1,
    rhoc_steps: int = 60,
) -> None:
    eos = [
        (0.8, 500.0, "n0p8_K500"),
        (1.0, 50.0, "n1_K50"),
        (1.5, 5.0, "n1p5_K5"),
    ]

    rhoc_grid = np.logspace(np.log10(rhoc_min), np.log10(rhoc_max), int(rhoc_steps))

    print("\n[Problem1-3] Mass-Radius curves (vary rhoc until Rs<=2M)")
    curves = []
    for n, K, tag in eos:
        print(f"  - EOS {tag}: n={n}, K={K}")
        rh, Rs, M = _sweep_one_eos(K=K, n=n, rhoc_grid=rhoc_grid, outdir=outdir, label=tag)
        curves.append((n, K, tag, rh, Rs, M))

    # plot M-R (x=Rs, y=M)
    plt.figure()
    for n, K, tag, rh, Rs, M in curves:
        if Rs.size == 0:
            continue
        plt.plot(Rs, M, marker="o", markersize=2, linewidth=1.0, label=f"n={n:g}, K={K:g}")
    plt.xlabel(r"$R_s$")
    plt.ylabel(r"$M$")
    plt.title("Mass-Radius curves (Problem 1)")
    plt.grid(True)
    plt.legend()
    figpath = outdir / "problem1_mass_radius_curves.png"
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved: {figpath}")
    print(f"  (raw data saved as .npz per EOS in {outdir})")


def main() -> None:
    ap = argparse.ArgumentParser(description="NRGW-LITE Problem 1 runner (Newtonian structure).")
    ap.add_argument("--outdir", type=str, default="nrgw_lite_out", help="Output directory.")

    ap.add_argument("--single-n", type=float, default=None, help="Run a single (n,K,rhoc) case and write problem1.dat.")
    ap.add_argument("--single-K", type=float, default=1.0e2)
    ap.add_argument("--single-rhoc", type=float, default=1.28e-3)
    ap.add_argument("--single-dat", type=str, default="problem1.dat")

    ap.add_argument("--profiles", action="store_true", help="Run density profile cases (Problem1-2).")
    ap.add_argument("--mr", action="store_true", help="Run mass-radius sweeps (Problem1-3).")
    ap.add_argument("--rhoc-min", type=float, default=1e-5)
    ap.add_argument("--rhoc-max", type=float, default=1e-1)
    ap.add_argument("--rhoc-steps", type=int, default=60)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.single_n is not None:
        run_single_case(outdir, n=args.single_n, K=args.single_K, rhoc=args.single_rhoc, dat_name=args.single_dat)
        print("\nDone.")
        return

    if not args.profiles and not args.mr:
        # default: run both
        args.profiles = True
        args.mr = True

    if args.profiles:
        run_density_profiles(outdir)
    if args.mr:
        run_mass_radius_curves(outdir, rhoc_min=args.rhoc_min, rhoc_max=args.rhoc_max, rhoc_steps=args.rhoc_steps)

    print("\nDone.")


if __name__ == "__main__":
    main()
