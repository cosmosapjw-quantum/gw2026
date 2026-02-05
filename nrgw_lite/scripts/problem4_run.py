from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..models.gr_structure_metric import (
    GRMetricSolution,
    solve_gr_structure_problem4,
    write_problem4_dat,
)


def _safe_tag(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace(".", "p").replace("-", "m")


def run_single(outdir: Path, *, n: float, K: float, rhoc: float, dat_name: str = "problem4.dat") -> GRMetricSolution:
    print(f"\n[Problem4-single] Solving GR metric structure: n={n}, K={K}, rhoc={rhoc}")
    sol = solve_gr_structure_problem4(K=K, n=n, rhoc=rhoc)
    fn = outdir / dat_name
    write_problem4_dat(str(fn), sol)
    print(f"  -> Rs={sol.Rs:.6g}, M={sol.M:.6g}, Lambda_s={sol.Lambda_s:.6g}, Phi_c={sol.Phi_c:.6g}, res_inf={sol.residual_inf:.3e}")
    print(f"  wrote: {fn}")

    # quick plots for this single case
    plt.figure()
    plt.plot(sol.rhat, sol.Lambda)
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\Lambda(\hat r)$")
    plt.title(f"Lambda profile (n={n:g}, K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    p = outdir / f"problem4_Lambda_single_n{_safe_tag(n)}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(sol.rhat, sol.Phi)
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\Phi(\hat r)$")
    plt.title(f"Phi profile (n={n:g}, K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    p = outdir / f"problem4_Phi_single_n{_safe_tag(n)}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(sol.rhat, sol.rho)
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\rho(\hat r)$")
    plt.title(f"Density profile (n={n:g}, K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    p = outdir / f"problem4_rho_single_n{_safe_tag(n)}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  saved: problem4_*_single_n{_safe_tag(n)}.png")
    return sol


def run_profiles(outdir: Path, *, K: float = 1.0e2, rhoc: float = 1.28e-3, n_list: Sequence[float] = None) -> List[GRMetricSolution]:
    if n_list is None:
        n_list = [1.0 / np.sqrt(2.0), 1.0, np.sqrt(2.0)]

    sols: List[GRMetricSolution] = []

    print(f"\n[Problem4-2] Profiles (K={K:g}, rhoc={rhoc:g}) for n in {list(n_list)}")
    # Use continuation across n list? (not necessarily monotonic). We'll keep independent solves.
    for n in n_list:
        sol = solve_gr_structure_problem4(K=K, n=float(n), rhoc=float(rhoc))
        sols.append(sol)
        fn = outdir / f"problem4_n{_safe_tag(float(n))}.dat"
        write_problem4_dat(str(fn), sol)
        print(f"  n={float(n):g}: Rs={sol.Rs:.6g}, M={sol.M:.6g}, res_inf={sol.residual_inf:.3e}  -> {fn}")

    # overlay plots
    plt.figure()
    for sol in sols:
        plt.plot(sol.rhat, sol.Lambda, label=f"n={sol.n:g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\Lambda(\hat r)$")
    plt.title(f"Lambda profiles (K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    plt.legend()
    p = outdir / "problem4_profiles_Lambda.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    for sol in sols:
        plt.plot(sol.rhat, sol.Phi, label=f"n={sol.n:g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\Phi(\hat r)$")
    plt.title(f"Phi profiles (K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    plt.legend()
    p = outdir / "problem4_profiles_Phi.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    for sol in sols:
        plt.plot(sol.rhat, sol.rho, label=f"n={sol.n:g}")
    plt.xlabel(r"$\hat r$")
    plt.ylabel(r"$\rho(\hat r)$")
    plt.title(f"Density profiles (K={K:g}, rhoc={rhoc:g})")
    plt.grid(True)
    plt.legend()
    p = outdir / "problem4_profiles_rho.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    print("  saved overlay plots:")
    print("   - problem4_profiles_Lambda.png")
    print("   - problem4_profiles_Phi.png")
    print("   - problem4_profiles_rho.png")

    return sols


def _find_max_mass(rhoc: np.ndarray, M: np.ndarray) -> Tuple[float, float, int]:
    """Return (rhoc_at_max, Mmax, idx). Uses dM/drhoc sign change if possible."""
    rhoc = np.asarray(rhoc, float)
    M = np.asarray(M, float)
    if rhoc.size < 3:
        i = int(np.argmax(M))
        return float(rhoc[i]), float(M[i]), i

    dM = np.gradient(M, rhoc)
    # find first sign change + -> - near maximum
    for i in range(1, dM.size):
        if np.isfinite(dM[i - 1]) and np.isfinite(dM[i]) and (dM[i - 1] > 0.0) and (dM[i] < 0.0):
            # choose closer to zero derivative
            j = i - 1 if abs(dM[i - 1]) < abs(dM[i]) else i
            return float(rhoc[j]), float(M[j]), int(j)

    # fallback argmax
    i = int(np.argmax(M))
    return float(rhoc[i]), float(M[i]), i


def run_sweeps(outdir: Path, *, rhoc_min: float, rhoc_max: float, rhoc_steps: int) -> None:
    """Problem4-3/4: vary rhoc for 3 EOS sets and make curves + max mass."""
    eos_list = [
        (1.0 / np.sqrt(2.0), 1000.0, "n=1/sqrt(2),K=1000"),
        (1.0, 50.0, "n=1,K=50"),
        (np.sqrt(2.0), 5.0, "n=sqrt(2),K=5"),
    ]

    rhocs = np.logspace(np.log10(rhoc_min), np.log10(rhoc_max), int(rhoc_steps))
    print(f"\n[Problem4-3/4] Sweeps: rhoc in [{rhoc_min:g}, {rhoc_max:g}] ({rhoc_steps} points, logspace)")
    print("EOS sets:")
    for n, K, tag in eos_list:
        print(f"  - {tag}")

    all_data = []

    for n, K, tag in eos_list:
        print(f"\n[Sweep] {tag}")
        Rs_list: List[float] = []
        M_list: List[float] = []
        Ls_list: List[float] = []
        Phic_list: List[float] = []
        rhoc_list: List[float] = []

        # continuation warm-start with last successful root
        guess = None

        for rhoc in rhocs:
            print(f"  rhoc={rhoc:.3e} ...", end="")
            try:
                sol = solve_gr_structure_problem4(K=K, n=float(n), rhoc=float(rhoc), unknown0=guess)
            except Exception as e:
                print(f" FAIL ({type(e).__name__})")
                break

            # compactness safety (approach horizon)
            if sol.Rs <= 2.0 * sol.M:
                print(" STOP (Rs<=2M)")
                break

            rhoc_list.append(float(rhoc))
            Rs_list.append(sol.Rs)
            M_list.append(sol.M)
            Ls_list.append(sol.Lambda_s)
            Phic_list.append(sol.Phi_c)

            # update guess = (Rs, Lambda_s, Phi_c)
            guess = np.array([sol.Rs, sol.Lambda_s, sol.Phi_c], dtype=float)

            print(f" ok  Rs={sol.Rs:.4g} M={sol.M:.4g} res={sol.residual_inf:.1e}")

        rhoc_arr = np.asarray(rhoc_list, float)
        Rs_arr = np.asarray(Rs_list, float)
        M_arr = np.asarray(M_list, float)

        if rhoc_arr.size == 0:
            print(f"  [WARN] no data for {tag}")
            continue

        # find max mass
        rhoc_star, Mmax, imax = _find_max_mass(rhoc_arr, M_arr)
        print(f"  [MAX] {tag}: Mmax={Mmax:.6g} at rhoc≈{rhoc_star:.6g} (index {imax}/{rhoc_arr.size})")

        # save data
        npz_path = outdir / f"problem4_sweep_{_safe_tag(float(n))}_K{_safe_tag(float(K))}.npz"
        np.savez(
            npz_path,
            rhoc=rhoc_arr,
            Rs=Rs_arr,
            M=M_arr,
            n=float(n),
            K=float(K),
            rhoc_max_mass=float(rhoc_star),
            M_max=float(Mmax),
        )
        print(f"  saved raw: {npz_path}")

        all_data.append((tag, rhoc_arr, Rs_arr, M_arr))

    # plots: for each relationship, overlay EOS sets
    if not all_data:
        print("[WARN] no sweep data produced.")
        return

    # rhoc - M
    plt.figure()
    for tag, rhoc_arr, Rs_arr, M_arr in all_data:
        plt.plot(rhoc_arr, M_arr, label=tag)
    plt.xscale("log")
    plt.xlabel(r"$\rho_c$")
    plt.ylabel(r"$M$")
    plt.title("GR: central density vs mass")
    plt.grid(True)
    plt.legend()
    p = outdir / "problem4_rhoc_M.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    # rhoc - Rs
    plt.figure()
    for tag, rhoc_arr, Rs_arr, M_arr in all_data:
        plt.plot(rhoc_arr, Rs_arr, label=tag)
    plt.xscale("log")
    plt.xlabel(r"$\rho_c$")
    plt.ylabel(r"$R_s$")
    plt.title("GR: central density vs radius")
    plt.grid(True)
    plt.legend()
    p = outdir / "problem4_rhoc_Rs.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    # M - Rs
    plt.figure()
    for tag, rhoc_arr, Rs_arr, M_arr in all_data:
        plt.plot(Rs_arr, M_arr, label=tag)
    plt.xlabel(r"$R_s$")
    plt.ylabel(r"$M$")
    plt.title("GR: mass-radius curve")
    plt.grid(True)
    plt.legend()
    p = outdir / "problem4_M_Rs.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved sweep plots:")
    print("  - problem4_rhoc_M.png")
    print("  - problem4_rhoc_Rs.png")
    print("  - problem4_M_Rs.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="NRGW_LITE Problem 4 (GR metric form) runner")
    ap.add_argument("--outdir", type=str, default="nrgw_lite_out", help="output directory (created if missing)")
    ap.add_argument("--single", action="store_true", help="run a single model and write problem4.dat")
    ap.add_argument("--profiles", action="store_true", help="run default 3 n values and make overlay profile plots")
    ap.add_argument("--sweeps", action="store_true", help="run the rhoc sweeps for Problem 4-3/4")

    ap.add_argument("--n", type=float, default=1.0, help="single-run polytropic index n")
    ap.add_argument("--K", type=float, default=1.0e2, help="single-run polytropic constant K")
    ap.add_argument("--rhoc", type=float, default=1.28e-3, help="single-run central density rhoc")

    ap.add_argument("--rhoc-min", type=float, default=1e-8, help="sweep: minimum rhoc")
    ap.add_argument("--rhoc-max", type=float, default=1e-1, help="sweep: maximum rhoc")
    ap.add_argument("--rhoc-steps", type=int, default=80, help="sweep: number of rhoc points (logspace)")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not (args.single or args.profiles or args.sweeps):
        # default: do profiles
        args.profiles = True

    if args.single:
        run_single(outdir, n=args.n, K=args.K, rhoc=args.rhoc, dat_name="problem4.dat")

    if args.profiles:
        run_profiles(outdir, K=1.0e2, rhoc=1.28e-3)

    if args.sweeps:
        run_sweeps(outdir, rhoc_min=args.rhoc_min, rhoc_max=args.rhoc_max, rhoc_steps=args.rhoc_steps)


if __name__ == "__main__":
    main()
