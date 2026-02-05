import numpy as np
from equations import Equations, K, n

# 기존 shooting method 구현이 초기값에 따라 다소 민감하게 반응해서 AI에게 탐색 최적화를 요청해서 업그레이드 된 버전
# tree search와 유사한 구조로 랜덤 초기값, line search, 및 pruning을 통해 안정성을 확보하라고 명령
# 알고리즘 아이디어는 본인 제안, 세부적 구현을 AI에게 의뢰

class Numerical(Equations):
    def __init__(self):
        super().__init__()

        # --- stage schedule (coarse -> fine) ---
        self.point_prune = 80
        self.point_main = 150

        # --- regularization / matching ---
        self.eps_center = 1e-6
        self.r_match = 0.5

        # --- Newton/FD knobs ---
        self.rel_step = 1e-4
        self.armijo_c = 1e-4
        self.alpha_min = 2.0**-20

        # cache per rhoc
        self.rhoc = None
        self.h_c = None

        # func() default resolution
        self.point_eval = self.point_main

        # stats
        self.stats = dict(
            func_calls=0,
            jac_batches=0,
            newton_iters=0,
            retries=0,
            failures=0,
        )

    def central_entalphy(self, rhoc):
        return K * (1.0 + n) * rhoc**(1.0 / n)

    # ------------------------------------------------------------
    # Custom Heun (RK2) for Newton/Poisson system in (phi, phidot, h)
    # ------------------------------------------------------------
    def rkheunNewton_end(self, phi_ini, phidot_ini, h_ini, rhat_ini, rhat_fin, point, Rs):
        """
        Integrate from rhat_ini to rhat_fin using custom Heun (RK2) on:
          phi'    = phidot
          phidot' = phiddot(rhat, phi, phidot, h, Rs)
          h'      = -phidot
        Supports broadcasting over batch Rs (and batch Ms through ICs).
        """
        step = (rhat_fin - rhat_ini) / point
        rhat = float(rhat_ini)

        Rs_arr = np.asarray(Rs, dtype=float)
        phi = np.asarray(phi_ini, dtype=float)
        phidot = np.asarray(phidot_ini, dtype=float)
        h = np.asarray(h_ini, dtype=float)

        # broadcast for batch evaluation
        phi, phidot, h, Rs_arr = np.broadcast_arrays(phi, phidot, h, Rs_arr)

        for _ in range(int(point)):
            # k1 at (rhat, y)
            k1_phi = phidot
            k1_phidot = self.phiddot(rhat, phi, phidot, h, Rs_arr)
            k1_h = -phidot

            # predictor (Euler)
            phi_p = phi + step * k1_phi
            phidot_p = phidot + step * k1_phidot
            h_p = h + step * k1_h
            r2 = rhat + step

            # k2 at (r2, y_pred)
            k2_phi = phidot_p
            k2_phidot = self.phiddot(r2, phi_p, phidot_p, h_p, Rs_arr)
            k2_h = -phidot_p

            # corrector (average slopes)
            phi = phi + 0.5 * step * (k1_phi + k2_phi)
            phidot = phidot + 0.5 * step * (k1_phidot + k2_phidot)
            h = h + 0.5 * step * (k1_h + k2_h)
            rhat = r2

        return phi, phidot, h

    # ------------------------------------------------------------
    # Boundary integrations
    # ------------------------------------------------------------
    def bound_inter(self, rhoc, Rs, point=None):
        """
        Inner integration: rhat=eps -> r_match
        Regularized series start:
          phidot ~ a*rhat, phi ~ 0.5*a*rhat^2, h ~ h_c - phi
        where a = (4π/3) Rs^2 rhoc (Newtonian Poisson center)
        """
        if point is None:
            point = self.point_eval

        # ensure cache
        rhoc = float(rhoc)
        if (self.rhoc is None) or (self.h_c is None) or (rhoc != self.rhoc):
            self.rhoc = rhoc
            self.h_c = self.central_entalphy(rhoc)

        eps = float(self.eps_center)
        Rs_arr = np.asarray(Rs, dtype=float)

        a = (4.0*np.pi/3.0) * (Rs_arr**2) * rhoc
        phi0 = 0.5 * a * (eps**2)
        phidot0 = a * eps
        h0 = self.h_c - phi0

        phi, phidot, h = self.rkheunNewton_end(
            phi0, phidot0, h0,
            0.0, self.r_match, int(point),
            Rs_arr
        )
        return phi, phidot, h

    def bound_outer(self, Rs, Ms, point=None):
        """
        Outer integration: rhat=1 -> r_match (inward)
        Surface BC (vacuum):
          phi(1) = -Ms/Rs, phidot(1) = +Ms/Rs, h(1)=0
        (rho_from_h will keep density zero outside since h<=0 region maps to rho=0)
        """
        if point is None:
            point = self.point_eval

        Rs_arr = np.asarray(Rs, dtype=float)
        Ms_arr = np.asarray(Ms, dtype=float)

        phi0 = -Ms_arr / Rs_arr
        phidot0 = Ms_arr / Rs_arr
        h0 = 0.0

        phi, phidot, h = self.rkheunNewton_end(
            phi0, phidot0, h0,
            1.0, self.r_match, int(point),
            Rs_arr
        )
        return phi, phidot, h

    # ------------------------------------------------------------
    # 2D shooting residual in (Rs, Ms)
    # ------------------------------------------------------------
    def func(self, x, point=None):
        """
        Residual at r_match:
          F1 = h_inter - h_outer
          F2 = phidot_inter - phidot_outer
        This matches the design doc definition (enthalpy and derivative continuity).
        """
        self.stats["func_calls"] += 1

        if point is None:
            point = self.point_eval

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            Rs = x[0]
            Ms = x[1]
        else:
            if x.shape[0] != 2:
                x = x.T
            Rs = x[0]
            Ms = x[1]

        Rs = np.abs(Rs)
        Ms = np.abs(Ms)

        phi_i, phidot_i, h_i = self.bound_inter(self.rhoc, Rs, point=point)
        phi_o, phidot_o, h_o = self.bound_outer(Rs, Ms, point=point)

        h_diff = h_i - h_o
        phidot_diff = phidot_i - phidot_o

        return np.stack([h_diff, phidot_diff], axis=0)

    # ------------------------------------------------------------
    # Batch FD Jacobian (central difference) + damped Newton
    # ------------------------------------------------------------
    def residual_norm(self, F):
        F = np.asarray(F, dtype=float)
        return np.sqrt(np.sum(F*F, axis=0))

    def jacobian_and_Fx_batch(self, x, rel_step=None, point=None):
        """
        Evaluate Fx and 2x2 Jacobian in one batched func() call:
          X = [x, x±h e1, x±h e2] -> 5 evaluations
        """
        self.stats["jac_batches"] += 1

        if rel_step is None:
            rel_step = self.rel_step
        if point is None:
            point = self.point_eval

        x = np.asarray(x, dtype=float)
        h = rel_step * np.maximum(1.0, np.abs(x))
        h = np.where(np.abs(x) < h, 0.5 * np.maximum(np.abs(x), 1e-12), h)

        X = np.array([
            [x[0], x[0] + h[0], x[0] - h[0], x[0],        x[0]],
            [x[1], x[1],        x[1],        x[1] + h[1], x[1] - h[1]],
        ], dtype=float)

        FX = np.asarray(self.func(X, point=point), dtype=float)  # (2,5)
        Fx = FX[:, 0]

        dF_dRs = (FX[:, 1] - FX[:, 2]) / (2.0 * h[0])
        dF_dMs = (FX[:, 3] - FX[:, 4]) / (2.0 * h[1])
        J = np.column_stack([dF_dRs, dF_dMs])
        return Fx, J

    def newton_damped(self, x0, tol=1e-10, xtol=1e-10, maxiter=25, rel_step=None, point=None):
        """
        Damped Newton with Armijo-like decrease on ||F|| + positivity guard.
        FD Jacobian is central-difference (batched evaluation).
        """
        if rel_step is None:
            rel_step = self.rel_step
        if point is None:
            point = self.point_eval

        x = np.asarray(x0, dtype=float)
        x = np.maximum(np.abs(x), 1e-12)

        for _ in range(int(maxiter)):
            self.stats["newton_iters"] += 1

            Fx, J = self.jacobian_and_Fx_batch(x, rel_step=rel_step, point=point)
            if (not np.all(np.isfinite(Fx))) or (not np.all(np.isfinite(J))):
                return x, float("inf"), False

            nrm = float(np.linalg.norm(Fx))
            try:
                dx = np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                return x, nrm, False

            step_rel = float(np.max(np.abs(dx) / np.maximum(1.0, np.abs(x))))
            if (nrm < tol) and (step_rel < xtol):
                return x, nrm, True

            alpha = 1.0
            while alpha >= self.alpha_min:
                x_new = x - alpha * dx
                if (x_new[0] > 0.0) and (x_new[1] > 0.0):
                    F_new = np.asarray(self.func(x_new, point=point), dtype=float)
                    nrm_new = float(np.linalg.norm(F_new))
                    if np.isfinite(nrm_new) and (nrm_new <= (1.0 - self.armijo_c * alpha) * nrm):
                        x = x_new
                        break
                alpha *= 0.5

            if alpha < self.alpha_min:
                x = np.maximum(x - self.alpha_min * dx, 1e-12)

        F_end = np.asarray(self.func(x, point=point), dtype=float)
        return x, float(np.linalg.norm(F_end)), False

    # ------------------------------------------------------------
    # Multi-start / pruning for robustness
    # ------------------------------------------------------------
    def generate_seeds(self, Rs0, Ms0, n_seeds=64, sigma=0.6, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        base = np.array([Rs0, Ms0], dtype=float)
        Z = rng.normal(0.0, sigma, size=(2, int(n_seeds)))
        seeds = base[:, None] * np.exp(Z)
        seeds[:, 0] = base
        return np.clip(seeds, 1e-12, 1e12)

    def shooting_optimized(
        self, Rs0, Ms0,
        point_prune=None, point_main=None,
        n_seeds=96, prune_topk=8, sigma=0.6,
        tol=1e-10, maxiter=25, rel_step=None,
        parallel=False, n_jobs=-1, rng_seed=0
    ):
        if rel_step is None:
            rel_step = self.rel_step
        if point_prune is None:
            point_prune = self.point_prune
        if point_main is None:
            point_main = self.point_main

        rng = np.random.default_rng(int(rng_seed))
        seeds = self.generate_seeds(Rs0, Ms0, n_seeds=n_seeds, sigma=sigma, rng=rng)

        F_seeds = self.func(seeds, point=point_prune)
        norms = self.residual_norm(F_seeds)
        norms = np.where(np.isfinite(norms), norms, np.inf)

        idx = np.argsort(norms)[:int(prune_topk)]
        top = seeds[:, idx].T  # (topk, 2)

        def solve_one(x0):
            return self.newton_damped(
                x0, tol=tol, xtol=1e-10, maxiter=maxiter,
                rel_step=0.5 * rel_step, point=point_main
            )

        if parallel and len(top) > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(solve_one)(top[i]) for i in range(len(top))
            )
        else:
            results = [solve_one(top[i]) for i in range(len(top))]

        converged = [r for r in results if r[2]]
        best = min(converged, key=lambda r: r[1]) if converged else min(results, key=lambda r: r[1])

        x_best, nrm_best, ok_best = best
        return float(x_best[0]), float(x_best[1]), float(nrm_best), bool(ok_best)

    # ------------------------------------------------------------
    # Main shooting entry (2-stage schedule + retry)
    # ------------------------------------------------------------
    def shooting(
        self, rhoc, Rs0, Ms0,
        point_prune=None, point_main=None,
        tol_main=1e-10, maxiter_main=25,
        n_retry=1,
        n_seeds=128, prune_topk=10, sigma=0.6,
        parallel=False, n_jobs=-1
    ):
        self.rhoc = float(rhoc)
        self.h_c = self.central_entalphy(self.rhoc)

        # smoothing scale (set 0.0 to disable)
        self.DELTA_H = 1e-12 * max(1.0, abs(self.h_c))

        if point_prune is None:
            point_prune = self.point_prune
        if point_main is None:
            point_main = self.point_main

        x0 = np.array([Rs0, Ms0], dtype=float)
        x0 = np.maximum(np.abs(x0), 1e-12)

        # coarse stage
        self.point_eval = int(point_prune)
        F0 = np.asarray(self.func(x0), dtype=float)
        nrm0 = float(np.linalg.norm(F0))

        if (not np.isfinite(nrm0)) or (nrm0 > 1e-2):
            x0, _, _ = self.newton_damped(
                x0, tol=1e-4, xtol=1e-10, maxiter=10,
                rel_step=self.rel_step, point=int(point_prune)
            )

        # fine stage
        self.point_eval = int(point_main)
        x, nrm, ok = self.newton_damped(
            x0, tol=tol_main, xtol=1e-10, maxiter=maxiter_main,
            rel_step=0.5 * self.rel_step, point=int(point_main)
        )
        if ok and np.isfinite(nrm):
            return float(x[0]), float(x[1]), float(nrm), True

        # retry stage (multi-start)
        for k in range(int(n_retry)):
            self.stats["retries"] += 1
            sig = sigma + 0.2 * k
            Rs, Ms, nrm2, ok2 = self.shooting_optimized(
                float(x[0]), float(x[1]),
                point_prune=int(point_prune), point_main=int(point_main),
                n_seeds=n_seeds, prune_topk=prune_topk, sigma=sig,
                tol=tol_main, maxiter=maxiter_main, rel_step=self.rel_step,
                parallel=parallel, n_jobs=n_jobs, rng_seed=1000 + k
            )
            if ok2 and np.isfinite(nrm2):
                return Rs, Ms, nrm2, True

        self.stats["failures"] += 1
        return float(x[0]), float(x[1]), float(nrm), False
