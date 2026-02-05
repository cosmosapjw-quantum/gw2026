# 뉴턴 버전과 차이가 거의 없음. 다만 바뀐 TOV에 해당하여 AI에게 조정을 요청함

import numpy as np
from scipy.differentiate import jacobian
from equations import Equations, K, n

class Numerical(Equations):
    def __init__(self):
        super().__init__()

        # --- stage schedule (coarse -> fine) ---
        self.point_prune = 80
        self.point_main  = 150

        # --- geometry/regularization ---
        self.eps_center = 1e-6   # start at rhat=eps (avoid r=0 singular)
        self.r_match    = 0.5    # matching point

        # --- Newton/FD knobs ---
        self.rel_step   = 1e-4
        self.armijo_c   = 1e-4
        self.alpha_min  = 2.0**-20

        # per-rhoc cache
        self.rhoc   = None
        self.h_c    = None

        # active point count used by func()
        self.point_eval = self.point_main

        # lightweight stats
        self.stats = dict(
            func_calls=0,
            jac_batches=0,
            newton_iters=0,
            retries=0,
            failures=0,
        )

    def central_entalphy(self, rhoc):
        return 1.0 + K * (1.0 + n) * rhoc**(1.0 / n)

    # -------------------------
    # SciPy FD Jacobian helper (optional)
    # -------------------------
    def get_numerical_jacobian(self, f, x0, eps=1e-8):
        x0 = np.asarray(x0, dtype=float)
        n_dim = len(x0)
        fx = np.asarray(f(x0), dtype=float)
        jac = np.zeros((n_dim, n_dim), dtype=float)

        for i in range(n_dim):
            x_perturbed = x0.copy()
            x_perturbed[i] += eps
            fx_perturbed = np.asarray(f(x_perturbed), dtype=float)
            jac[:, i] = (fx_perturbed - fx) / eps
        return jac

    def nrmethod2D(self, f, x0, eps):
        """
        Damped Newton using scipy.differentiate.jacobian (FD-based).
        Kept for compatibility; main pipeline uses newton_damped below.
        """
        x = np.asarray(x0, dtype=float)

        max_newton = 30
        c_armijo = 1e-4
        alpha_min = 2.0**-20

        for _ in range(max_newton):
            fx = np.asarray(f(x), dtype=float)
            if not np.all(np.isfinite(fx)):
                raise RuntimeError("f(x) became non-finite.")

            fnorm = np.linalg.norm(fx, ord=2)
            if fnorm < eps:
                return x

            init_step = 1e-3 * np.maximum(1.0, np.abs(x))

            J = jacobian(
                f, x,
                order=2,
                maxiter=1,
                initial_step=init_step
            ).df

            dx = np.linalg.solve(J, fx)

            alpha = 1.0
            while alpha >= alpha_min:
                x_new = x - alpha * dx
                if (x_new[0] > 0.0) and (x_new[1] > 0.0):
                    fx_new = np.asarray(f(x_new), dtype=float)
                    if np.all(np.isfinite(fx_new)):
                        fnorm_new = np.linalg.norm(fx_new, ord=2)
                        if fnorm_new <= (1.0 - c_armijo * alpha) * fnorm:
                            x = x_new
                            break
                alpha *= 0.5

            if alpha < alpha_min:
                x = np.maximum(x - alpha_min * dx, 1e-12)

            if np.max(np.abs(alpha * dx)) < eps:
                return x

        raise RuntimeError("Newton did not converge within max iterations.")

    # -------------------------
    # Custom Heun(RK2) integrator for TOV in (m, h) with independent var rhat
    # -------------------------
    def rkheunTOV_end(self, m_ini, h_ini, rhat_ini, rhat_fin, point, Rs):
        """
        Integrate from rhat_ini to rhat_fin (can be inward if rhat_fin < rhat_ini)
        using custom Heun (RK2) on the system:
          dm/drhat = tov_mdot(...)
          dh/drhat = tov_hdot(...)
        Supports broadcasting over Rs (and thus over batch evaluations for FD Jacobian).
        """
        step = (rhat_fin - rhat_ini) / point
        rhat = float(rhat_ini)

        Rs_arr = np.asarray(Rs, dtype=float)
        m = np.asarray(m_ini, dtype=float)
        h = np.asarray(h_ini, dtype=float)

        m, h, Rs_arr = np.broadcast_arrays(m, h, Rs_arr)

        for _ in range(int(point)):
            k1_m = self.tov_mdot(rhat, m, h, Rs_arr)
            k1_h = self.tov_hdot(rhat, m, h, Rs_arr)

            m_p = m + step * k1_m
            h_p = h + step * k1_h
            r2 = rhat + step

            k2_m = self.tov_mdot(r2, m_p, h_p, Rs_arr)
            k2_h = self.tov_hdot(r2, m_p, h_p, Rs_arr)

            m = m + 0.5 * step * (k1_m + k2_m)
            h = h + 0.5 * step * (k1_h + k2_h)
            rhat = r2

        return m, h

    # -------------------------
    # Boundary integrations (TOV)
    # -------------------------
    def bound_inter(self, rhoc, Rs, point=None):
        """
        Inner integration: rhat = eps_center -> r_match
        BC at center (regularized at eps):
          h(eps) = h_c
          m(eps) ≈ (4π/3) ε_c r^3  with r = Rs*eps
        """
        if point is None:
            point = self.point_eval

        rhoc = float(rhoc)
        # refresh cache only when needed
        if (self.rhoc is None) or (self.h_c is None) or (rhoc != self.rhoc):
            self.rhoc = rhoc
            self.h_c = self.central_entalphy(rhoc)

        eps = float(self.eps_center)
        Rs_arr = np.asarray(Rs, dtype=float)

        h0 = self.h_c
        eps_c = self.eps_from_h(h0)

        r0 = Rs_arr * eps
        m0 = (4.0 * np.pi / 3.0) * eps_c * (r0**3)

        m_end, h_end = self.rkheunTOV_end(
            m0, h0,
            eps, self.r_match, int(point),
            Rs_arr
        )
        return m_end, h_end

    def bound_outer(self, Rs, Ms, point=None):
        """
        Outer integration: rhat = 1 -> r_match (inward)
        BC at surface:
          h(1) = 0
          m(1) = Ms
        """
        if point is None:
            point = self.point_eval

        Rs_arr = np.asarray(Rs, dtype=float)
        Ms_arr = np.asarray(Ms, dtype=float)

        m0 = Ms_arr
        h0 = 0.0

        m_end, h_end = self.rkheunTOV_end(
            m0, h0,
            1.0, self.r_match, int(point),
            Rs_arr
        )
        return m_end, h_end

    # -------------------------
    # 2D shooting residual for (Rs, Ms)
    # -------------------------
    def func(self, x, point=None):
        """
        x = [Rs, Ms] or batch (2,k)/(k,2)
        residual:
          F1 = m_inter(r_match) - m_outer(r_match)
          F2 = h_inter(r_match) - h_outer(r_match)
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

        # positivity guard (keeps search in physical quadrant)
        Rs = np.abs(Rs)
        Ms = np.abs(Ms)

        m_inter, h_inter = self.bound_inter(self.rhoc, Rs, point=point)
        m_outer, h_outer = self.bound_outer(Rs, Ms, point=point)

        m_diff = m_inter - m_outer
        h_diff = h_inter - h_outer

        return np.stack([m_diff, h_diff], axis=0)

    # -------------------------
    # Batch FD Jacobians (central difference)
    # -------------------------
    def residual_norm(self, F):
        F = np.asarray(F, dtype=float)
        return np.sqrt(np.sum(F*F, axis=0))

    def jacobian_fd2_batch(self, x, rel_step=None, point=None):
        if rel_step is None:
            rel_step = self.rel_step
        if point is None:
            point = self.point_eval

        x = np.asarray(x, dtype=float)
        h = rel_step * np.maximum(1.0, np.abs(x))
        h = np.where(np.abs(x) < h, 0.5 * np.maximum(np.abs(x), 1e-12), h)

        X = np.array([
            [x[0] + h[0], x[0] - h[0], x[0],        x[0]],
            [x[1],        x[1],        x[1] + h[1], x[1] - h[1]],
        ], dtype=float)

        FX = np.asarray(self.func(X, point=point), dtype=float)  # (2,4)

        dF_dRs = (FX[:, 0] - FX[:, 1]) / (2.0 * h[0])
        dF_dMs = (FX[:, 2] - FX[:, 3]) / (2.0 * h[1])

        return np.column_stack([dF_dRs, dF_dMs])  # (2,2)

    def jacobian_and_Fx_batch(self, x, rel_step=None, point=None):
        """
        Evaluate Fx and 2x2 Jacobian in one batched call:
          X = [x, x±h e1, x±h e2] -> 5 evaluations in one func() call.
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

    # -------------------------
    # Damped Newton (FD Jacobian), multi-start optional
    # -------------------------
    def newton_damped(self, x0, tol=1e-10, xtol=1e-10, maxiter=25, rel_step=None, point=None):
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

    def shooting(
        self, rhoc, Rs0, Ms0,
        point_prune=None, point_main=None,
        tol_main=1e-10, maxiter_main=25,
        n_retry=1,
        n_seeds=128, prune_topk=10, sigma=0.6,
        parallel=False, n_jobs=-1
    ):
        # cache + smoothing scale
        self.rhoc = float(rhoc)
        self.h_c  = self.central_entalphy(self.rhoc)
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
