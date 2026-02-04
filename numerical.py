import numpy as np
from scipy.differentiate import jacobian  # 사용자의 환경에 해당 모듈이 있다고 가정
import time

# equations.py에서 Equations 클래스와 상수를 가져옴
from equations import Equations, K, n

class Numerical(Equations):

    def __init__(self):
        super().__init__() # 부모 클래스 초기화
        # --- stage schedule (forced: coarse -> fine) ---
        self.point_prune = 80     # coarse eval / pruning / early Newton
        self.point_main  = 150    # main Newton (accuracy target ~1e-3 or better)

        # --- regularization knobs ---
        self.eps_center = 1e-6    # start integration at rhat=eps
        self.r_match    = 0.5     # matching radius

        # --- Newton/FD knobs ---
        self.rel_step   = 1e-4    # scale-aware FD step
        self.armijo_c   = 1e-4
        self.alpha_min  = 2.0**-20

        # per-rhoc cache
        self.rhoc   = None
        self.h_c    = None        # central enthalpy cache
        self.DELTA_H = 0.0        # smoothing scale used by rho_from_h

        # func() uses this unless point is explicitly passed
        self.point_eval = self.point_main

        # optional lightweight stats
        self.stats = dict(
            func_calls=0,
            jac_batches=0,
            newton_iters=0,
            retries=0,
            failures=0,
        )

    def central_entalphy(self, rhoc):
        # equations.py에서 가져온 K, n 사용
        return K * (1 + n) * rhoc**(1.0 / n)

    def get_numerical_jacobian(self, f, x0, eps=1e-8):
        x0 = np.asarray(x0, dtype=float)
        n_dim = len(x0)
        fx = f(x0)
        jac = np.zeros((n_dim, n_dim))
        
        for i in range(n_dim):
            x_perturbed = x0.copy()
            x_perturbed[i] += eps
            fx_perturbed = f(x_perturbed)
            jac[:, i] = (fx_perturbed - fx) / eps
        return jac

    def nrmethod2D(self, f, x0, eps):
        """
        Damped Newton with a very light Armijo-style decrease test + positivity guard.
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
                x_new = x - alpha*dx
                if (x_new[0] > 0.0) and (x_new[1] > 0.0):
                    fx_new = np.asarray(f(x_new), dtype=float)
                    if np.all(np.isfinite(fx_new)):
                        fnorm_new = np.linalg.norm(fx_new, ord=2)
                        if fnorm_new <= (1.0 - c_armijo*alpha) * fnorm:
                            x = x_new
                            break
                alpha *= 0.5

            if alpha < alpha_min:
                x = np.maximum(x - alpha_min*dx, 1e-12)

            if np.max(np.abs(alpha*dx)) < eps:
                return x

        raise RuntimeError("Newton did not converge within max iterations.")

    def rkheunPoisson_full(self, phiddot, hdot, phi_ini, h_ini, t_ini, t_fin, point, Rs):
        step = (t_fin - t_ini) / point
        t = np.linspace(t_ini, t_fin, point+1)

        Rs_arr = np.asarray(Rs, dtype=float)
        
        phi0 = np.asarray(phi_ini[0], dtype=float)
        phidot0 = np.asarray(phi_ini[1], dtype=float)
        h0 = np.asarray(h_ini, dtype=float)
        
        if phi0.ndim == 0: phi0 = phi0.reshape(1)
        if phidot0.ndim == 0: phidot0 = phidot0.reshape(1)
        if h0.ndim == 0: h0 = h0.reshape(1)
        
        phi0, phidot0, h0, Rs_arr = np.broadcast_arrays(phi0, phidot0, h0, Rs_arr)
        batch_shape = phi0.shape

        phi = np.empty((point+1,)+batch_shape, dtype=float)
        phidot = np.empty((point+1,)+batch_shape, dtype=float)
        h = np.empty((point+1,)+batch_shape, dtype=float)

        phi[0] = phi0
        phidot[0] = phidot0
        h[0] = h0

        for i in range(point):
            k1phi = phiddot(t[i], phi[i], phidot[i], h[i], Rs_arr)
            k1h   = hdot(t[i], phi[i], phidot[i], h[i], Rs_arr)

            k2phi = phiddot(t[i+1], phi[i]+step*phidot[i], phidot[i]+step*k1phi, h[i]+step*k1h, Rs_arr)
            k2h   = hdot(t[i+1], phi[i]+step*phidot[i], phidot[i]+step*k1phi, h[i]+step*k1h, Rs_arr)

            phi[i+1]    = phi[i]    + step*phidot[i] + 0.5 * step**2 * k1phi
            phidot[i+1] = phidot[i] + 0.5*step*(k1phi+k2phi)
            h[i+1]      = h[i]      + 0.5*step*(k1h+k2h)

        return t, phi, phidot, h

    def rkheunPoisson_end(self, phiddot, hdot, phi_ini, h_ini, t_ini, t_fin, point, Rs):
        step = (t_fin - t_ini) / point
        r = float(t_ini)

        Rs_arr = np.asarray(Rs, dtype=float)
        phi    = np.asarray(phi_ini[0], dtype=float)
        phidot = np.asarray(phi_ini[1], dtype=float)
        h      = np.asarray(h_ini, dtype=float)

        phi, phidot, h, Rs_arr = np.broadcast_arrays(phi, phidot, h, Rs_arr)

        for _ in range(point):
            k1_phi    = phidot
            k1_phidot = phiddot(r, phi, phidot, h, Rs_arr)
            k1_h      = hdot(r, phi, phidot, h, Rs_arr)

            phi_p    = phi    + step*k1_phi
            phidot_p = phidot + step*k1_phidot
            h_p      = h      + step*k1_h

            r2 = r + step
            k2_phi    = phidot_p
            k2_phidot = phiddot(r2, phi_p, phidot_p, h_p, Rs_arr)
            k2_h      = hdot(r2, phi_p, phidot_p, h_p, Rs_arr)

            phi    = phi    + 0.5*step*(k1_phi + k2_phi)
            phidot = phidot + 0.5*step*(k1_phidot + k2_phidot)
            h      = h      + 0.5*step*(k1_h + k2_h)

            r = r2

        return phi, phidot

    def bound_inter(self, rhoc, Rs, point=None):
        if point is None:
            point = self.point_eval

        if (self.rhoc is None) or (self.h_c is None):
            self.rhoc = float(rhoc)
            self.h_c  = self.central_entalphy(self.rhoc)

        eps = self.eps_center
        Rs_arr = np.asarray(Rs, dtype=float)

        a = (4.0*np.pi/3.0) * (Rs_arr**2) * float(rhoc)
        phi0    = 0.5 * a * (eps**2)
        phidot0 = a * eps
        h0      = self.h_c - 0.5 * a * (eps**2)

        phi, phidot = self.rkheunPoisson_end(
            self.phiddot, self.hdot,
            [phi0, phidot0], h0,
            eps, self.r_match, int(point), Rs_arr
        )
        return phi, phidot

    def bound_outer(self, Rs, Ms, point=None):
        if point is None:
            point = self.point_eval

        phi, phidot = self.rkheunPoisson_end(
            self.phiddot, self.hdot,
            [-Ms/Rs, Ms/Rs], 0.0,
            1.0, self.r_match, int(point), Rs
        )
        return phi, phidot

    def func(self, x, point=None):
        self.stats['func_calls'] += 1

        if point is None:
            point = self.point_eval

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            Rs = x[0]; Ms = x[1]
        else:
            if x.shape[0] != 2:
                x = x.T
            Rs = x[0]; Ms = x[1]

        Rs = np.abs(Rs)
        Ms = np.abs(Ms)

        phi_inter, phidot_inter = self.bound_inter(self.rhoc, Rs, point=point)
        phi_outer, phidot_outer = self.bound_outer(Rs, Ms, point=point)

        phi_diff    = (phi_inter - phi_outer) - (self.h_c + Ms/Rs)
        phidot_diff = (phidot_inter - phidot_outer)

        return np.stack([phi_diff, phidot_diff], axis=0)

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
        h = np.where(np.abs(x) < h, 0.5*np.maximum(np.abs(x), 1e-12), h)

        X = np.array([
            [x[0] + h[0], x[0] - h[0], x[0],        x[0]],
            [x[1],        x[1],        x[1] + h[1], x[1] - h[1]],
        ], dtype=float)

        FX = np.asarray(self.func(X, point=point), dtype=float)

        dF_dRs = (FX[:, 0] - FX[:, 1]) / (2.0*h[0])
        dF_dMs = (FX[:, 2] - FX[:, 3]) / (2.0*h[1])

        J = np.column_stack([dF_dRs, dF_dMs])
        return J

    def jacobian_and_Fx_batch(self, x, rel_step=None, point=None):
        self.stats['jac_batches'] += 1

        if rel_step is None:
            rel_step = self.rel_step
        if point is None:
            point = self.point_eval

        x = np.asarray(x, dtype=float)
        h = rel_step * np.maximum(1.0, np.abs(x))
        h = np.where(np.abs(x) < h, 0.5*np.maximum(np.abs(x), 1e-12), h)

        X = np.array([
            [x[0], x[0] + h[0], x[0] - h[0], x[0],        x[0]],
            [x[1], x[1],        x[1],        x[1] + h[1], x[1] - h[1]],
        ], dtype=float)

        FX = np.asarray(self.func(X, point=point), dtype=float)
        Fx = FX[:, 0]

        dF_dRs = (FX[:, 1] - FX[:, 2]) / (2.0*h[0])
        dF_dMs = (FX[:, 3] - FX[:, 4]) / (2.0*h[1])
        J = np.column_stack([dF_dRs, dF_dMs])

        return Fx, J

    def newton_damped(self, x0, tol=1e-10, xtol=1e-10, maxiter=25, rel_step=None, point=None):
        if rel_step is None:
            rel_step = self.rel_step
        if point is None:
            point = self.point_eval

        x = np.asarray(x0, dtype=float)
        x = np.maximum(np.abs(x), 1e-12)

        for _ in range(int(maxiter)):
            self.stats['newton_iters'] += 1

            Fx, J = self.jacobian_and_Fx_batch(x, rel_step=rel_step, point=point)
            if not np.all(np.isfinite(Fx)) or not np.all(np.isfinite(J)):
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
                x_new = x - alpha*dx
                if (x_new[0] > 0.0) and (x_new[1] > 0.0):
                    F_new = np.asarray(self.func(x_new, point=point), dtype=float)
                    nrm_new = float(np.linalg.norm(F_new))
                    if np.isfinite(nrm_new) and (nrm_new <= (1.0 - self.armijo_c*alpha)*nrm):
                        x = x_new
                        break
                alpha *= 0.5

            if alpha < self.alpha_min:
                x = np.maximum(x - self.alpha_min*dx, 1e-12)

        F_end = np.asarray(self.func(x, point=point), dtype=float)
        return x, float(np.linalg.norm(F_end)), False

    def generate_seeds(self, Rs0, Ms0, n_seeds=64, sigma=0.6, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        base = np.array([Rs0, Ms0], dtype=float)

        Z = rng.normal(0.0, sigma, size=(2, int(n_seeds)))
        seeds = base[:, None] * np.exp(Z)
        seeds[:, 0] = base
        seeds = np.clip(seeds, 1e-12, 1e12)
        return seeds

    def shooting_optimized(self, Rs0, Ms0,
                           point_prune=None, point_main=None,
                           n_seeds=96, prune_topk=8, sigma=0.6,
                           tol=1e-10, maxiter=25, rel_step=None,
                           parallel=False, n_jobs=-1, rng_seed=0):
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
        top = seeds[:, idx].T

        def solve_one(x0):
            x, nrm, ok = self.newton_damped(
                x0, tol=tol, xtol=1e-10, maxiter=maxiter,
                rel_step=0.5*rel_step, point=point_main
            )
            return x, nrm, ok

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

    def shooting(self, rhoc, Rs0, Ms0,
                 point_prune=None, point_main=None,
                 tol_main=1e-10, maxiter_main=25,
                 n_retry=1,
                 n_seeds=128, prune_topk=10, sigma=0.6,
                 parallel=False, n_jobs=-1):
        
        self.rhoc = float(rhoc)
        self.h_c  = self.central_entalphy(self.rhoc)
        self.DELTA_H = 1e-12 * max(1.0, abs(self.h_c))

        if point_prune is None:
            point_prune = self.point_prune
        if point_main is None:
            point_main  = self.point_main

        x0 = np.array([Rs0, Ms0], dtype=float)
        x0 = np.maximum(np.abs(x0), 1e-12)

        self.point_eval = int(point_prune)
        F0 = np.asarray(self.func(x0), dtype=float)
        nrm0 = float(np.linalg.norm(F0))

        if (not np.isfinite(nrm0)) or (nrm0 > 1e-2):
            x0, _, _ = self.newton_damped(
                x0, tol=1e-4, xtol=1e-10, maxiter=10,
                rel_step=self.rel_step, point=int(point_prune)
            )

        self.point_eval = int(point_main)
        x, nrm, ok = self.newton_damped(
            x0, tol=tol_main, xtol=1e-10, maxiter=maxiter_main,
            rel_step=0.5*self.rel_step, point=int(point_main)
        )
        if ok and np.isfinite(nrm):
            return float(x[0]), float(x[1]), float(nrm), True

        for k in range(int(n_retry)):
            self.stats['retries'] += 1
            sig = sigma + 0.2*k
            Rs, Ms, nrm2, ok2 = self.shooting_optimized(
                float(x[0]), float(x[1]),
                point_prune=int(point_prune), point_main=int(point_main),
                n_seeds=n_seeds, prune_topk=prune_topk, sigma=sig,
                tol=tol_main, maxiter=maxiter_main, rel_step=self.rel_step,
                parallel=parallel, n_jobs=n_jobs, rng_seed=1000+k
            )
            if ok2 and np.isfinite(nrm2):
                return Rs, Ms, nrm2, True

        self.stats['failures'] += 1
        return float(x[0]), float(x[1]), float(nrm), False