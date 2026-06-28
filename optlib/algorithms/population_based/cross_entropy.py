"""Cross-Entropy Method — Rubinstein & Kroese (2004)."""
from __future__ import annotations
import time
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class CrossEntropyMethod(BaseOptimizer):
    def __init__(self, pop_size=50, elite_frac=0.20, sigma_init=0.3,
                 sigma_min=1e-6, alpha=0.7,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size=pop_size; self.elite_frac=elite_frac
        self.sigma_init=sigma_init; self.sigma_min=sigma_min
        self.alpha=alpha; self._name="CEM"

    def optimize(self, func, bounds, max_iter=200, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, span = len(lb), ub-lb
        history = OptimizationHistory(); nfev = 0
        n_elite = max(2, int(self.elite_frac*self.pop_size))
        mu      = (lb+ub)/2.0; sigma = self.sigma_init*span
        best_x  = mu.copy(); best_f = func(mu); nfev += 1
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            samples = self._bounce(mu+sigma*self.rng.standard_normal((self.pop_size,d)), lb, ub)
            fits    = np.array([func(x) for x in samples]); nfev += self.pop_size
            elite   = samples[np.argpartition(fits, n_elite)[:n_elite]]
            mu    = self.alpha*elite.mean(axis=0)    + (1-self.alpha)*mu
            sigma = self.alpha*(elite.std(axis=0)+self.sigma_min) + (1-self.alpha)*sigma
            sigma = np.maximum(sigma, self.sigma_min)
            ib    = int(np.argmin(fits))
            if fits[ib] < best_f: best_f=fits[ib]; best_x=samples[ib].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev, sigma_mean=float(sigma.mean()))
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, sigma=float(sigma.mean()))
            if best_f < tol or sigma.max() < self.sigma_min*10: break

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=best_f<tol, message="CEM completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)
