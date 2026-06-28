"""Firefly Algorithm — Yang (2009)."""
from __future__ import annotations
import time
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class FireflyAlgorithm(BaseOptimizer):
    def __init__(self, pop_size=40, alpha=0.5, alpha_decay=0.97,
                 beta0=1.0, gamma=1.0,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size=pop_size; self.alpha=alpha; self.alpha_decay=alpha_decay
        self.beta0=beta0; self.gamma=gamma; self._name="FA"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, N   = len(lb), self.pop_size
        span   = ub - lb
        history = OptimizationHistory(); nfev = 0
        alpha  = self.alpha

        pop  = self._random_population(lb, ub, N)
        fits = np.array([func(x) for x in pop]); nfev += N
        best_idx = np.argmin(fits); best_x=pop[best_idx].copy(); best_f=fits[best_idx]
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            I = 1.0/(1.0+fits-fits.min()+1e-300)
            new_pop = pop.copy()
            for i in range(N):
                for j in range(N):
                    if I[j] <= I[i]: continue
                    diff = (pop[j]-pop[i])/(span+1e-300)
                    r2   = float(np.dot(diff,diff))
                    beta = self.beta0*np.exp(-self.gamma*r2)
                    eps  = self.rng.random(d)-0.5
                    new_pop[i] = new_pop[i]+beta*(pop[j]-pop[i])+alpha*span*eps
            pop  = self._bounce(new_pop, lb, ub)
            fits = np.array([func(x) for x in pop]); nfev += N
            idx  = np.argmin(fits)
            if fits[idx] < best_f: best_f=fits[idx]; best_x=pop[idx].copy()
            alpha *= self.alpha_decay
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev, alpha=alpha)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, alpha=alpha)
            if best_f < tol: break

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=best_f<tol, message="FA completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)
