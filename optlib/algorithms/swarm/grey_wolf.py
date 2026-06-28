"""Grey Wolf Optimizer — Mirjalili et al. (2014)."""
from __future__ import annotations
import time
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class GreyWolfOptimizer(BaseOptimizer):
    def __init__(self, pop_size=30, seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size = pop_size; self._name = "GWO"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, N   = len(lb), self.pop_size
        history = OptimizationHistory(); nfev = 0

        pop  = self._random_population(lb, ub, N)
        fits = np.array([func(x) for x in pop]); nfev += N
        order = np.argsort(fits)
        alpha_x,beta_x,delta_x = pop[order[0]].copy(),pop[order[1]].copy(),pop[order[2]].copy()
        alpha_f = fits[order[0]]
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            a = 2.0 - 2.0*it/max_iter
            for i in range(N):
                x1 = self._hunt(pop[i], alpha_x, a, d)
                x2 = self._hunt(pop[i], beta_x,  a, d)
                x3 = self._hunt(pop[i], delta_x, a, d)
                pop[i] = self._clip((x1+x2+x3)/3.0, lb, ub)
            fits = np.array([func(x) for x in pop]); nfev += N
            order = np.argsort(fits)
            if fits[order[0]] < alpha_f:
                alpha_f=fits[order[0]]; alpha_x=pop[order[0]].copy()
            beta_x=pop[order[1]].copy(); delta_x=pop[order[2]].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, alpha_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev, a=a)
            self._log(it, alpha_f, nfev=nfev, elapsed=elapsed, a=a)
            if 0.0 <= alpha_f < tol: break

        return OptimizationResult(x=alpha_x, fun=alpha_f, nfev=nfev, nit=it,
            success=0.0 <= alpha_f < tol, message="GWO completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)

    def _hunt(self, wolf, leader, a, d):
        r1=self.rng.random(d); r2=self.rng.random(d)
        return leader - (2*a*r1-a)*np.abs(2*r2*leader-wolf)
