"""Generalised Pattern Search (GPS) — Torczon (1997)."""
from __future__ import annotations
import time
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult
from optlib.preprocessing.preprocessing import latin_hypercube_sample


class PatternSearch(BaseOptimizer):
    def __init__(self, x0=None, delta_init=0.5, delta_min=1e-8,
                 tau=4.0, n_search=10,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.x0=x0; self.delta_init=delta_init; self.delta_min=delta_min
        self.tau=tau; self.n_search=n_search; self._name="PatternSearch"

    def optimize(self, func, bounds, max_iter=5_000, max_fev=None, **_):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, span = len(lb), ub-lb
        history = OptimizationHistory(); nfev = 0

        x = np.clip(self.x0 if self.x0 is not None
                    else lb+span*self.rng.random(d), lb, ub)
        f = func(x); nfev += 1
        best_x, best_f = x.copy(), f
        delta = self.delta_init*span
        E     = np.vstack([np.eye(d), -np.eye(d)])
        it    = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            improved = False
            if self.n_search > 0:
                pts = latin_hypercube_sample(self.n_search, d,
                      np.column_stack([lb,ub]), seed=None)
                for pt in pts:
                    pm = self._clip(x+np.round((pt-x)/delta)*delta, lb, ub)
                    fp = func(pm); nfev += 1
                    if fp < f: x,f=pm,fp; improved=True
            if not improved:
                for k in self.rng.permutation(2*d):
                    xt = self._clip(x+delta*E[k], lb, ub)
                    ft = func(xt); nfev += 1
                    if ft < f: x,f=xt,ft; improved=True; break
            delta = np.minimum(delta*self.tau, span) if improved else delta/self.tau
            if f < best_f: best_f,best_x = f,x.copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, float('nan'), float('nan'),
                               elapsed=elapsed, nfev=nfev, delta_max=float(delta.max()))
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, delta=float(delta.mean()))
            if float(delta.max()) < self.delta_min: break

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=float(delta.max())<self.delta_min, message="PatternSearch completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)
