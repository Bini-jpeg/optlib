"""Artificial Bee Colony — Karaboga (2005)."""
from __future__ import annotations
import time
from typing import Callable, Optional, Union
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class ArtificialBeeColony(BaseOptimizer):
    def __init__(self, pop_size=50, limit=None,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size = pop_size; self._limit = limit; self._name = "ABC"

    def optimize(self, func, bounds, max_iter=1000, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, N   = len(lb), self.pop_size
        limit  = self._limit if self._limit is not None else N*d//2
        history = OptimizationHistory(); nfev = 0

        sources = self._random_population(lb, ub, N)
        fits    = np.array([func(x) for x in sources]); nfev += N
        trials  = np.zeros(N, dtype=int)
        best_idx = np.argmin(fits); best_x = sources[best_idx].copy(); best_f = fits[best_idx]

        def exploit(idx):
            nonlocal nfev
            k = self.rng.integers(0, N-1); k += (k >= idx)
            j = self.rng.integers(0, d)
            phi = self.rng.uniform(-1.0, 1.0)
            c = sources[idx].copy()
            c[j] = np.clip(c[j]+phi*(c[j]-sources[k,j]), lb[j], ub[j])
            fc = func(c); nfev += 1
            return c, fc

        it = 0
        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            for i in range(N):
                c, fc = exploit(i)
                if fc < fits[i]: sources[i]=c; fits[i]=fc; trials[i]=0
                else: trials[i]+=1

            ff = np.where(fits>=0, 1/(1+fits), 1+np.abs(fits))
            probs = ff/ff.sum()
            for _ in range(N):
                i = self.rng.choice(N, p=probs)
                c, fc = exploit(i)
                if fc < fits[i]: sources[i]=c; fits[i]=fc; trials[i]=0
                else: trials[i]+=1

            scouts = np.where(trials > limit)[0]
            if scouts.size:
                sources[scouts] = self._random_population(lb, ub, scouts.size)
                nf = np.array([func(x) for x in sources[scouts]]); nfev += scouts.size
                fits[scouts] = nf; trials[scouts] = 0

            idx = np.argmin(fits)
            if fits[idx] < best_f: best_f=fits[idx]; best_x=sources[idx].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev, scouts=int(scouts.size))
            self._log(it, best_f, nfev=nfev, elapsed=elapsed)
            if best_f < tol: break

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=best_f<tol, message="ABC completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)
