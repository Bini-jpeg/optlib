"""PSO (standard inertia-weight) + CLPSO (comprehensive learning)."""
from __future__ import annotations
import time
from typing import Callable, Optional, Union
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class PSO(BaseOptimizer):
    """Standard PSO with linearly decreasing inertia weight."""
    def __init__(self, pop_size=40, w_start=0.9, w_end=0.4,
                 c1=2.0, c2=2.0, v_clamp=0.5,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size = pop_size; self.w_start = w_start; self.w_end = w_end
        self.c1 = c1; self.c2 = c2; self.v_clamp = v_clamp
        self._name = "PSO"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, N   = len(lb), self.pop_size
        history = OptimizationHistory(); nfev = 0

        pos  = self._random_population(lb, ub, N)
        vel  = self.rng.uniform(-(ub-lb), (ub-lb), (N, d)) * 0.1
        fits = np.array([func(x) for x in pos]); nfev += N
        pbest = pos.copy(); pf = fits.copy()
        gi = np.argmin(pf); gbest = pbest[gi].copy(); gbest_f = pf[gi]
        v_max = self.v_clamp*(ub-lb) if self.v_clamp else None
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            w  = self.w_start - (self.w_start-self.w_end)*it/max_iter
            r1 = self.rng.random((N,d)); r2 = self.rng.random((N,d))
            vel = w*vel + self.c1*r1*(pbest-pos) + self.c2*r2*(gbest-pos)
            if v_max is not None: vel = np.clip(vel, -v_max, v_max)
            pos  = self._bounce(pos+vel, lb, ub)
            fits = np.array([func(x) for x in pos]); nfev += N
            imp  = fits < pf; pbest[imp] = pos[imp]; pf[imp] = fits[imp]
            gi   = np.argmin(pf)
            if pf[gi] < gbest_f: gbest_f = pf[gi]; gbest = pbest[gi].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, gbest_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev, w=w)
            self._log(it, gbest_f, nfev=nfev, elapsed=elapsed, w=w)
            if 0.0 <= gbest_f < tol: break

        return OptimizationResult(x=gbest, fun=gbest_f, nfev=nfev, nit=it,
            success=0.0 <= gbest_f < tol, message="PSO completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)


class CLPSO(BaseOptimizer):
    """Comprehensive Learning PSO — Liang et al. (2006)."""
    def __init__(self, pop_size=40, w_start=0.9, w_end=0.4, c=1.496,
                 refresh_gap=7, seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size = pop_size; self.w_start = w_start; self.w_end = w_end
        self.c = c; self.refresh_gap = refresh_gap; self._name = "CLPSO"

    def optimize(self, func, bounds, max_iter=1000, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, N   = len(lb), self.pop_size
        history = OptimizationHistory(); nfev = 0

        pos  = self._random_population(lb, ub, N)
        vel  = self.rng.uniform(-(ub-lb), (ub-lb), (N,d)) * 0.1
        fits = np.array([func(x) for x in pos]); nfev += N
        pbest = pos.copy(); pf = fits.copy()
        gbest_f = np.min(pf); gbest = pbest[np.argmin(pf)].copy()

        pc  = 0.05 + 0.45*(np.exp(10*np.arange(N)/(N-1))-1)/(np.exp(10)-1)
        fi  = np.tile(np.arange(N)[:,None], (1,d))
        lag = np.zeros(N, dtype=int)
        it  = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            w = self.w_start - (self.w_start-self.w_end)*it/max_iter
            for i in range(N):
                if lag[i] >= self.refresh_gap:
                    for j in range(d):
                        if self.rng.random() < pc[i]:
                            a,b = self.rng.choice(N,2,replace=False)
                            fi[i,j] = a if pf[a]<pf[b] else b
                        else: fi[i,j] = i
                    lag[i] = 0
            ex  = np.array([[pbest[fi[i,j],j] for j in range(d)] for i in range(N)])
            r   = self.rng.random((N,d))
            vel = np.clip(w*vel + self.c*r*(ex-pos), -(ub-lb), (ub-lb))
            pos = self._bounce(pos+vel, lb, ub)
            fits = np.array([func(x) for x in pos]); nfev += N
            imp = fits<pf; pbest[imp]=pos[imp]; pf[imp]=fits[imp]
            lag[imp]=0; lag[~imp]+=1
            gi = np.argmin(pf)
            if pf[gi] < gbest_f: gbest_f=pf[gi]; gbest=pbest[gi].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, gbest_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev, w=w)
            self._log(it, gbest_f, nfev=nfev, elapsed=elapsed, w=w)
            if 0.0 <= gbest_f < tol: break

        return OptimizationResult(x=gbest, fun=gbest_f, nfev=nfev, nit=it,
            success=0.0 <= gbest_f < tol, message="CLPSO completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)
