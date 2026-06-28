"""Simulated Annealing — exponential / linear / logarithmic / adaptive cooling."""
from __future__ import annotations
import math, time
from typing import Callable, Optional, Union
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class SimulatedAnnealing(BaseOptimizer):
    def __init__(self, T0=None, Tf=1e-8, alpha=0.995,
                 schedule='exponential', step_size=0.1,
                 neighbour_fn=None, n_restarts=0,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.T0=T0; self.Tf=Tf; self.alpha=alpha; self.schedule=schedule
        self.step_size=step_size; self.neighbour_fn=neighbour_fn
        self.n_restarts=n_restarts; self._name=f"SA-{schedule}"

    def optimize(self, func, bounds, max_iter=10_000, max_fev=None, tol=1e-10):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d, span = len(lb), ub-lb
        history = OptimizationHistory(); nfev = 0

        x = self._random_population(lb, ub, 1)[0]
        f = func(x); nfev += 1
        best_x, best_f = x.copy(), f

        T = self.T0 or self._auto_T0(func, lb, ub, d, 80); nfev += 80
        T_start = T
        win: list[bool] = []
        no_impr = 0
        stag = max_iter//(self.n_restarts+1) if self.n_restarts else max_iter
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, max_fev): break
            x_new = (self.neighbour_fn(x, self.rng) if self.neighbour_fn
                     else self._bounce(x+self.rng.standard_normal(d)*self.step_size*span, lb, ub))
            f_new = func(x_new); nfev += 1
            delta = f_new - f
            if delta < 0 or (T > 1e-300 and self.rng.random() < math.exp(-delta/T)):
                x, f = x_new, f_new; win.append(True)
            else: win.append(False)
            if len(win) > 100: win.pop(0)
            if f < best_f: best_f, best_x = f, x.copy(); no_impr = 0
            else: no_impr += 1
            T = self._cool(T, T_start, it, max_iter, win)
            if self.n_restarts and no_impr >= stag:
                x, f = best_x.copy(), best_f; T = T_start*0.5; no_impr = 0
            elapsed = time.perf_counter()-t0
            rate = float(np.mean(win)) if win else 1.0
            if self.store_history:
                history.record(it, best_f, f, float('nan'),
                               elapsed=elapsed, nfev=nfev, T=T, accept_rate=rate)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, T=T)
            if 0.0 <= best_f < tol or T < self.Tf: break

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=best_f<tol, message=f"SA-{self.schedule} completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)

    def _cool(self, T, T_start, it, max_iter, win):
        s = self.schedule
        if s == 'exponential': return max(T*self.alpha, self.Tf)
        if s == 'linear':      return max(T_start*(1.0-it/max_iter), self.Tf)
        if s == 'logarithmic': return max(T_start/math.log(2.0+it), self.Tf)
        if s == 'adaptive':
            rate = float(np.mean(win)) if win else 1.0
            if   rate > 0.90: factor = 0.990
            elif rate > 0.50: factor = 0.997
            elif rate > 0.15: factor = 0.999
            else:             factor = 0.995
            return max(T*factor, self.Tf)
        return T

    def _auto_T0(self, func, lb, ub, d, n):
        pts  = self._random_population(lb, ub, n)
        vals = np.array([func(p) for p in pts])
        avg  = float(np.abs(np.diff(vals)).mean()) or 1.0
        return avg/(-math.log(0.80)+1e-300)
