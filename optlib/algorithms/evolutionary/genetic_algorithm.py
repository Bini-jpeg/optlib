"""
Genetic Algorithm — real-valued encoding.
SBX crossover + polynomial mutation + tournament selection + elitism.
"""
from __future__ import annotations
import time
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class GeneticAlgorithm(BaseOptimizer):
    """
    Parameters
    ----------
    pop_size    : population size (default 100)
    eta_c       : SBX crossover distribution index (default 20)
    eta_m       : polynomial mutation distribution index (default 20)
    p_cross     : crossover probability (default 0.9)
    p_mut       : per-gene mutation probability (None → 1/d)
    elite_frac  : fraction of population carried as elites (default 0.05)
    """
    def __init__(self, pop_size=100, eta_c=20.0, eta_m=20.0,
                 p_cross=0.9, p_mut=None, elite_frac=0.05,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size   = pop_size
        self.eta_c      = eta_c
        self.eta_m      = eta_m
        self.p_cross    = p_cross
        self.p_mut      = p_mut
        self.elite_frac = elite_frac
        self._name      = "GA"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, tol=1e-8, n_tol_check=30):
        t0      = time.perf_counter()
        lb, ub  = self._parse_bounds(bounds)
        d       = len(lb)
        p_mut   = self.p_mut if self.p_mut is not None else 1.0 / d
        n_elite = max(1, int(self.elite_frac * self.pop_size))
        history = OptimizationHistory()
        nfev    = 0

        pop  = self._random_population(lb, ub, self.pop_size)
        fits = np.array([func(ind) for ind in pop]); nfev += self.pop_size

        best_idx = np.argmin(fits)
        best_x   = pop[best_idx].copy()
        best_f   = fits[best_idx]
        it       = 0

        for it in range(1, max_iter + 1):
            if not self._budget_ok(nfev, max_fev):
                break
            order  = np.argsort(fits)
            elites = pop[order[:n_elite]].copy()
            e_fits = fits[order[:n_elite]].copy()

            n_off     = self.pop_size - n_elite
            offspring = np.empty((n_off, d))
            o_idx     = 0
            while o_idx < n_off:
                p1 = self._tournament(fits); p2 = self._tournament(fits)
                x1, x2 = pop[p1].copy(), pop[p2].copy()
                if self.rng.random() < self.p_cross:
                    x1, x2 = self._sbx(x1, x2, lb, ub)
                x1 = self._poly_mut(x1, lb, ub, p_mut)
                offspring[o_idx] = x1; o_idx += 1
                if o_idx < n_off:
                    x2 = self._poly_mut(x2, lb, ub, p_mut)
                    offspring[o_idx] = x2; o_idx += 1

            offspring = self._clip(offspring, lb, ub)
            o_fits    = np.array([func(ind) for ind in offspring])
            nfev     += n_off

            pop  = np.vstack([elites, offspring])
            fits = np.concatenate([e_fits, o_fits])

            idx = np.argmin(fits)
            if fits[idx] < best_f:
                best_f = fits[idx]; best_x = pop[idx].copy()

            elapsed = time.perf_counter() - t0
            if self.store_history:
                history.record(it, best_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, mean=fits.mean())

            if 0.0 <= best_f < tol:
                break
            if it >= n_tol_check:
                recent = history.best_values[-n_tol_check:]
                if max(recent) - min(recent) < tol:
                    break

        return OptimizationResult(
            x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=0.0 <= best_f < tol, message="GA completed",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter() - t0)

    def _tournament(self, fits, k=3):
        idx = self.rng.integers(0, len(fits), k)
        return int(idx[np.argmin(fits[idx])])

    def _sbx(self, x1, x2, lb, ub):
        c1, c2 = x1.copy(), x2.copy()
        eta    = self.eta_c
        for j in range(len(x1)):
            if abs(x1[j] - x2[j]) < 1e-14 or self.rng.random() > 0.5:
                continue
            y1, y2 = min(x1[j], x2[j]), max(x1[j], x2[j])
            yl, yu = lb[j], ub[j]
            for sign, ya in [(1, y1), (-1, y2)]:
                beta = 1.0 + 2.0 * (ya - yl if sign==1 else yu - ya) / (y2 - y1)
                alpha = 2.0 - beta**(-(eta + 1.0))
                u = self.rng.random()
                betaq = (u * alpha)**(1/(eta+1)) if u <= 1/alpha \
                        else (1/(2 - u*alpha))**(1/(eta+1))
                mid = 0.5*(y1+y2)
                if sign == 1: c1[j] = mid - 0.5*betaq*(y2-y1)
                else:         c2[j] = mid + 0.5*betaq*(y2-y1)
            if self.rng.random() > 0.5: c1[j], c2[j] = c2[j], c1[j]
        return c1, c2

    def _poly_mut(self, x, lb, ub, p):
        y, eta = x.copy(), self.eta_m
        for j in range(len(x)):
            if self.rng.random() > p: continue
            lo, hi = lb[j], ub[j]
            delta  = min(y[j]-lo, hi-y[j]) / (hi-lo)
            u      = self.rng.random()
            dq = ((2*u+(1-2*u)*(1-delta)**(eta+1))**(1/(eta+1))-1) if u < 0.5 \
                 else (1-(2*(1-u)+2*(u-0.5)*(1-delta)**(eta+1))**(1/(eta+1)))
            y[j] = np.clip(y[j] + dq*(hi-lo), lo, hi)
        return y
