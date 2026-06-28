"""
Differential Evolution — rand1bin / best1bin / ctbest1bin / rand2bin
                         / JADE / SHADE / L-SHADE.
All variants share a single class; strategy is selected by name.
"""
from __future__ import annotations
import time
from typing import Callable, List, Optional, Union
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


def _lehmer_mean(a):
    return float(np.dot(a,a)/(a.sum()+1e-300))


class DifferentialEvolution(BaseOptimizer):
    """
    Parameters
    ----------
    strategy  : 'rand1bin' | 'best1bin' | 'ctbest1bin' | 'rand2bin'
                | 'jade' | 'shade' | 'lshade'
    pop_size  : None → 10*d
    F / CR    : scale factor / crossover rate (ignored by adaptive variants)
    p_best    : fraction of best vectors used in current-to-pbest mutation
    H         : SHADE history size
    """
    STRATEGIES = {"rand1bin","best1bin","ctbest1bin","rand2bin",
                  "jade","shade","lshade"}

    def __init__(self, pop_size=None, strategy="rand1bin",
                 F=0.5, CR=0.9, p_best=0.11, archive_size=1.0, H=10,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.pop_size     = pop_size
        self.strategy     = strategy
        self.F0, self.CR0 = F, CR
        self.p_best       = p_best
        self.archive_size = archive_size
        self.H            = H
        self._name        = f"DE-{strategy}"

    def optimize(self, func, bounds, max_iter=1000, max_fev=None, tol=1e-10):
        t0      = time.perf_counter()
        lb, ub  = self._parse_bounds(bounds)
        d       = len(lb)
        NP      = self.pop_size or max(10*d, 15)
        history = OptimizationHistory()
        nfev    = 0

        pop  = self._random_population(lb, ub, NP)
        fits = np.array([func(x) for x in pop]); nfev += NP

        best_idx = np.argmin(fits)
        best_x   = pop[best_idx].copy()
        best_f   = fits[best_idx]

        archive  = []
        MF       = np.full(self.H, 0.5)
        MCR      = np.full(self.H, 0.5)
        h_idx    = 0
        mu_F     = 0.5
        mu_CR    = 0.5
        NP_init  = NP
        NP_min   = max(4, NP//5)
        it       = 0

        for it in range(1, max_iter + 1):
            if not self._budget_ok(nfev, max_fev):
                break
            S_F, S_CR   = [], []
            new_pop  = pop.copy()
            new_fits = fits.copy()

            for i in range(NP):
                F, CR = self._sample_params(mu_F, mu_CR, MF, MCR)
                v = self._mutate(pop, fits, i, d, lb, ub, F, archive)
                u = self._crossover(pop[i], v, CR, d)
                fu = func(u); nfev += 1
                if fu <= fits[i]:
                    if self.strategy in ("shade","lshade"):
                        archive.append(pop[i].copy())
                    new_pop[i]  = u
                    new_fits[i] = fu
                    if self.strategy in ("jade","shade","lshade"):
                        S_F.append(F); S_CR.append(CR)

            # Archive trim
            max_arch = int(self.archive_size * NP)
            if len(archive) > max_arch:
                drop = self.rng.choice(len(archive), len(archive)-max_arch, replace=False)
                archive = [archive[k] for k in range(len(archive)) if k not in drop]

            if S_F:
                sf = np.array(S_F); sc = np.array(S_CR)
                MF[h_idx]  = _lehmer_mean(sf)
                MCR[h_idx] = float(sc.mean())
                h_idx = (h_idx+1) % self.H
                if self.strategy == "jade":
                    mu_F  = 0.9*mu_F  + 0.1*_lehmer_mean(sf)
                    mu_CR = 0.9*mu_CR + 0.1*float(sc.mean())

            if self.strategy == "lshade":
                NP_new = max(NP_min, int(NP_init-(NP_init-NP_min)*it/max_iter))
                if NP_new < NP:
                    worst = np.argsort(new_fits)[NP_new:]
                    new_pop  = np.delete(new_pop,  worst, axis=0)
                    new_fits = np.delete(new_fits, worst)
                    NP = NP_new

            pop  = new_pop; fits = new_fits
            idx  = np.argmin(fits)
            if fits[idx] < best_f:
                best_f = fits[idx]; best_x = pop[idx].copy()

            elapsed = time.perf_counter() - t0
            if self.store_history:
                history.record(it, best_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev,
                               NP=NP, F=float(np.mean(S_F)) if S_F else self.F0)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, NP=NP)

            if 0.0 <= best_f < tol:
                break

        return OptimizationResult(
            x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=0.0 <= best_f < tol, message=f"DE-{self.strategy} completed",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter() - t0)

    def _sample_params(self, mu_F, mu_CR, MF, MCR):
        if self.strategy == "jade":
            F  = float(np.clip(self.rng.standard_cauchy()*0.1+mu_F, 0, 1))
            CR = float(np.clip(self.rng.normal(mu_CR, 0.1), 0, 1))
        elif self.strategy in ("shade","lshade"):
            ri = self.rng.integers(0, self.H)
            F  = float(np.clip(self.rng.standard_cauchy()*0.1+MF[ri], 0, 1))
            CR = float(np.clip(self.rng.normal(MCR[ri], 0.1), 0, 1))
        else:
            F, CR = self.F0, self.CR0
        return F, CR

    def _mutate(self, pop, fits, i, d, lb, ub, F, archive):
        NP   = len(pop)
        idxs = [x for x in range(NP) if x != i]
        s    = self.strategy
        if s == "rand1bin":
            r1,r2,r3 = self.rng.choice(idxs,3,replace=False)
            v = pop[r1] + F*(pop[r2]-pop[r3])
        elif s == "best1bin":
            best = np.argmin(fits)
            r1,r2 = self.rng.choice(idxs,2,replace=False)
            v = pop[best] + F*(pop[r1]-pop[r2])
        elif s == "ctbest1bin":
            best = np.argmin(fits)
            r1,r2 = self.rng.choice(idxs,2,replace=False)
            v = pop[i] + F*(pop[best]-pop[i]) + F*(pop[r1]-pop[r2])
        elif s == "rand2bin":
            r1,r2,r3,r4,r5 = self.rng.choice(idxs,5,replace=False)
            v = pop[r1] + F*(pop[r2]-pop[r3]) + F*(pop[r4]-pop[r5])
        else:  # jade/shade/lshade — current-to-pbest/1
            n_pb   = max(2, int(self.p_best*NP))
            pb_idx = np.argpartition(fits, n_pb)[:n_pb]
            pbest  = pop[self.rng.choice(pb_idx)]
            r1     = self.rng.choice(idxs)
            pool   = list(pop) + archive
            pool   = [x for j,x in enumerate(pool) if j != i and j != r1]
            r2_vec = pool[self.rng.integers(0, len(pool))]
            v = pop[i] + F*(pbest-pop[i]) + F*(pop[r1]-r2_vec)
        return self._bounce(v, lb, ub)

    def _crossover(self, x, v, CR, d):
        j_rand = self.rng.integers(0, d)
        mask   = self.rng.random(d) < CR
        mask[j_rand] = True
        return np.where(mask, v, x)
