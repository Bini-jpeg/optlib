"""Nelder-Mead Simplex — Nelder & Mead (1965); Gao & Han (2012) adaptive variant."""
from __future__ import annotations
import time
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class NelderMead(BaseOptimizer):
    def __init__(self, x0=None, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                 tol_fun=1e-8, tol_x=1e-8, adaptive=False,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.x0=x0; self.alpha=alpha; self.gamma=gamma; self.rho=rho
        self.sigma=sigma; self.tol_fun=tol_fun; self.tol_x=tol_x
        self.adaptive=adaptive; self._name="NelderMead"

    def optimize(self, func, bounds, max_iter=10_000, max_fev=None, **_):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d = len(lb)
        history = OptimizationHistory(); nfev = 0
        budget  = max_fev or (max_iter*(d+1))

        if self.adaptive:
            al,gm,rh,sh = 1.0, 1+2/d, 0.75-0.5/d, 1-1/d
        else:
            al,gm,rh,sh = self.alpha,self.gamma,self.rho,self.sigma

        x0 = np.clip(self.x0 if self.x0 is not None
                     else lb+(ub-lb)*self.rng.random(d), lb, ub)
        sim = np.empty((d+1,d)); sim[0] = x0
        for k in range(d):
            p=x0.copy(); p[k]+=0.05*(ub[k]-lb[k]); sim[k+1]=np.clip(p,lb,ub)
        fsim = np.array([func(s) for s in sim]); nfev += d+1
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, budget): break
            order=np.argsort(fsim); sim=sim[order]; fsim=fsim[order]
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, fsim[0], fsim.mean(), fsim.std(),
                               elapsed=elapsed, nfev=nfev)
            self._log(it, fsim[0], nfev=nfev, elapsed=elapsed, f_worst=fsim[-1])
            if (max(abs(fsim[1:]-fsim[0])) < self.tol_fun and
                    max(np.max(abs(sim[1:]-sim[0]),axis=1)) < self.tol_x): break
            xbar = sim[:-1].mean(axis=0)
            xr   = np.clip(xbar+al*(xbar-sim[-1]), lb, ub)
            fxr  = func(xr); nfev += 1
            if fsim[0] <= fxr < fsim[-2]:
                sim[-1]=xr; fsim[-1]=fxr; continue
            if fxr < fsim[0]:
                xe = np.clip(xbar+gm*(xr-xbar), lb, ub); fxe=func(xe); nfev+=1
                sim[-1],fsim[-1] = (xe,fxe) if fxe<fxr else (xr,fxr); continue
            if fxr < fsim[-1]:
                xc=np.clip(xbar+rh*(xr-xbar),lb,ub); fxc=func(xc); nfev+=1
                if fxc<=fxr: sim[-1]=xc; fsim[-1]=fxc; continue
            else:
                xcc=np.clip(xbar+rh*(sim[-1]-xbar),lb,ub); fxcc=func(xcc); nfev+=1
                if fxcc<fsim[-1]: sim[-1]=xcc; fsim[-1]=fxcc; continue
            for k in range(1,d+1):
                sim[k]=np.clip(sim[0]+sh*(sim[k]-sim[0]),lb,ub)
                fsim[k]=func(sim[k]); nfev+=1

        order=np.argsort(fsim)
        return OptimizationResult(x=sim[order[0]], fun=fsim[order[0]], nfev=nfev, nit=it,
            success=True, message="NelderMead completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)
