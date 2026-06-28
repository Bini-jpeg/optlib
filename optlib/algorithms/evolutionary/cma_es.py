"""CMA-ES with IPOP restarts.  Reference: Hansen (2016) arXiv:1604.00772."""
from __future__ import annotations
import time
from typing import Callable, List, Optional, Union
import numpy as np
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult


class CMAES(BaseOptimizer):
    def __init__(self, sigma0=None, pop_size=None, n_restarts=0,
                 tol_fun=1e-11, tol_x=1e-11, max_fev=None,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.sigma0     = sigma0
        self._lam       = pop_size
        self.n_restarts = n_restarts
        self.tol_fun    = tol_fun
        self.tol_x      = tol_x
        self._name      = "CMA-ES"

    def optimize(self, func, bounds, max_iter=5000, max_fev=None, **_):
        t0      = time.perf_counter()
        lb, ub  = self._parse_bounds(bounds)
        d       = len(lb)
        history = OptimizationHistory()
        nfev    = 0
        budget  = max_fev or int(1e5 * d)

        g_best_x = None; g_best_f = np.inf; total_it = 0

        for restart in range(self.n_restarts + 1):
            s0 = (self.sigma0 or 0.3*np.mean(ub-lb)) * (2.0**restart)
            lam = (self._lam or 4+int(3*np.log(d))) * (2**restart)
            bx, bf, nf, ni = self._run(
                func, lb, ub, d, lam, s0,
                max_iter-total_it, budget-nfev, history, t0)
            nfev += nf; total_it += ni
            if bf < g_best_f: g_best_f, g_best_x = bf, bx
            if g_best_f < self.tol_fun or nfev >= budget: break

        return OptimizationResult(
            x=g_best_x, fun=g_best_f, nfev=nfev, nit=total_it,
            success=g_best_f < self.tol_fun, message="CMA-ES completed",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter()-t0)

    def _run(self, func, lb, ub, d, lam, sigma, max_iter, budget, history, t0_global):
        mu    = lam // 2
        nfev  = 0
        raw_w = np.array([np.log(mu+0.5)-np.log(i+1) for i in range(mu)])
        w     = raw_w / raw_w.sum()
        mu_eff = 1.0/np.dot(w,w)

        c_s  = (mu_eff+2)/(d+mu_eff+5)
        d_s  = 1+2*max(0,np.sqrt((mu_eff-1)/(d+1))-1)+c_s
        c_c  = (4+mu_eff/d)/(d+4+2*mu_eff/d)
        c1   = 2/((d+1.3)**2+mu_eff)
        c_mu = min(1-c1, 2*(mu_eff-2+1/mu_eff)/((d+2)**2+mu_eff))
        chi  = d**0.5*(1-1/(4*d)+1/(21*d**2))

        m    = lb + self.rng.random(d)*(ub-lb)
        p_c  = np.zeros(d); p_s = np.zeros(d)
        C    = np.eye(d); B = np.eye(d); D = np.ones(d)
        invC = np.eye(d); ev_it = 0

        best_x = m.copy(); best_f = np.inf; hist: list[float] = []
        it = 0

        for it in range(1, max_iter+1):
            if nfev >= budget: break

            zs  = self.rng.standard_normal((lam, d))
            ys  = (B@(D[:,None]*(B.T@zs.T))).T
            xs  = self._bounce(m + sigma*ys, lb, ub)
            fs  = np.array([func(x) for x in xs]); nfev += lam

            order = np.argsort(fs)
            if fs[order[0]] < best_f:
                best_f = fs[order[0]]; best_x = xs[order[0]].copy()
            hist.append(float(best_f))

            y_w = np.dot(w, ys[order[:mu]])
            m   = m + sigma*y_w

            p_s = (1-c_s)*p_s + np.sqrt(c_s*(2-c_s)*mu_eff)*(invC@y_w)
            n_ps = np.linalg.norm(p_s)
            hs   = n_ps/np.sqrt(1-(1-c_s)**(2*(it+1))) < (1.4+2/(d+1))*chi

            p_c = (1-c_c)*p_c + hs*np.sqrt(c_c*(2-c_c)*mu_eff)*y_w

            y_mu = ys[order[:mu]]
            rank_mu = sum(w[k]*np.outer(y_mu[k],y_mu[k]) for k in range(mu))
            C = ((1-c1-c_mu)*C
                 + c1*(np.outer(p_c,p_c)+(1-hs)*c_c*(2-c_c)*C)
                 + c_mu*rank_mu)
            sigma *= np.exp(c_s/d_s*(n_ps/chi-1))
            sigma  = float(np.clip(sigma, 1e-12, 1e4))

            if it - ev_it > 1/(c1+c_mu)/d/10:
                C = np.triu(C)+np.triu(C,1).T
                D2,B = np.linalg.eigh(C)
                D    = np.sqrt(np.maximum(D2,1e-20))
                invC = B@np.diag(1/D)@B.T
                ev_it = it

            elapsed = time.perf_counter()-t0_global
            if self.store_history:
                history.record(it, best_f, fs.mean(), fs.std(),
                               elapsed=elapsed, nfev=nfev, sigma=sigma)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, sigma=sigma)

            if best_f < self.tol_fun: break
            n_tail = 10+int(30*d/lam)
            if len(hist)>=n_tail and max(hist[-n_tail:])-min(hist[-n_tail:])<self.tol_fun: break
            if sigma*np.max(D) < self.tol_x: break
            if np.max(np.diag(C)) > 1e14: break

        return best_x, best_f, nfev, it
