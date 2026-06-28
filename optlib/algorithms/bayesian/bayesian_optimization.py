"""Bayesian Optimisation — GP surrogate with EI / UCB / PI acquisition."""
from __future__ import annotations
import time
from typing import Callable, List, Optional, Union
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
from optlib.base import BaseOptimizer, OptimizationHistory, OptimizationResult
from optlib.preprocessing.preprocessing import latin_hypercube_sample


# ── Gaussian Process ────────────────────────────────────────────────────────

class GaussianProcess:
    """ARD RBF kernel GP. hyperparameters optimised via log-marginal-likelihood."""
    def __init__(self, noise=1e-6, n_restarts=5):
        self.noise = noise; self.n_restarts = n_restarts
        self._X = self._y = self._alpha = self._L = None

    @staticmethod
    def _kernel(X1, X2, sv, ls):
        diff = (X1[:,None,:]-X2[None,:,:]) / ls
        return sv*np.exp(-0.5*(diff**2).sum(-1))

    def fit(self, X, y):
        self._X = X.copy()
        self._ym, self._ys = y.mean(), y.std()+1e-10
        self._y = (y-self._ym)/self._ys
        d = X.shape[1]; best_lml = -np.inf; best_th = np.zeros(d+2)
        for _ in range(self.n_restarts):
            th0 = (np.array([np.log(np.var(self._y)+1e-8)]+[0.0]*d+[np.log(self.noise)])
                   + np.random.default_rng().standard_normal(d+2)*0.5)
            r = minimize(lambda t: -self._lml(t), th0, method='L-BFGS-B',
                         options={'maxiter':100,'ftol':1e-6})
            if -r.fun > best_lml: best_lml=-r.fun; best_th=r.x
        self._build(best_th)
        return self

    def _build(self, th):
        sv,ls,nv = self._decode(th)
        K  = self._kernel(self._X,self._X,sv,ls)+(nv+1e-6)*np.eye(len(self._X))
        self._L  = cho_factor(K, lower=True)
        self._al = cho_solve(self._L, self._y)
        self._sv, self._ls = sv, ls

    def _decode(self, th):
        d  = self._X.shape[1]
        return float(np.exp(th[0])), np.exp(th[1:1+d]), float(np.exp(th[-1]))

    def _lml(self, th):
        try:
            sv,ls,nv = self._decode(th)
            K  = self._kernel(self._X,self._X,sv,ls)+(nv+1e-6)*np.eye(len(self._X))
            L  = cho_factor(K, lower=True)
            al = cho_solve(L, self._y)
            return float(-0.5*np.dot(self._y,al)-np.sum(np.log(np.diag(L[0])))
                         -0.5*len(self._y)*np.log(2*np.pi))
        except Exception: return -1e30

    def predict(self, Xn):
        Ks  = self._kernel(Xn,self._X,self._sv,self._ls)
        Kss = self._kernel(Xn,Xn,self._sv,self._ls)
        mu  = Ks@self._al
        v   = cho_solve(self._L, Ks.T)
        var = np.maximum(np.diag(Kss)-np.einsum('ij,ij->j',Ks.T,v), 1e-12)
        return mu*self._ys+self._ym, np.sqrt(var)*self._ys


# ── Bayesian Optimisation ──────────────────────────────────────────────────

class BayesianOptimization(BaseOptimizer):
    """
    GP-based Bayesian Optimisation.
    Best for expensive functions (< ~500 evaluations total budget).

    Parameters
    ----------
    n_init        : LHS warm-up evaluations (default 5)
    acquisition   : 'ei' | 'ucb' | 'pi'
    kappa         : UCB trade-off coefficient (default 2.576)
    xi            : EI/PI jitter (default 0.01)
    gp_restarts   : GP hyperparameter fitting restarts (default 3)
    acq_restarts  : multi-start acquisition optimisation (default 10)
    """
    def __init__(self, n_init=5, acquisition='ei', kappa=2.576, xi=0.01,
                 gp_noise=1e-5, gp_restarts=3, acq_restarts=10,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.n_init=n_init; self.acquisition=acquisition
        self.kappa=kappa; self.xi=xi; self.gp_noise=gp_noise
        self.gp_restarts=gp_restarts; self.acq_restarts=acq_restarts
        self._name=f"BayesOpt-{acquisition}"

    def optimize(self, func, bounds, max_iter=50, max_fev=None, tol=1e-8):
        t0 = time.perf_counter()
        lb, ub = self._parse_bounds(bounds)
        d = len(lb); history = OptimizationHistory(); nfev = 0
        budget = max_fev or (self.n_init + max_iter)
        b_arr  = np.column_stack([lb,ub])
        gp     = GaussianProcess(noise=self.gp_noise, n_restarts=self.gp_restarts)

        X = latin_hypercube_sample(self.n_init, d, b_arr, seed=self.seed)
        Y = np.array([func(x) for x in X]); nfev += self.n_init
        best_idx = np.argmin(Y); best_x=X[best_idx].copy(); best_f=Y[best_idx]
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, budget): break
            try: gp.fit(X, Y)
            except Exception:
                xn = self._random_population(lb,ub,1)[0]; yn=func(xn); nfev+=1
                X=np.vstack([X,xn]); Y=np.append(Y,yn); continue

            xn  = self._opt_acq(gp, lb, ub, d, best_f)
            yn  = func(xn); nfev += 1
            X   = np.vstack([X,xn]); Y = np.append(Y,yn)
            if yn < best_f: best_f=yn; best_x=xn.copy()

            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, float(np.mean(Y)), float(np.std(Y)),
                               elapsed=elapsed, nfev=nfev)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed, n_pts=len(Y))
            if best_f < tol: break

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=best_f<tol, message=f"BayesOpt-{self.acquisition} completed",
            history=history, algorithm=self._name, elapsed=time.perf_counter()-t0)

    def _acq(self, x2d, gp, fb):
        mu, std = gp.predict(x2d); std = np.maximum(std, 1e-12)
        acq = self.acquisition
        if acq=='ei':
            z = (fb-mu-self.xi)/std; return -(std*(z*norm.cdf(z)+norm.pdf(z)))
        if acq=='ucb': return -(mu-self.kappa*std)
        if acq=='pi':  z=(fb-mu-self.xi)/std; return -norm.cdf(z)
        raise ValueError(f"Unknown acquisition: {acq}")

    def _opt_acq(self, gp, lb, ub, d, fb):
        best_v=np.inf; best_x=lb+(ub-lb)*0.5
        for x0 in self._random_population(lb, ub, self.acq_restarts):
            try:
                r = minimize(lambda x: float(self._acq(x[None,:],gp,fb)),
                             x0, method='L-BFGS-B', bounds=list(zip(lb,ub)),
                             options={'maxiter':50,'ftol':1e-6})
                if r.fun < best_v: best_v=r.fun; best_x=r.x
            except Exception: continue
        return np.clip(best_x, lb, ub)
