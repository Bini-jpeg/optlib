"""
Demo 5 — Binary Feature Selection (equal evaluation budget).

Problem : Select a minimal subset of the 30 breast-cancer features that
          maximises Random Forest cross-validation accuracy.
Encoding: binary chromosome — bit j = 1 means feature j is included.
Budget  : BUDGET = 400 CV evaluations per algorithm.
          One evaluation = one 5-fold CV run (~0.15 s with RF n=30).

Comparison: Binary GA vs Binary PSO vs Random Search.

Outputs (in outputs/demo05/):
  convergence_vs_nfev.png
  timing.png
  feature_mask_<algo>.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets        import load_breast_cancer
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler

import optlib as oz
from optlib.base import OptimizationHistory, OptimizationResult
from demos.demo_utils import (run_suite, print_summary_table,
                               plot_convergence_nfev, plot_timing,
                               plot_quality_vs_time, PALETTE)

OUTPUT  = Path("outputs/demo05"); OUTPUT.mkdir(parents=True, exist_ok=True)
BUDGET  = 400   # CV evaluations per algorithm
N_RUNS  = 3
SEED0   = 42
LAMBDA  = 0.05  # feature-count penalty weight

# ── Dataset ────────────────────────────────────────────────────────────────

data       = load_breast_cancer()
X_raw, y   = data.data, data.target
feat_names = data.feature_names
X          = StandardScaler().fit_transform(X_raw)
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_FEAT     = X.shape[1]

def evaluate_mask(mask: np.ndarray) -> float:
    """Binary mask → −accuracy + penalty*fraction_used. Lower = better."""
    sel = np.where(mask > 0.5)[0]
    if len(sel) == 0:
        return 1.0
    clf    = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)
    scores = cross_val_score(clf, X[:, sel], y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(1.0 - scores.mean() + LAMBDA * len(sel) / N_FEAT)


# ── Binary GA ──────────────────────────────────────────────────────────────

class BinaryGA(oz.BaseOptimizer):
    def __init__(self, pop_size=40, p_mut=0.015, elite=3,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size=pop_size; self.p_mut=p_mut; self.elite=elite
        self._name="BinaryGA"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, **_):
        t0 = time.perf_counter()
        n       = N_FEAT
        history = OptimizationHistory(); nfev = 0
        budget  = max_fev or max_iter

        pop  = (self.rng.random((self.pop_size, n)) > 0.5).astype(float)
        fits = np.array([func(c) for c in pop]); nfev += self.pop_size
        best_c = pop[np.argmin(fits)].copy(); best_f = fits.min()
        it = 0

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, budget): break
            order  = np.argsort(fits)
            elites = pop[order[:self.elite]].copy(); ef = fits[order[:self.elite]].copy()
            offs   = []
            while len(offs) < self.pop_size - self.elite:
                ix = self.rng.integers(0, self.pop_size, 6)
                p1 = ix[np.argmin(fits[ix[:3]])]; p2 = ix[3+np.argmin(fits[ix[3:]])]
                a,b = sorted(self.rng.integers(0,n,2))
                c1  = np.concatenate([pop[p1,:a], pop[p2,a:b], pop[p1,b:]])
                mask = self.rng.random(n) < self.p_mut; c1[mask] = 1-c1[mask]
                offs.append(c1)
            offs  = np.array(offs[:self.pop_size-self.elite])
            of    = np.array([func(c) for c in offs]); nfev += len(offs)
            pop   = np.vstack([elites,offs]); fits = np.concatenate([ef,of])
            idx   = np.argmin(fits)
            if fits[idx] < best_f: best_f=fits[idx]; best_c=pop[idx].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed,
                      n_feat=int(best_c.sum()))

        return OptimizationResult(x=best_c, fun=best_f, nfev=nfev, nit=it,
            success=True, message="BinaryGA done",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter()-t0)


# ── Binary PSO ─────────────────────────────────────────────────────────────

class BinaryPSO(oz.BaseOptimizer):
    """
    Kennedy & Eberhart (1997) binary PSO.
    Velocity interpreted as probability of bit = 1 via sigmoid.
    """
    def __init__(self, pop_size=30, w=0.7, c1=1.5, c2=1.5,
                 seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self.pop_size=pop_size; self.w=w; self.c1=c1; self.c2=c2
        self._name="BinaryPSO"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, **_):
        t0 = time.perf_counter()
        n       = N_FEAT
        history = OptimizationHistory(); nfev = 0
        budget  = max_fev or max_iter

        pos  = (self.rng.random((self.pop_size, n)) > 0.5).astype(float)
        vel  = self.rng.uniform(-4, 4, (self.pop_size, n))
        fits = np.array([func(p) for p in pos]); nfev += self.pop_size
        pbest = pos.copy(); pf = fits.copy()
        gi = np.argmin(pf); gbest = pbest[gi].copy(); gbest_f = pf[gi]
        it = 0

        def sigmoid(v): return 1.0/(1.0+np.exp(-np.clip(v,-20,20)))

        for it in range(1, max_iter+1):
            if not self._budget_ok(nfev, budget): break
            r1=self.rng.random((self.pop_size,n)); r2=self.rng.random((self.pop_size,n))
            vel = self.w*vel + self.c1*r1*(pbest-pos) + self.c2*r2*(gbest-pos)
            vel = np.clip(vel, -6, 6)
            pos = (self.rng.random((self.pop_size,n)) < sigmoid(vel)).astype(float)
            fits = np.array([func(p) for p in pos]); nfev += self.pop_size
            imp  = fits < pf; pbest[imp]=pos[imp]; pf[imp]=fits[imp]
            gi   = np.argmin(pf)
            if pf[gi] < gbest_f: gbest_f=pf[gi]; gbest=pbest[gi].copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, gbest_f, fits.mean(), fits.std(),
                               elapsed=elapsed, nfev=nfev)
            self._log(it, gbest_f, nfev=nfev, elapsed=elapsed,
                      n_feat=int(gbest.sum()))

        return OptimizationResult(x=gbest, fun=gbest_f, nfev=nfev, nit=it,
            success=True, message="BinaryPSO done",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter()-t0)


# ── Random search baseline ─────────────────────────────────────────────────

class BinaryRandom(oz.BaseOptimizer):
    def __init__(self, seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self._name = "RandomSearch"

    def optimize(self, func, bounds, max_iter=500, max_fev=None, **_):
        t0=time.perf_counter(); n=N_FEAT
        history=OptimizationHistory(); nfev=0; budget=max_fev or max_iter
        best_x=np.ones(n); best_f=np.inf; it=0
        for it in range(1, budget+1):
            if not self._budget_ok(nfev, budget): break
            x  = (self.rng.random(n) > 0.5).astype(float)
            fx = func(x); nfev += 1
            if fx < best_f: best_f=fx; best_x=x.copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, elapsed=elapsed, nfev=nfev)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed)
        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=False, message="done", history=history,
            algorithm=self._name, elapsed=time.perf_counter()-t0)


def make_algos(seed):
    return {
        "BinaryGA":    BinaryGA(pop_size=30, seed=seed),
        "BinaryPSO":   BinaryPSO(pop_size=20, seed=seed),
        "RandomSearch":BinaryRandom(seed=seed),
    }


def main():
    # Baseline: all features
    clf0 = RandomForestClassifier(n_estimators=50, random_state=SEED0, n_jobs=1)
    base_acc = cross_val_score(clf0, X, y, cv=cv, scoring="accuracy").mean()
    print(f"Feature Selection — Breast Cancer  (d={N_FEAT}, budget={BUDGET} evals)\n")
    print(f"  Baseline (all features) CV accuracy = {base_acc*100:.2f}%")
    print(f"  Objective: minimise (1−accuracy) + {LAMBDA}·(fraction used)\n")

    # Fake bounds — not used by binary algos but required by run_suite
    dummy_bounds = [(0, 1)] * N_FEAT

    results = run_suite(make_algos, evaluate_mask, dummy_bounds,
                        budget=BUDGET, n_runs=N_RUNS, seed0=SEED0)

    print_summary_table(results, title=f"Feature Selection (budget={BUDGET})")

    # ── Print best mask for each algo ─────────────────────────────────────
    print("\n  Selected feature counts (best run per algo):")
    for name, r in results.items():
        bi   = int(np.argmin(r['finals']))
        # We don't store x in results; re-run to get mask
        algo = make_algos(SEED0)[name]
        res  = algo.optimize(evaluate_mask, dummy_bounds, max_fev=BUDGET)
        sel  = np.where(res.x > 0.5)[0]
        clf2 = RandomForestClassifier(n_estimators=50, random_state=SEED0)
        acc2 = cross_val_score(clf2, X[:,sel], y, cv=cv, scoring="accuracy").mean()
        print(f"    {name:<16}  features={len(sel):>2}/{N_FEAT}"
              f"  CV acc={acc2*100:.2f}%"
              f"  obj={r['finals'][bi]:.4f}")

        # Feature mask bar
        fig, ax = plt.subplots(figsize=(12,3))
        colors  = ["steelblue" if i in sel else "lightgray" for i in range(N_FEAT)]
        ax.bar(range(N_FEAT), res.x, color=colors, edgecolor="none")
        ax.set_xticks(range(N_FEAT))
        ax.set_xticklabels([f[:9] for f in feat_names], rotation=90, fontsize=6)
        ax.set_ylabel("Selected"); ax.set_ylim(0,1.2)
        ax.set_title(f"{name} — {len(sel)}/{N_FEAT} features  (CV={acc2*100:.2f}%)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(OUTPUT / f"feature_mask_{name}.png"), dpi=150); plt.close(fig)

    # ── Convergence & timing plots ─────────────────────────────────────────
    fig = plot_convergence_nfev(
        results, BUDGET,
        title=f"Feature Selection Convergence  (budget={BUDGET}, {N_RUNS} runs)",
        log_y=False,
        save_path=str(OUTPUT / "convergence_vs_nfev.png")); plt.close(fig)

    fig = plot_timing(results, title="Feature Selection — Timing",
                      save_path=str(OUTPUT / "timing.png")); plt.close(fig)

    fig = plot_quality_vs_time(results,
                               title="Feature Selection — Quality vs Time",
                               log_y=False,
                               save_path=str(OUTPUT / "quality_vs_time.png")); plt.close(fig)

    print(f"\nDemo 5 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
