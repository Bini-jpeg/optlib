"""
Demo 3 — Hyperparameter Optimisation (equal evaluation budget).

Budget : BUDGET = 80 cross-validation evaluations per algorithm.
         One evaluation ≈ 0.3–0.5 s (5-fold CV of an SVM).
         This is why BO is designed for: it extracts maximum information
         per expensive evaluation.

Algorithms: BayesOpt-EI, BayesOpt-UCB, PSO, CEM, Random Search.

Outputs (in outputs/demo03/):
  convergence_vs_nfev.png
  timing.png
  quality_vs_time.png
  landscape_svm.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets        import load_breast_cancer
from sklearn.svm             import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler

import optlib as oz
from optlib.base import OptimizationHistory, OptimizationResult
from demos.demo_utils import (run_suite, print_summary_table,
                               plot_convergence_nfev, plot_timing,
                               plot_quality_vs_time)

OUTPUT  = Path("outputs/demo03"); OUTPUT.mkdir(parents=True, exist_ok=True)
BUDGET  = 80     # CV evaluations per algorithm (expensive: ~0.4s each)
N_RUNS  = 3
SEED0   = 42

# ── Dataset ────────────────────────────────────────────────────────────────

X_raw, y  = load_breast_cancer(return_X_y=True)
X         = StandardScaler().fit_transform(X_raw)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(params: np.ndarray) -> float:
    """Minimise −accuracy.  params = [log10(C), log10(gamma)]."""
    C, gamma = 10.0**params[0], 10.0**params[1]
    svm      = SVC(C=C, gamma=gamma, kernel="rbf", random_state=42)
    scores   = cross_val_score(svm, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(-scores.mean())

# Search: log10(C) ∈ [-2, 3],  log10(gamma) ∈ [-5, 1]
BOUNDS = [(-2.0, 3.0), (-5.0, 1.0)]


# ── Random Search baseline (wraps as a BaseOptimizer) ─────────────────────

class RandomSearch(oz.BaseOptimizer):
    def __init__(self, seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self._name = "RandomSearch"

    def optimize(self, func, bounds, max_iter=1000, max_fev=None, **_):
        import time; t0 = time.perf_counter()
        lb, ub  = self._parse_bounds(bounds)
        history = OptimizationHistory()
        nfev    = 0; budget = max_fev or max_iter
        best_x  = (lb+ub)/2; best_f = np.inf; it = 0

        for it in range(1, budget+1):
            if not self._budget_ok(nfev, budget): break
            x  = self._random_population(lb, ub, 1)[0]
            fx = func(x); nfev += 1
            if fx < best_f: best_f=fx; best_x=x.copy()
            elapsed = time.perf_counter()-t0
            if self.store_history:
                history.record(it, best_f, elapsed=elapsed, nfev=nfev)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed)

        return OptimizationResult(x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=False, message="RandomSearch done",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter()-t0)


def make_algos(seed):
    return {
        "BayesOpt-EI":  oz.BayesianOptimization(n_init=5, acquisition="ei",
                            gp_restarts=2, acq_restarts=8, seed=seed),
        "BayesOpt-UCB": oz.BayesianOptimization(n_init=5, acquisition="ucb",
                            gp_restarts=2, acq_restarts=8, seed=seed),
        "PSO":          oz.PSO(pop_size=10, seed=seed),
        "CEM":          oz.CrossEntropyMethod(pop_size=15, seed=seed),
        "RandomSearch": RandomSearch(seed=seed),
    }


def main():
    print(f"SVM HPO — Breast Cancer  (budget={BUDGET} CV evals per algo)\n")

    results = run_suite(make_algos, objective, BOUNDS,
                        budget=BUDGET, n_runs=N_RUNS, seed0=SEED0, verbose=True)

    # Convert negative accuracy back for display
    for r in results.values():
        r['finals'] = [-f for f in r['finals']]   # now positive accuracy

    print_summary_table(
        {n: {**r, 'finals': [-f for f in r['finals']]} for n,r in results.items()},
        title=f"SVM HPO — CV Accuracy (budget={BUDGET} evals)")

    # Restore for plotting (we want to show accuracy going UP)
    # Re-negate: history stores negative accuracy values
    fig, ax = plt.subplots(figsize=(10, 5))
    from demos.demo_utils import PALETTE
    for idx, (name, r) in enumerate(results.items()):
        color = PALETTE[idx % len(PALETTE)]
        traces = []
        for h in r['histories']:
            if not h.nfev_values: continue
            xs = [0] + list(h.nfev_values)
            ys = [-v*100 for v in ([h.best_values[0]]+list(h.best_values))]
            xs.append(BUDGET); ys.append(ys[-1])
            traces.append((np.array(xs), np.array(ys)))
        if not traces: continue
        grid = np.linspace(0, BUDGET, 300)
        mat  = np.array([np.interp(grid, tr[0], tr[1]) for tr in traces])
        med  = np.median(mat, axis=0)
        p25  = np.percentile(mat, 25, axis=0)
        p75  = np.percentile(mat, 75, axis=0)
        ax.plot(grid, med, color=color, linewidth=2, label=name)
        ax.fill_between(grid, p25, p75, alpha=0.12, color=color)

    ax.set_xlabel("CV Evaluations (nfev)", fontsize=12)
    ax.set_ylabel("Best CV Accuracy (%)", fontsize=12)
    ax.set_title(f"HPO Convergence — SVM on Breast Cancer  (budget={BUDGET}, {N_RUNS} runs median)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, BUDGET)
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "convergence_vs_nfev.png"), dpi=150); plt.close(fig)

    # Restore original results for timing plots
    for r in results.values():
        r['finals'] = [-f for f in r['finals']]  # back to negative accuracy

    fig = plot_timing(results, title="HPO — Algorithm Timing",
                      save_path=str(OUTPUT / "timing.png")); plt.close(fig)
    fig = plot_quality_vs_time(
        results,
        title="HPO Quality (−accuracy) vs Time",
        log_y=False,
        save_path=str(OUTPUT / "quality_vs_time.png")); plt.close(fig)

    # ── Accuracy landscape ─────────────────────────────────────────────────
    print("\nComputing accuracy landscape (grid)...")
    G  = 18; cs=np.linspace(-2,3,G); gs=np.linspace(-5,1,G)
    ZZ = np.array([[-objective(np.array([c,g]))*100 for c in cs] for g in gs])
    fig2, ax2 = plt.subplots(figsize=(8,6))
    cf = ax2.contourf(cs, gs, ZZ, levels=20, cmap="RdYlGn")
    plt.colorbar(cf, ax=ax2, label="CV Accuracy (%)")
    for idx, (name, r) in enumerate(results.items()):
        # best run across seeds
        bi   = int(np.argmin(r['finals']))
        best = r['histories'][bi]
        if best.nfev_values:
            # reconstruct best x from history is not stored; run fresh
            pass
    ax2.set_xlabel("log₁₀(C)"); ax2.set_ylabel("log₁₀(γ)")
    ax2.set_title("SVM Accuracy Landscape", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(str(OUTPUT / "landscape_svm.png"), dpi=150); plt.close(fig2)

    print(f"\nDemo 3 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
