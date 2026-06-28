"""
Demo 3 — Hyperparameter Optimisation (equal evaluation budget).

Budget  : BUDGET = 80 cross-validation evaluations per algorithm.
Objective: 1 − CV_accuracy  ∈ [0, 1]  (positive, minimised).
           This makes the tol stopping criterion meaningful (tol=1e-4
           ≈ 0.01 % accuracy improvement threshold).

Why this objective?  Using −accuracy would cause every algorithm to break
immediately: best_f = −0.984 satisfies best_f < tol=1e-10 trivially
(any negative number is less than a small positive), burning the budget
on a single evaluation.

Best algorithms use the budget most efficiently:
  BayesOpt : maximises information per expensive eval (few evals, high quality)
  PSO/CEM  : sample many points, useful when evals are cheap
  Random   : baseline — no model, pure luck

Outputs (in outputs/demo03/):
  convergence_vs_nfev.png
  timing.png
  quality_vs_time.png
  landscape_svm.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import time
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
BUDGET  = 80   # CV evaluations per algorithm (each ~0.3–0.5 s)
N_RUNS  = 3
SEED0   = 42

# ── Dataset ────────────────────────────────────────────────────────────────

X_raw, y  = load_breast_cancer(return_X_y=True)
X         = StandardScaler().fit_transform(X_raw)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Objective: 1 − accuracy ∈ [0, 1].  Minimum is 0 (100 % accuracy).
# CRITICAL: must be POSITIVE so that tol stopping criterion (best_f < tol)
# only fires near true convergence, not immediately on any valid solution.
def objective(params: np.ndarray) -> float:
    C, gamma = 10.0 ** params[0], 10.0 ** params[1]
    svm      = SVC(C=C, gamma=gamma, kernel="rbf", random_state=42)
    scores   = cross_val_score(svm, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(1.0 - scores.mean())   # misclassification rate, in [0, 1]

BOUNDS = [(-2.0, 3.0), (-5.0, 1.0)]   # [log10(C), log10(gamma)]


# ── Random Search baseline ─────────────────────────────────────────────────

class RandomSearch(oz.BaseOptimizer):
    def __init__(self, seed=None, log_interval=None, store_history=True):
        super().__init__(seed, log_interval, store_history)
        self._name = "RandomSearch"

    def optimize(self, func, bounds, max_iter=1000, max_fev=None, **_):
        t0 = time.perf_counter()
        lb, ub  = self._parse_bounds(bounds)
        history = OptimizationHistory()
        nfev    = 0
        budget  = max_fev or max_iter
        best_x  = (lb + ub) / 2
        best_f  = np.inf
        it      = 0

        for it in range(1, budget + 1):
            if not self._budget_ok(nfev, budget):
                break
            x  = self._random_population(lb, ub, 1)[0]
            fx = func(x); nfev += 1
            if fx < best_f:
                best_f = fx; best_x = x.copy()
            elapsed = time.perf_counter() - t0
            if self.store_history:
                history.record(it, best_f, elapsed=elapsed, nfev=nfev)
            self._log(it, best_f, nfev=nfev, elapsed=elapsed)

        return OptimizationResult(
            x=best_x, fun=best_f, nfev=nfev, nit=it,
            success=False, message="RandomSearch done",
            history=history, algorithm=self._name,
            elapsed=time.perf_counter() - t0)


def make_algos(seed):
    return {
        "BayesOpt-EI":  oz.BayesianOptimization(
                            n_init=5, acquisition="ei",
                            gp_restarts=2, acq_restarts=8,
                            seed=seed),
        "BayesOpt-UCB": oz.BayesianOptimization(
                            n_init=5, acquisition="ucb",
                            gp_restarts=2, acq_restarts=8,
                            seed=seed),
        "PSO":          oz.PSO(pop_size=10, seed=seed),
        "CEM":          oz.CrossEntropyMethod(pop_size=15, seed=seed),
        "RandomSearch": RandomSearch(seed=seed),
    }


def main():
    print(f"SVM HPO — Breast Cancer")
    print(f"Objective : 1 − CV_accuracy  (lower = better)")
    print(f"Budget    : {BUDGET} evaluations per algorithm  (~{BUDGET*0.4:.0f}s per algo)\n")

    results = run_suite(make_algos, objective, BOUNDS,
                        budget=BUDGET, n_runs=N_RUNS, seed0=SEED0, verbose=True)

    # ── Summary — show both objective and accuracy ─────────────────────────
    print(f"\n{'═'*72}")
    print(f"  SVM HPO — Results (budget={BUDGET} evals, {N_RUNS} runs)")
    print(f"{'═'*72}")
    print(f"  {'Algorithm':<16} {'Median Acc':>12} {'Obj(1-acc)':>12} "
          f"{'nfev':>7} {'Time (s)':>9} {'kev/s':>7}")
    print(f"  {'─'*68}")
    rows = sorted(results.items(), key=lambda kv: np.median(kv[1]['finals']))
    for name, r in rows:
        med_obj = float(np.median(r['finals']))
        med_acc = 1.0 - med_obj
        med_t   = float(np.nanmedian(r['elapsed']))
        med_n   = int(np.median(r['nfev']))
        med_tp  = float(np.median(r['throughput'])) / 1e3
        print(f"  {name:<16} {med_acc*100:>11.3f}% {med_obj:>12.4e} "
              f"{med_n:>7,} {med_t:>9.2f} {med_tp:>7.3f}")
    print(f"{'═'*72}")

    # ── Convergence plot (accuracy %, not raw objective) ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    from demos.demo_utils import PALETTE
    for idx, (name, r) in enumerate(results.items()):
        color  = PALETTE[idx % len(PALETTE)]
        traces = []
        for h in r['histories']:
            if not h.nfev_values:
                continue
            xs = [0] + list(h.nfev_values)
            # Convert objective → accuracy %
            ys = [(1.0 - v) * 100 for v in ([h.best_values[0]] + list(h.best_values))]
            xs.append(BUDGET); ys.append(ys[-1])
            traces.append((np.array(xs, dtype=float), np.array(ys)))
        if not traces:
            continue
        grid = np.linspace(0, BUDGET, 300)
        mat  = np.array([np.interp(grid, tr[0], tr[1]) for tr in traces])
        med  = np.median(mat, axis=0)
        p25  = np.percentile(mat, 25, axis=0)
        p75  = np.percentile(mat, 75, axis=0)
        ax.plot(grid, med, color=color, linewidth=2, label=name)
        ax.fill_between(grid, p25, p75, alpha=0.12, color=color)

    ax.set_xlabel("CV Evaluations (nfev)", fontsize=12)
    ax.set_ylabel("Best CV Accuracy (%)", fontsize=12)
    ax.set_title(
        f"SVM HPO Convergence — Breast Cancer  (budget={BUDGET}, {N_RUNS} runs, median ± IQR)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, BUDGET)
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "convergence_vs_nfev.png"), dpi=150)
    plt.close(fig)

    # ── Timing ────────────────────────────────────────────────────────────
    fig = plot_timing(results, title="HPO — Timing Comparison",
                      save_path=str(OUTPUT / "timing.png"))
    plt.close(fig)

    fig = plot_quality_vs_time(results, title="HPO — Quality vs Time",
                               log_y=False,
                               save_path=str(OUTPUT / "quality_vs_time.png"))
    plt.close(fig)

    # ── SVM accuracy landscape ─────────────────────────────────────────────
    print("\nComputing accuracy landscape (18×18 grid)...")
    G  = 18
    cs = np.linspace(-2, 3, G)
    gs = np.linspace(-5, 1, G)
    ZZ = np.zeros((G, G))
    for i, gc in enumerate(cs):
        for j, gg in enumerate(gs):
            ZZ[j, i] = (1.0 - objective(np.array([gc, gg]))) * 100  # accuracy %

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    cf = ax2.contourf(cs, gs, ZZ, levels=20, cmap="RdYlGn")
    plt.colorbar(cf, ax=ax2, label="CV Accuracy (%)")
    ax2.contour(cs, gs, ZZ, levels=10, colors="white", linewidths=0.4, alpha=0.5)

    # Mark best point found per algorithm (use first run's best x)
    for idx, (name, r) in enumerate(results.items()):
        # Re-run once at budget to get best x (histories don't store x)
        algo = make_algos(SEED0)[name]
        res  = algo.optimize(objective, BOUNDS, max_fev=BUDGET)
        ax2.scatter(*res.x, s=140, zorder=5, color=PALETTE[idx % len(PALETTE)],
                    label=f"{name} ({(1-res.fun)*100:.2f}%)")

    ax2.set_xlabel("log₁₀(C)", fontsize=12)
    ax2.set_ylabel("log₁₀(γ)", fontsize=12)
    ax2.set_title("SVM Accuracy Landscape\n"
                  "(markers = best point found by each algorithm)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    fig2.tight_layout()
    fig2.savefig(str(OUTPUT / "landscape_svm.png"), dpi=150)
    plt.close(fig2)

    print(f"\nDemo 3 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
