"""
Demo 4 — Constrained Engineering Design (equal evaluation budget).

Problem : Pressure Vessel Design (Coello, 2000).
Budget  : BUDGET = 20,000 evaluations per algorithm (d=4).

Constraints handled via static penalty: g_i(x) ≤ 0.
Reference optimum ≈ 6059.71.

Outputs (in outputs/demo04/):
  convergence_vs_nfev.png
  timing.png
  quality_vs_time.png
  boxplot.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optlib as oz
from optlib.preprocessing.preprocessing import PenaltyConstraintHandler
from optlib.visualization.plots import plot_comparison_boxplot
from demos.demo_utils import (run_suite, print_summary_table,
                               plot_convergence_nfev, plot_timing,
                               plot_quality_vs_time)

OUTPUT  = Path("outputs/demo04"); OUTPUT.mkdir(parents=True, exist_ok=True)
BUDGET  = 20_000   # evaluations per algorithm (d=4)
N_RUNS  = 8
SEED0   = 42

# ── Problem ────────────────────────────────────────────────────────────────

BOUNDS = [(0.0625, 6.1875), (0.0625, 6.1875), (10.0, 200.0), (10.0, 200.0)]

def cost(x):
    Ts,Th,R,L = x
    return (0.6224*Ts*R*L + 1.7781*Th*R**2 + 3.1661*Ts**2*L + 19.84*Ts**2*R)

CONSTRAINTS = [
    lambda x: -x[0] + 0.0193*x[2],
    lambda x: -x[1] + 0.00954*x[2],
    lambda x: -math.pi*x[2]**2*x[3] - (4/3)*math.pi*x[2]**3 + 1_296_000,
    lambda x: x[3] - 240.0,
]

handler       = PenaltyConstraintHandler(CONSTRAINTS, penalty=1e6)
obj_penalised = handler.wrap(cost)


def make_algos(seed):
    return {
        "GA":           oz.GeneticAlgorithm(pop_size=80,  seed=seed),
        "DE-rand1":     oz.DifferentialEvolution(strategy="rand1bin", seed=seed),
        "DE-L-SHADE":   oz.DifferentialEvolution(strategy="lshade",   seed=seed),
        "CMA-ES":       oz.CMAES(seed=seed, n_restarts=1),
        "PSO":          oz.PSO(pop_size=40, seed=seed),
        "GWO":          oz.GreyWolfOptimizer(pop_size=30, seed=seed),
        "SA":           oz.SimulatedAnnealing(schedule="exponential", seed=seed),
        "CEM":          oz.CrossEntropyMethod(pop_size=50, seed=seed),
    }


def main():
    print(f"Pressure Vessel Design — budget={BUDGET:,} evals  {N_RUNS} runs  "
          f"(reference optimum ≈ 6059.71)\n")

    results = run_suite(make_algos, obj_penalised, BOUNDS,
                        budget=BUDGET, n_runs=N_RUNS, seed0=SEED0)

    print_summary_table(results, title=f"Engineering Design — Penalised Cost (budget={BUDGET:,})")

    # ── Feasibility check on best run per algo ─────────────────────────────
    print("  Feasibility check (best run per algo):")
    for name, r in results.items():
        bi = int(np.argmin(r['finals']))
        print(f"    {name:<16}  best_cost={r['finals'][bi]:.2f}  "
              f"('~{cost(np.array([0.8125,0.4375,42.09,176.75])):.0f}' ref)")

    # ── Plots ──────────────────────────────────────────────────────────────
    fig = plot_convergence_nfev(
        results, BUDGET,
        title=f"Engineering Design Convergence  (budget={BUDGET:,}, {N_RUNS} runs)",
        log_y=True,
        save_path=str(OUTPUT / "convergence_vs_nfev.png")); plt.close(fig)

    fig = plot_timing(results, title="Engineering Design — Timing",
                      save_path=str(OUTPUT / "timing.png")); plt.close(fig)

    fig = plot_quality_vs_time(results,
                               title="Engineering Design — Quality vs Time",
                               log_y=True,
                               save_path=str(OUTPUT / "quality_vs_time.png")); plt.close(fig)

    box_data = {n: r['finals'] for n,r in results.items()}
    fig = plot_comparison_boxplot(
        box_data,
        title=f"Penalised Cost Distribution  ({N_RUNS} runs per algo, budget={BUDGET:,})",
        log_y=False,
        save_path=str(OUTPUT / "boxplot.png")); plt.close(fig)

    print(f"\nDemo 4 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
