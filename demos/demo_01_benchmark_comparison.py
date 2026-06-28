"""
Demo 1 — Continuous Benchmark Comparison (equal evaluation budget).

Budget : BUDGET = 50,000 function evaluations per algorithm per function.
Metrics: best value at budget exhaustion, wall-clock time, throughput.

Outputs (in outputs/demo01/):
  convergence_<func>.png  — median best vs nfev (IQR band)
  timing.png              — runtime + throughput bars
  quality_vs_time.png     — scatter: accuracy vs speed
  table.png               — colour-coded summary table
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optlib as oz
from optlib.benchmarks import REGISTRY
from optlib.visualization.plots import plot_benchmark_table, plot_landscape_2d
from demos.demo_utils import (run_suite, print_summary_table,
                               plot_convergence_nfev, plot_timing,
                               plot_quality_vs_time)

OUTPUT  = Path("outputs/demo01"); OUTPUT.mkdir(parents=True, exist_ok=True)
BUDGET  = 50_000          # evaluations per algorithm per function
N_RUNS  = 5               # independent seeds
SEED0   = 42
FUNCS   = ["ackley", "rastrigin", "rosenbrock", "schwefel"]


def make_algos(seed):
    # log_interval=None keeps demo output clean (suite prints its own summary)
    return {
        "GA":           oz.GeneticAlgorithm(pop_size=80,  seed=seed),
        "DE-rand1":     oz.DifferentialEvolution(strategy="rand1bin",  seed=seed),
        "DE-L-SHADE":   oz.DifferentialEvolution(strategy="lshade",    seed=seed),
        "CMA-ES":       oz.CMAES(seed=seed),
        "PSO":          oz.PSO(pop_size=40,  seed=seed),
        "CLPSO":        oz.CLPSO(pop_size=40, seed=seed),
        "ABC":          oz.ArtificialBeeColony(pop_size=50, seed=seed),
        "GWO":          oz.GreyWolfOptimizer(pop_size=30, seed=seed),
        "FA":           oz.FireflyAlgorithm(pop_size=40, seed=seed),
        "SA":           oz.SimulatedAnnealing(schedule="exponential", seed=seed),
        "CEM":          oz.CrossEntropyMethod(pop_size=50, seed=seed),
        "NelderMead":   oz.NelderMead(seed=seed),
        "PatternSearch":oz.PatternSearch(seed=seed),
    }


def main():
    all_results = {}     # func -> results dict

    for fname in FUNCS:
        info   = REGISTRY[fname]
        func   = info["func"]
        bounds = info["bounds"]
        print(f"\n{'='*60}")
        print(f"  Function: {fname.upper()}  (d={len(bounds)}, budget={BUDGET:,})")
        print(f"{'='*60}")

        results = run_suite(make_algos, func, bounds,
                            budget=BUDGET, n_runs=N_RUNS, seed0=SEED0)
        all_results[fname] = results
        print_summary_table(results, title=f"{fname} — results after {BUDGET:,} evals")

        # Convergence vs nfev
        fig = plot_convergence_nfev(
            results, BUDGET,
            title=f"Convergence — {fname}  (d={len(bounds)}, budget={BUDGET:,}, "
                  f"{N_RUNS} runs, median ± IQR)",
            log_y=True,
            save_path=str(OUTPUT / f"convergence_{fname}.png"))
        plt.close(fig)

    # ── Timing (averaged across all functions) ─────────────────────────────
    # Merge: for each algo, pool elapsed times from all functions
    merged: dict = {}
    for fname, results in all_results.items():
        for name, r in results.items():
            if name not in merged:
                merged[name] = dict(elapsed=[], throughput=[], finals=[], nfev=[])
            merged[name]['elapsed']    += r['elapsed']
            merged[name]['throughput'] += r['throughput']
            merged[name]['finals']     += r['finals']
            merged[name]['nfev']       += r['nfev']

    fig = plot_timing(merged,
                      title=f"Timing Comparison — averaged over {FUNCS}",
                      save_path=str(OUTPUT / "timing.png"))
    plt.close(fig)

    # ── Quality vs time (focal function: ackley) ───────────────────────────
    fig = plot_quality_vs_time(
        all_results["ackley"],
        title=f"Quality vs Time — ackley  ({N_RUNS} runs median)",
        save_path=str(OUTPUT / "quality_vs_time.png"))
    plt.close(fig)

    # ── Colour-coded summary table ─────────────────────────────────────────
    table_data = {
        algo: {fname: float(np.median(all_results[fname][algo]['finals']))
               for fname in FUNCS}
        for algo in make_algos(SEED0)
    }
    fig = plot_benchmark_table(table_data,
                               save_path=str(OUTPUT / "table.png"))
    plt.close(fig)

    # ── 2-D landscape for visual reference ────────────────────────────────
    from optlib.benchmarks.continuous_functions import ackley
    fig = plot_landscape_2d(ackley, bounds=[(-32.768, 32.768)]*2,
                            title="Ackley landscape (2-D slice)",
                            save_path=str(OUTPUT / "landscape_ackley.png"))
    plt.close(fig)

    print(f"\nDemo 1 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
