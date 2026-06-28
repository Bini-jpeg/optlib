"""
tests/test_algorithms.py
Smoke-tests: each algorithm must find the Sphere minimum within tolerance
on a low-dimensional problem.  No scipy / pytest dependency needed.
Run with:  python tests/test_algorithms.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import traceback

import optlib as oz
from optlib.benchmarks.continuous_functions import sphere, rosenbrock, ackley

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, result, tol=0.1, expected=0.0):
    ok = abs(result.fun - expected) < tol
    status = PASS if ok else FAIL
    print(f"  [{status}] {name:<22}  fun={result.fun:.4e}  nit={result.nit}")
    return ok


# ── Tests ────────────────────────────────────────────────────────────────────

def run_all():
    D       = 5
    BOUNDS  = [(-5.12, 5.12)] * D
    SEED    = 42
    passed  = 0
    total   = 0

    print(f"\nSphere (d={D})  tol=0.5\n{'─'*50}")

    cases = [
        ("GeneticAlgorithm",    oz.GeneticAlgorithm(pop_size=60, seed=SEED), 300),
        ("DE-rand1bin",         oz.DifferentialEvolution(strategy="rand1bin", seed=SEED), 300),
        ("DE-best1bin",         oz.DifferentialEvolution(strategy="best1bin", seed=SEED), 300),
        ("DE-ctbest1bin",       oz.DifferentialEvolution(strategy="ctbest1bin", seed=SEED), 300),
        ("DE-JADE",             oz.DifferentialEvolution(strategy="jade",    seed=SEED), 300),
        ("DE-SHADE",            oz.DifferentialEvolution(strategy="shade",   seed=SEED), 300),
        ("DE-L-SHADE",          oz.DifferentialEvolution(strategy="lshade",  seed=SEED), 300),
        ("CMA-ES",              oz.CMAES(seed=SEED), 500),
        ("PSO",                 oz.PSO(pop_size=30, seed=SEED), 300),
        ("CLPSO",               oz.CLPSO(pop_size=30, seed=SEED), 300),
        ("ArtificialBeeColony", oz.ArtificialBeeColony(pop_size=30, seed=SEED), 300),
        ("GreyWolfOptimizer",   oz.GreyWolfOptimizer(pop_size=20, seed=SEED), 300),
        ("FireflyAlgorithm",    oz.FireflyAlgorithm(pop_size=30, seed=SEED), 300),
        ("SA-exponential",      oz.SimulatedAnnealing(schedule="exponential", seed=SEED), 5000),
        ("SA-linear",           oz.SimulatedAnnealing(schedule="linear",    seed=SEED), 15000),
        ("SA-adaptive",         oz.SimulatedAnnealing(schedule="adaptive",  seed=SEED), 15000),
        ("NelderMead",          oz.NelderMead(seed=SEED), 3000),
        ("NelderMead-adaptive", oz.NelderMead(adaptive=True, seed=SEED), 3000),
        ("PatternSearch",       oz.PatternSearch(seed=SEED), 3000),
        ("CrossEntropyMethod",  oz.CrossEntropyMethod(pop_size=40, seed=SEED), 200),
        ("BayesianOpt-EI",      oz.BayesianOptimization(n_init=5, acquisition="ei", seed=SEED), 30),
        ("BayesianOpt-UCB",     oz.BayesianOptimization(n_init=5, acquisition="ucb", seed=SEED), 30),
    ]

    for name, algo, iters in cases:
        total += 1
        try:
            r  = algo.optimize(sphere, BOUNDS, max_iter=iters)
            ok = check(name, r, tol=0.5)
            if ok: passed += 1
        except Exception as e:
            print(f"  [{FAIL}] {name:<22}  ERROR: {e}")
            traceback.print_exc()

    print(f"\nRosenbrock (d={D})  tol=0.5\n{'─'*50}")
    rb_bounds = [(-2.048, 2.048)] * D
    for name, algo, iters in [
        ("DE-L-SHADE",  oz.DifferentialEvolution(strategy="lshade", seed=SEED), 500),
        ("CMA-ES",      oz.CMAES(seed=SEED), 2000),
        ("PSO",         oz.PSO(pop_size=50, seed=SEED), 500),
    ]:
        total += 1
        try:
            r  = algo.optimize(rosenbrock, rb_bounds, max_iter=iters)
            ok = check(name, r, tol=0.5, expected=0.0)
            if ok: passed += 1
        except Exception as e:
            print(f"  [{FAIL}] {name:<22}  ERROR: {e}")

    # ── Discrete: TSP ─────────────────────────────────────────────────────
    print(f"\nTSP (10 cities)  upper-bound=350\n{'─'*50}")
    from optlib.benchmarks.discrete_problems import TSP
    from optlib.algorithms.discrete.tabu_search import solve_tsp_tabu
    from optlib.algorithms.discrete.ant_colony  import AntColonyTSP

    tsp = TSP(n_cities=10, seed=SEED)
    total += 1
    try:
        _, L, _ = solve_tsp_tabu(tsp, max_iter=500, seed=SEED)
        ok  = L < 350
        print(f"  [{'PASS' if ok else 'FAIL'}] TabuSearch              tour_len={L:.2f}")
        if ok: passed += 1
    except Exception as e:
        print(f"  [{FAIL}] TabuSearch              ERROR: {e}")

    total += 1
    try:
        aco = AntColonyTSP(variant="MMAS", seed=SEED)
        _, L, _ = aco.solve(tsp, max_iter=100)
        ok  = L < 350
        print(f"  [{'PASS' if ok else 'FAIL'}] ACO-MMAS                tour_len={L:.2f}")
        if ok: passed += 1
    except Exception as e:
        print(f"  [{FAIL}] ACO-MMAS                ERROR: {e}")

    # ── History / Result shape checks ─────────────────────────────────────
    print(f"\nResult object checks\n{'─'*50}")
    total += 1
    r = oz.DifferentialEvolution(strategy="rand1bin", seed=SEED).optimize(
        sphere, BOUNDS, max_iter=50)
    ok = (isinstance(r.x, np.ndarray) and
          isinstance(r.fun, float) and
          len(r.history) > 0 and
          len(r.history.best_values) == len(r.history.iterations))
    print(f"  [{'PASS' if ok else 'FAIL'}] OptimizationResult shape")
    if ok: passed += 1

    print(f"\n{'═'*50}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'═'*50}\n")
    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
