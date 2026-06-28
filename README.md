# optlib 🦁

> **A comprehensive, from-scratch optimisation library that goes far beyond gradient descent.**

`optlib` implements 14 major metaheuristic and derivative-free optimisation algorithms — many with multiple SOTA variants — in clean, efficient NumPy/SciPy code. It includes preprocessing utilities, structured logging, rich visualisations, standard benchmarks, and five end-to-end demo scripts.

---

## Algorithms at a Glance

| Category | Algorithm | Key Variant(s) | Best For |
|---|---|---|---|
| **Evolutionary** | Genetic Algorithm (GA) | SBX + polynomial mutation | Combinatorial, real-valued, feature selection |
| | Differential Evolution (DE) | rand1bin, best1bin, ctbest1bin, JADE, **SHADE**, **L-SHADE** | Continuous, SOTA single-objective |
| | CMA-ES | IPOP restarts | Continuous, moderate dimension (d ≤ 200) |
| **Swarm** | Particle Swarm (PSO) | Standard + **CLPSO** | Continuous, fast convergence |
| | Artificial Bee Colony (ABC) | — | Multi-modal continuous |
| | Grey Wolf Optimizer (GWO) | — | Continuous, simple and fast |
| | Firefly Algorithm (FA) | — | Multi-modal, attraction-based |
| **Annealing** | Simulated Annealing (SA) | Exponential, linear, logarithmic, adaptive cooling | Discrete and continuous, escapes local optima |
| **Bayesian** | Bayesian Optimisation (BO) | EI, UCB, PI acquisition; ARD GP surrogate | Expensive black-box functions (HPO) |
| **Direct Search** | Nelder-Mead Simplex | Adaptive coefficients (Gao & Han) | Smooth low-D, no gradients |
| | Generalised Pattern Search (GPS) | LHS search step | Noisy, non-smooth |
| **Distribution** | Cross-Entropy Method (CEM) | Gaussian, with smoothing | Continuous, fast + simple |
| **Discrete** | Tabu Search (TS) | 2-opt, aspiration criterion | TSP, combinatorial |
| | Ant Colony (ACO) | **Ant System** + **MAX-MIN AS** | TSP, routing problems |

---

## Installation

```bash
git clone https://github.com/your-handle/optlib.git
cd optlib
pip install -e .
# or just: pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, NumPy, SciPy, Matplotlib, scikit-learn

---

## Quick Start

```python
import optlib as oz
from optlib.benchmarks import ackley

bounds = [(-32.768, 32.768)] * 10   # 10-dimensional Ackley

# ── L-SHADE (SOTA DE variant) ──────────────────────────────────────────────
result = oz.DifferentialEvolution(strategy="lshade", seed=42, log_interval=50) \
           .optimize(ackley, bounds, max_iter=500)
print(result)

# ── CMA-ES with IPOP restarts ──────────────────────────────────────────────
result = oz.CMAES(n_restarts=2, seed=42, log_interval=100) \
           .optimize(ackley, bounds, max_iter=5000)

# ── Bayesian Optimisation (for expensive functions) ────────────────────────
result = oz.BayesianOptimization(acquisition="ei", n_init=5) \
           .optimize(my_expensive_fn, bounds, max_iter=50)

# ── Particle Swarm ────────────────────────────────────────────────────────
result = oz.PSO(pop_size=40, seed=42).optimize(ackley, bounds, max_iter=300)

# ── Simulated Annealing ───────────────────────────────────────────────────
result = oz.SimulatedAnnealing(schedule="exponential") \
           .optimize(ackley, bounds, max_iter=10_000)
```

All optimisers return a uniform `OptimizationResult`:
```python
result.x        # best solution (ndarray)
result.fun      # best objective value (float)
result.nfev     # function evaluations used
result.nit      # iterations
result.elapsed  # wall-clock time (seconds)
result.history  # per-iteration best/mean/std values
```

---

## Constraint Handling

```python
from optlib.preprocessing.preprocessing import PenaltyConstraintHandler

constraints = [
    lambda x: x[0]**2 + x[1]**2 - 1.0,   # g(x) ≤ 0
]
handler = PenaltyConstraintHandler(constraints, penalty=1e4)
penalised_obj = handler.wrap(my_objective)

result = oz.CMAES().optimize(penalised_obj, bounds)
```

---

## Discrete Problems

```python
from optlib.benchmarks           import TSP
from optlib.algorithms.discrete.ant_colony  import AntColonyTSP
from optlib.algorithms.discrete.tabu_search import solve_tsp_tabu

tsp = TSP(n_cities=30, seed=42)

# Ant Colony (MAX-MIN Ant System)
aco = AntColonyTSP(variant="MMAS", seed=42, log_interval=20)
tour, length, history = aco.solve(tsp, max_iter=200)

# Tabu Search with 2-opt
tour, length, history = solve_tsp_tabu(tsp, tabu_tenure=15, seed=42)
```

---

## Visualisations

```python
from optlib.visualization.plots import (
    plot_convergence,           # convergence curves (multiple algos)
    plot_comparison_boxplot,    # box-plot final values N runs
    plot_landscape_2d,          # contour map + trajectory overlay
    plot_tsp_tour,              # route plot for TSP
    plot_multirun_convergence,  # median ± IQR bands
    plot_benchmark_table,       # colour-coded summary table
)
```

---

## Demos

| Script | What it shows |
|---|---|
| `demos/demo_01_benchmark_comparison.py` | All continuous optimisers on Ackley, Rastrigin, Rosenbrock, Schwefel |
| `demos/demo_02_tsp.py` | GA, SA, Tabu Search, ACO-MMAS on a 25-city TSP |
| `demos/demo_03_bayesian_hpo.py` | Bayesian Opt vs PSO vs CEM for SVM hyperparameter tuning |
| `demos/demo_04_engineering_design.py` | Constrained pressure-vessel design — DE/SA/CMA-ES/GWO |
| `demos/demo_05_feature_selection.py` | Binary GA for feature selection on Breast Cancer dataset |

Run all demos:
```bash
python demos/demo_01_benchmark_comparison.py
python demos/demo_02_tsp.py
python demos/demo_03_bayesian_hpo.py
python demos/demo_04_engineering_design.py
python demos/demo_05_feature_selection.py
```
Outputs land in `outputs/demo0*/`.

---

## Run Tests

```bash
python tests/test_algorithms.py
```
Tests all 22+ algorithm variants on Sphere and Rosenbrock, plus TSP smoke-tests.

---

## Algorithm Selection Guide

```
Your problem is...
├── Continuous, smooth, cheap evaluations
│   ├── d ≤ 20         → CMA-ES or Nelder-Mead
│   ├── d ≤ 200        → CMA-ES or L-SHADE
│   └── d > 200        → L-SHADE or PSO
│
├── Continuous, multimodal (many local minima)
│   ├── Fast           → DE-L-SHADE, GWO
│   └── Very multimodal → FA + Firefly, PSO-CLPSO
│
├── Expensive (< 500 evaluations budget)
│   └──                → Bayesian Optimisation (BO)
│
├── Continuous, constrained
│   ├── Smooth         → CMA-ES + penalty, or DE + penalty
│   └── Non-smooth     → SA or Pattern Search
│
├── Discrete / combinatorial
│   ├── Routing (TSP)  → ACO-MMAS or Tabu Search
│   ├── Binary vectors → Binary GA
│   └── General        → SA (with custom neighbour_fn)
│
└── Noisy / stochastic
    └──                → CEM, SA, or Bayesian Opt
```

---

## Project Structure

```
optlib/
├── optlib/
│   ├── base.py                          # BaseOptimizer, OptimizationResult
│   ├── algorithms/
│   │   ├── evolutionary/
│   │   │   ├── genetic_algorithm.py     # GA (SBX + polynomial mutation)
│   │   │   ├── differential_evolution.py # DE + JADE + SHADE + L-SHADE
│   │   │   └── cma_es.py               # CMA-ES (IPOP restarts)
│   │   ├── swarm/
│   │   │   ├── pso.py                  # Standard PSO + CLPSO
│   │   │   ├── artificial_bee_colony.py
│   │   │   ├── grey_wolf.py
│   │   │   └── firefly.py
│   │   ├── annealing/
│   │   │   └── simulated_annealing.py  # 4 cooling schedules
│   │   ├── bayesian/
│   │   │   └── bayesian_optimization.py # GP surrogate + EI/UCB/PI
│   │   ├── direct_search/
│   │   │   ├── nelder_mead.py
│   │   │   └── pattern_search.py
│   │   ├── population_based/
│   │   │   └── cross_entropy.py
│   │   └── discrete/
│   │       ├── tabu_search.py
│   │       └── ant_colony.py           # Ant System + MAX-MIN AS
│   ├── preprocessing/
│   │   └── preprocessing.py            # MinMax/Standard scalers, LHS, constraint handlers
│   ├── logging_utils/
│   │   └── logger.py
│   ├── visualization/
│   │   └── plots.py                    # 7 plot types
│   └── benchmarks/
│       ├── continuous_functions.py     # 15 standard test functions
│       └── discrete_problems.py        # TSP + 0/1 Knapsack
├── demos/                              # 5 end-to-end demo scripts
├── tests/
│   └── test_algorithms.py
├── requirements.txt
└── setup.py
```

---

## References

- **DE/L-SHADE**: Tanabe & Fukunaga (2014). *Improving the Search Performance of SHADE Using Linear Population Size Reduction.*
- **CMA-ES**: Hansen (2016). *The CMA Evolution Strategy: A Tutorial.* arXiv:1604.00772.
- **ABC**: Karaboga (2005). *An Idea Based on Honey Bee Swarm for Numerical Optimization.*
- **PSO/CLPSO**: Liang et al. (2006). *Comprehensive Learning Particle Swarm Optimizer.*
- **GWO**: Mirjalili et al. (2014). *Grey Wolf Optimizer.*
- **Firefly**: Yang (2009). *Firefly Algorithms for Multimodal Optimization.*
- **CEM**: Rubinstein & Kroese (2004). *The Cross-Entropy Method.*
- **GPS**: Torczon (1997). *On the Convergence of Pattern Search Algorithms.*
- **MMAS**: Stützle & Hoos (2000). *MAX-MIN Ant System.*
- **Tabu Search**: Glover (1989). *Tabu Search — Part I.*

---

## License

MIT — see `LICENSE`.
