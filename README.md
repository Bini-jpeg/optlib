# optlib - gradient-free optimizer library

`optlib` has 14 derivative-free optimisation algorithms. it includes preprocessing, logging, visualisations, benchmarks and some demos. I initially built it for specific tasks I had, but I decided to create a general mini library

---

| category | algorithm | variants | best for |
|---|---|---|---|
| **Evolutionary** | Genetic Algorithm (GA) | polynomial mutation | combinatorial, real-valued, feature selection |
| | Differential Evolution | rand1bin, best1bin, ctbest1bin, JADE, **SHADE**, **L-SHADE** | continuous and single-objective |
| | CMA-ES | IPOP restarts | Continuous (dim ≤ 200) |
| **Swarm** | Particle Swarm) | **CLPSO** | continuous and fast convergence |
| | Artificial Bee Colony | — | multi-modal continuous |
| | Grey Wolf Optimizer | — | continuous, simple and fast |
| | Firefly Algorithm | — | multi-modal, attraction-based |
| **Annealing** | Simulated Annealing | exponential, linear, logarithmic, adaptive cooling | discrete and continuous |
| **Bayesian** | Bayesian Optimisation | EI, UCB, PI acquisition; ARD GP surrogate | expensive black-box functions |
| **Direct Search** | Nelder-Mead simplex | adaptive coefficients | smooth low-D w. no gradients |
| | Generalised Pattern Search | LHS search step | noisy, non-smooth |
| **Distribution** | Cross-Entropy Method | gaussian with smoothing | continuous |
| **Discrete** | Tabu Search | 2-opt, aspiration criterion | TSP, combinatorial |
| | Ant Colony (ACO) | **Ant System** + **MAX-MIN AS** | TSP, routing problems |

---

## Installing

```bash
git clone https://github.com/your-handle/optlib.git
cd optlib
pip install -r requirements.txt
# or:   pip install -e .
```

---

## Usage

```python
import optlib as oz
from optlib.benchmarks import ackley

bounds = [(-32.768, 32.768)] * 10   # 10-dim ackley

result = oz.DifferentialEvolution(strategy="lshade", seed=42, log_interval=50) \
           .optimize(ackley, bounds, max_iter=500)
print(result)

result = oz.CMAES(n_restarts=2, seed=42, log_interval=100) \
           .optimize(ackley, bounds, max_iter=5000)

result = oz.BayesianOptimization(acquisition="ei", n_init=5) \
           .optimize(my_expensive_fn, bounds, max_iter=50)

result = oz.PSO(pop_size=40, seed=42).optimize(ackley, bounds, max_iter=300)

result = oz.SimulatedAnnealing(schedule="exponential") \
           .optimize(ackley, bounds, max_iter=10_000)
```

All optimisers return a `OptimizationResult`:
```python
result.x        # best solution
result.fun      # best objective value
result.nfev     # function evaluations used
result.nit      # iterations
result.elapsed  # time in seconds
result.history  # best/mean/std values at each iter
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

## Discrete problems

```python
from optlib.benchmarks           import TSP
from optlib.algorithms.discrete.ant_colony  import AntColonyTSP
from optlib.algorithms.discrete.tabu_search import solve_tsp_tabu

tsp = TSP(n_cities=30, seed=42)

aco = AntColonyTSP(variant="MMAS", seed=42, log_interval=20)
tour, length, history = aco.solve(tsp, max_iter=200)

tour, length, history = solve_tsp_tabu(tsp, tabu_tenure=15, seed=42)
```

---

## Visualise

```python
from optlib.visualization.plots import (
    plot_convergence,
    plot_comparison_boxplot,
    plot_landscape_2d,
    plot_tsp_tour,
    plot_multirun_convergence,
    plot_benchmark_table,
)
```

---

## Demos

to run demos:
```bash
python demos/demo_01_benchmark_comparison.py
python demos/demo_02_tsp.py
python demos/demo_03_bayesian_hpo.py
python demos/demo_04_engineering_design.py
python demos/demo_05_feature_selection.py
```
outputs will be in `outputs/demo0*/`.

---

## which algo to use

```
if your problem is...
├── continuous, smooth, cheap to eval
│   ├── d ≤ 20         → CMA-ES or Nelder-Mead
│   ├── d ≤ 200        → CMA-ES or L-SHADE
│   └── d > 200        → L-SHADE or PSO
│
├── continuous and multimodal
│   ├── quick           → DE-L-SHADE, GWO
│   └── very multimodal → FA + Firefly, PSO-CLPSO
│
├── expensive to evaluate
│   └──                → Bayesian
│
├── continuous with constrains
│   ├── smooth         → CMA-ES + penalty, or DE + penalty
│   └── non-smooth     → SA or pattern search
│
├── discrete / combinatorial
│   ├── routing (TSP)  → ACO-MMAS or Tabu Search
│   ├── binary vectors → Binary GA
│   └── general        → SA (with custom neighbour_fn)
│
└── noisy / stochastic
    └──                → CEM, SA, or Bayesian Opt
```
