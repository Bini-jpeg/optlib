"""
optlib  –  A comprehensive optimisation library beyond gradient descent.
"""
from optlib.base import BaseOptimizer, OptimizationResult, OptimizationHistory

from optlib.algorithms.evolutionary.genetic_algorithm      import GeneticAlgorithm
from optlib.algorithms.evolutionary.differential_evolution import DifferentialEvolution
from optlib.algorithms.evolutionary.cma_es                 import CMAES
from optlib.algorithms.swarm.pso                           import PSO, CLPSO
from optlib.algorithms.swarm.artificial_bee_colony         import ArtificialBeeColony
from optlib.algorithms.swarm.grey_wolf                     import GreyWolfOptimizer
from optlib.algorithms.swarm.firefly                       import FireflyAlgorithm
from optlib.algorithms.annealing.simulated_annealing       import SimulatedAnnealing
from optlib.algorithms.bayesian.bayesian_optimization      import BayesianOptimization
from optlib.algorithms.direct_search.nelder_mead           import NelderMead
from optlib.algorithms.direct_search.pattern_search        import PatternSearch
from optlib.algorithms.population_based.cross_entropy      import CrossEntropyMethod

__version__ = "1.0.0"
__all__ = [
    "BaseOptimizer", "OptimizationResult", "OptimizationHistory",
    "GeneticAlgorithm", "DifferentialEvolution", "CMAES",
    "PSO", "CLPSO", "ArtificialBeeColony", "GreyWolfOptimizer", "FireflyAlgorithm",
    "SimulatedAnnealing", "BayesianOptimization",
    "NelderMead", "PatternSearch", "CrossEntropyMethod",
]
