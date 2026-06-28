"""
Tabu Search (TS) — Glover (1989).

General-purpose local search with short-term memory (tabu list).
An aspiration criterion allows overriding the tabu status when a move
improves the global best.

This module ships with a concrete 2-opt TSP solver built on the generic
framework.  You can supply any problem-specific classes via dependency
injection.
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from optlib.benchmarks.discrete_problems import TSP


# ── Generic TS framework ───────────────────────────────────────────────────

class TabuSearch:
    """
    Generic Tabu Search.

    Parameters
    ----------
    neighbour_fn  : callable(solution, rng) → list[(move, neighbour)]
    evaluate_fn   : callable(solution) → float  (lower = better)
    tabu_tenure   : int — how many iterations a move stays tabu (default 7)
    max_iter      : maximum iterations
    max_no_impr   : restart from best if stuck this many iters (default ∞)
    seed          : RNG seed
    log_interval  : print every N iters (None = silent)
    """

    def __init__(
        self,
        neighbour_fn:  Callable,
        evaluate_fn:   Callable,
        tabu_tenure:   int = 7,
        max_iter:      int = 1000,
        max_no_impr:   int = 200,
        seed:          Optional[int] = None,
        log_interval:  Optional[int] = None,
    ):
        self.neighbour_fn = neighbour_fn
        self.evaluate_fn  = evaluate_fn
        self.tabu_tenure  = tabu_tenure
        self.max_iter     = max_iter
        self.max_no_impr  = max_no_impr
        self.rng          = np.random.default_rng(seed)
        self.log_interval = log_interval

    def run(self, initial_solution: Any) -> Tuple[Any, float, List[float]]:
        """
        Returns
        -------
        best_solution, best_cost, history_of_best_costs
        """
        current    = initial_solution
        f_current  = self.evaluate_fn(current)
        best       = current
        f_best     = f_current
        tabu_list  = {}           # move -> expiry iteration
        history    = [f_best]
        no_impr    = 0
        t0         = time.perf_counter()

        for it in range(1, self.max_iter + 1):
            candidates = self.neighbour_fn(current, self.rng)
            best_cand  = None
            f_best_cand= np.inf
            best_move  = None

            for move, neighbour in candidates:
                fc = self.evaluate_fn(neighbour)
                # Accept if: not tabu OR aspiration criterion (beats global best)
                is_tabu   = tabu_list.get(move, 0) >= it
                aspiration = fc < f_best
                if (not is_tabu or aspiration) and fc < f_best_cand:
                    best_cand   = neighbour
                    f_best_cand = fc
                    best_move   = move

            if best_cand is None:        # all neighbours tabu — take best anyway
                move, best_cand = candidates[0]
                f_best_cand     = self.evaluate_fn(best_cand)
                best_move       = move

            # Commit move
            current   = best_cand
            f_current = f_best_cand
            if best_move is not None:
                tabu_list[best_move] = it + self.tabu_tenure

            if f_current < f_best:
                f_best = f_current; best = current
                no_impr = 0
            else:
                no_impr += 1

            history.append(f_best)

            if self.log_interval and it % self.log_interval == 0:
                elapsed = time.perf_counter() - t0
                print(f"[TabuSearch] iter={it:>5d}  best={f_best:.4f}  "
                      f"tabu_size={len(tabu_list):>4d}  elapsed={elapsed:.1f}s")

            if no_impr >= self.max_no_impr:
                # Restart from best
                current = best; f_current = f_best
                no_impr = 0

        return best, f_best, history


# ── TSP-specific glue ──────────────────────────────────────────────────────

def tsp_2opt_neighbours(tour: np.ndarray, rng, max_moves: int = 100):
    """
    Generate up to *max_moves* 2-opt moves for a TSP tour.
    Each move is ((i, j), new_tour) where the segment [i, j) is reversed.
    """
    n    = len(tour)
    pairs = list(rng.permutation(
        [(i, j) for i in range(1, n-1) for j in range(i+1, n)]
    ))[:max_moves]
    moves = []
    for (i, j) in pairs:
        new_tour    = tour.copy()
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        moves.append(((i, j), new_tour))
    return moves


def solve_tsp_tabu(tsp_instance: TSP,
                   tabu_tenure: int   = 15,
                   max_iter:    int   = 2000,
                   max_no_impr: int   = 300,
                   seed=None,
                   log_interval=None) -> Tuple[np.ndarray, float, List[float]]:
    """
    Solve a TSP instance with Tabu Search (2-opt neighbourhood).

    Returns
    -------
    best_tour, best_length, convergence_history
    """
    ts = TabuSearch(
        neighbour_fn = tsp_2opt_neighbours,
        evaluate_fn  = tsp_instance.tour_length,
        tabu_tenure  = tabu_tenure,
        max_iter     = max_iter,
        max_no_impr  = max_no_impr,
        seed         = seed,
        log_interval = log_interval,
    )
    # Warm start: nearest-neighbour + 2-opt
    init_tour = tsp_instance.nearest_neighbour()
    init_tour = tsp_instance.two_opt_improve(init_tour)

    return ts.run(init_tour)
