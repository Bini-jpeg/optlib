"""
Ant Colony Optimisation (ACO) — Dorigo et al. (1996).

Implements the Ant System (AS) and MAX-MIN Ant System (MMAS) for TSP.

Ant System (AS)
  All ants update pheromones after each tour.
  p_ij = (τ_ij^α · η_ij^β) / Σ (τ_ik^α · η_ik^β)
  τ_ij ← (1-ρ) τ_ij + Σ_k ΔQ/L_k  (if ant k traverses edge ij)

MAX-MIN Ant System (MMAS) — Stützle & Hoos (2000)
  Only the best ant deposits pheromone.
  Pheromone bounded by [τ_min, τ_max] to prevent stagnation.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from optlib.benchmarks.discrete_problems import TSP


# ── Ant System ─────────────────────────────────────────────────────────────

class AntColonyTSP:
    """
    ACO for the symmetric Euclidean TSP.

    Parameters
    ----------
    n_ants      : number of ants per generation (default n_cities)
    alpha       : pheromone weight (default 1.0)
    beta        : heuristic weight (default 5.0)
    rho         : pheromone evaporation rate (default 0.1)
    Q           : pheromone deposit constant (default 1.0)
    variant     : 'AS' (Ant System) or 'MMAS' (MAX-MIN)
    tau_init    : initial pheromone (None → 1 / (n * C_nn))
    """

    def __init__(self, n_ants=None, alpha=1.0, beta=5.0, rho=0.1,
                 Q=1.0, variant='MMAS', tau_init=None,
                 seed=None, log_interval=None):
        self.n_ants     = n_ants
        self.alpha      = alpha
        self.beta       = beta
        self.rho        = rho
        self.Q          = Q
        self.variant    = variant
        self.tau_init   = tau_init
        self.rng        = np.random.default_rng(seed)
        self.log_interval = log_interval

    def solve(self, tsp: TSP, max_iter: int = 200) -> Tuple[np.ndarray, float, List[float]]:
        n        = tsp.n
        n_ants   = self.n_ants or n
        dist     = tsp.dist
        eta      = 1.0 / (dist + np.eye(n) * 1e-10)    # heuristic (1/d), avoid div-0

        # Initial pheromone
        nn_len   = tsp.tour_length(tsp.nearest_neighbour())
        tau0     = self.tau_init if self.tau_init else 1.0 / (n * nn_len)
        tau      = np.full((n, n), tau0, dtype=float)

        # MMAS bounds
        tau_max  = 1.0 / (self.rho * nn_len)
        tau_min  = tau_max / (2.0 * n)

        best_tour   = tsp.nearest_neighbour()
        best_tour   = tsp.two_opt_improve(best_tour)
        best_length = tsp.tour_length(best_tour)
        history     = [best_length]
        t0          = time.perf_counter()

        for it in range(1, max_iter + 1):
            tours    = np.empty((n_ants, n), dtype=int)
            lengths  = np.empty(n_ants)

            for k in range(n_ants):
                tours[k]   = self._construct_tour(n, tau, eta)
                lengths[k] = tsp.tour_length(tours[k])

            # ── Pheromone evaporation ──────────────────────────────────
            tau *= (1.0 - self.rho)

            # ── Pheromone deposit ──────────────────────────────────────
            if self.variant == 'AS':
                for k in range(n_ants):
                    delta = self.Q / lengths[k]
                    t     = tours[k]
                    for i in range(n):
                        a, b = t[i], t[(i+1) % n]
                        tau[a, b] += delta
                        tau[b, a] += delta
            else:  # MMAS: only best ant (iteration best or global best)
                iter_best_k = int(np.argmin(lengths))
                if lengths[iter_best_k] < best_length:
                    depositor_t = tours[iter_best_k]
                    depositor_l = lengths[iter_best_k]
                else:
                    depositor_t = best_tour
                    depositor_l = best_length
                delta = self.Q / depositor_l
                for i in range(n):
                    a, b = depositor_t[i], depositor_t[(i+1) % n]
                    tau[a, b] += delta
                    tau[b, a] += delta
                # Clamp
                tau = np.clip(tau, tau_min, tau_max)

            # ── Update global best ─────────────────────────────────────
            k_best = int(np.argmin(lengths))
            if lengths[k_best] < best_length:
                best_length = lengths[k_best]
                best_tour   = tours[k_best].copy()
                # 2-opt local improvement on new global best
                best_tour   = tsp.two_opt_improve(best_tour, max_passes=5)
                best_length = tsp.tour_length(best_tour)

            history.append(best_length)

            if self.log_interval and it % self.log_interval == 0:
                elapsed = time.perf_counter() - t0
                print(f"[ACO-{self.variant}] iter={it:>5d}  best={best_length:.2f}"
                      f"  tau_max={tau.max():.3e}  elapsed={elapsed:.1f}s")

        return best_tour, best_length, history

    def _construct_tour(self, n, tau, eta):
        """Build one ant's tour using probabilistic next-city selection."""
        visited = np.zeros(n, dtype=bool)
        tour    = np.empty(n, dtype=int)
        start   = int(self.rng.integers(0, n))
        tour[0] = start
        visited[start] = True

        for step in range(1, n):
            current = tour[step - 1]
            unvisited = ~visited

            attract = (tau[current] * unvisited) ** self.alpha * \
                      (eta[current]  * unvisited) ** self.beta
            total   = attract.sum()

            if total < 1e-300:
                # Fallback: random choice among unvisited
                choices = np.where(unvisited)[0]
                nxt     = int(self.rng.choice(choices))
            else:
                probs = attract / total
                nxt   = int(self.rng.choice(n, p=probs))

            tour[step]   = nxt
            visited[nxt] = True

        return tour
