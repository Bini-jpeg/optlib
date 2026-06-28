"""
Ant Colony Optimisation — Ant System (AS) + MAX-MIN Ant System (MMAS).

Speed notes
-----------
The main bottleneck is tour construction.  We precompute τᵅ and ηᵝ once
per generation (expensive exponentiation done n×n, not n×n×n_ants×iters)
and use np.searchsorted on the cumulative-sum vector instead of the much
slower np.random.choice(n, p=probs), which has significant Python overhead.
Together these give ~10-20× speedup over the naïve implementation.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from optlib.benchmarks.discrete_problems import TSP


class AntColonyTSP:
    """
    ACO for the symmetric Euclidean TSP.

    Parameters
    ----------
    n_ants   : ants per generation (default = n_cities)
    alpha    : pheromone exponent (default 1.0)
    beta     : heuristic exponent  (default 5.0)
    rho      : evaporation rate    (default 0.1)
    Q        : pheromone constant  (default 1.0)
    variant  : 'AS' or 'MMAS'
    """

    def __init__(self, n_ants=None, alpha=1.0, beta=5.0, rho=0.1,
                 Q=1.0, variant='MMAS', tau_init=None,
                 seed=None, log_interval=None):
        self.n_ants      = n_ants
        self.alpha       = alpha
        self.beta        = beta
        self.rho         = rho
        self.Q           = Q
        self.variant     = variant
        self.tau_init    = tau_init
        self.rng         = np.random.default_rng(seed)
        self.log_interval = log_interval

    def solve(self, tsp: TSP, max_iter: int = 200) -> Tuple[np.ndarray, float, List[float]]:
        n       = tsp.n
        n_ants  = self.n_ants or n
        dist    = tsp.dist
        # heuristic matrix (1/d), diagonal zeroed so unvisited masking works
        eta     = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (dist + 1e-10))

        nn_tour  = tsp.nearest_neighbour()
        nn_L     = tsp.tour_length(nn_tour)
        tau0     = self.tau_init if self.tau_init else 1.0 / (n * nn_L)
        tau      = np.full((n, n), tau0, dtype=float)
        tau_max  = 1.0 / (self.rho * nn_L)
        tau_min  = tau_max / (2.0 * n)

        best_tour  = tsp.two_opt_improve(nn_tour)
        best_L     = tsp.tour_length(best_tour)
        history    = [best_L]
        t0         = time.perf_counter()

        for it in range(1, max_iter + 1):
            # ── Precompute τᵅ · ηᵝ once per generation ────────────────
            # Shape (n, n); row i = attractiveness of edge i→j for all j
            attract = (tau ** self.alpha) * (eta ** self.beta)

            # ── Construct all ant tours ────────────────────────────────
            tours   = np.empty((n_ants, n), dtype=int)
            lengths = np.empty(n_ants)
            rand_all = self.rng.random((n_ants, n - 1))   # pre-drawn random numbers

            for k in range(n_ants):
                tours[k]   = self._build_tour(n, attract, rand_all[k])
                lengths[k] = tsp.tour_length(tours[k])

            # ── Pheromone evaporation ──────────────────────────────────
            tau *= (1.0 - self.rho)

            # ── Deposit ────────────────────────────────────────────────
            if self.variant == 'AS':
                for k in range(n_ants):
                    delta = self.Q / lengths[k]
                    t = tours[k]
                    for i in range(n):
                        a, b = t[i], t[(i+1) % n]
                        tau[a, b] += delta
                        tau[b, a] += delta
            else:   # MMAS — only best ant deposits
                kb  = int(np.argmin(lengths))
                dep = tours[kb] if lengths[kb] < best_L else best_tour
                dL  = min(lengths[kb], best_L)
                t   = dep
                for i in range(n):
                    a, b = t[i], t[(i+1) % n]
                    tau[a, b] += self.Q / dL
                    tau[b, a] += self.Q / dL
                tau = np.clip(tau, tau_min, tau_max)

            # ── Update global best ─────────────────────────────────────
            kb = int(np.argmin(lengths))
            if lengths[kb] < best_L:
                best_L    = lengths[kb]
                best_tour = tours[kb].copy()
                # Light 2-opt on new global best
                best_tour = tsp.two_opt_improve(best_tour, max_passes=3)
                best_L    = tsp.tour_length(best_tour)

            history.append(best_L)

            if self.log_interval and it % self.log_interval == 0:
                elapsed = time.perf_counter() - t0
                print(f"[ACO-{self.variant}] iter={it:>4d}  best={best_L:.2f}"
                      f"  tau_max={tau.max():.3e}  elapsed={elapsed:.1f}s")

        return best_tour, best_L, history

    # ── Fast tour construction ──────────────────────────────────────────────

    def _build_tour(self, n: int, attract: np.ndarray,
                    rand_row: np.ndarray) -> np.ndarray:
        """
        Build one ant tour using precomputed attract matrix.
        Uses searchsorted on cumulative sums — avoids np.random.choice overhead.
        rand_row : pre-drawn uniform random numbers of shape (n-1,)
        """
        visited  = np.zeros(n, dtype=bool)
        tour     = np.empty(n, dtype=int)
        start    = int(self.rng.integers(0, n))
        tour[0]  = start
        visited[start] = True

        for step in range(1, n):
            cur  = tour[step - 1]
            row  = attract[cur].copy()
            row[visited] = 0.0          # mask already-visited cities
            total = row.sum()

            if total < 1e-300:
                # Fallback: pick first unvisited city
                candidates = np.where(~visited)[0]
                nxt = int(candidates[0])
            else:
                cumsum = np.cumsum(row)
                r      = rand_row[step - 1] * total
                nxt    = int(np.searchsorted(cumsum, r))
                # Safety: clamp and skip visited (rounding edge cases)
                nxt = min(nxt, n - 1)
                while visited[nxt]:
                    nxt = (nxt + 1) % n

            tour[step]   = nxt
            visited[nxt] = True

        return tour
