"""
optlib/benchmarks/discrete_problems.py
Discrete combinatorial problems used in the demo scripts.
"""
from __future__ import annotations

import math
import numpy as np


# ── Travelling Salesman Problem ────────────────────────────────────────────

class TSP:
    """Symmetric Euclidean TSP instance.

    Parameters
    ----------
    cities : (n, 2) array  –  x, y coordinates.
    seed   : int | None    –  for random instance generation.
    """

    def __init__(self, cities: np.ndarray | None = None,
                 n_cities: int = 20, seed: int | None = 42):
        if cities is not None:
            self.cities = np.asarray(cities, dtype=float)
        else:
            rng = np.random.default_rng(seed)
            self.cities = rng.random((n_cities, 2)) * 100.0
        self.n = len(self.cities)
        # Precompute distance matrix
        diff      = self.cities[:, None, :] - self.cities[None, :, :]
        self.dist = np.sqrt((diff**2).sum(axis=-1))

    def tour_length(self, tour: np.ndarray) -> float:
        """Total distance of a closed tour (list/array of city indices)."""
        t    = np.asarray(tour, dtype=int)
        next_t = np.roll(t, -1)
        return float(self.dist[t, next_t].sum())

    def nearest_neighbour(self, start: int = 0) -> np.ndarray:
        """Greedy nearest-neighbour heuristic — good starting tour."""
        visited = [False] * self.n
        tour    = [start]
        visited[start] = True
        for _ in range(self.n - 1):
            cur  = tour[-1]
            best = -1
            best_d = math.inf
            for j in range(self.n):
                if not visited[j] and self.dist[cur, j] < best_d:
                    best_d = self.dist[cur, j]
                    best   = j
            tour.append(best)
            visited[best] = True
        return np.array(tour, dtype=int)

    def two_opt_improve(self, tour: np.ndarray, max_passes: int = 100) -> np.ndarray:
        """Local 2-opt improvement (used as post-processing or neighbourhood)."""
        t       = tour.copy()
        improved = True
        passes   = 0
        while improved and passes < max_passes:
            improved = False
            passes  += 1
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    if j - i == 1:
                        continue
                    d_before = self.dist[t[i-1], t[i]] + self.dist[t[j-1], t[j % self.n]]
                    d_after  = self.dist[t[i-1], t[j-1]] + self.dist[t[i], t[j % self.n]]
                    if d_after < d_before - 1e-10:
                        t[i:j] = t[i:j][::-1]
                        improved = True
        return t


# ── 0/1 Knapsack Problem ───────────────────────────────────────────────────

class Knapsack:
    """0/1 Knapsack Problem.

    Parameters
    ----------
    weights  : 1-D array of item weights.
    values   : 1-D array of item values.
    capacity : total weight capacity.
    """

    def __init__(self, weights: np.ndarray | None = None,
                 values: np.ndarray | None = None,
                 capacity: float | None = None,
                 n_items: int = 20, seed: int | None = 42):
        if weights is not None:
            self.weights  = np.asarray(weights, dtype=float)
            self.values   = np.asarray(values,  dtype=float)
            self.capacity = float(capacity)
        else:
            rng = np.random.default_rng(seed)
            self.weights  = rng.uniform(1.0, 10.0, n_items)
            self.values   = rng.uniform(1.0, 10.0, n_items)
            self.capacity = 0.5 * self.weights.sum()
        self.n = len(self.weights)

    def evaluate(self, selection: np.ndarray) -> float:
        """Negative total value (for minimisation); penalise overweight."""
        s       = np.asarray(selection, dtype=float)
        weight  = np.dot(s, self.weights)
        value   = np.dot(s, self.values)
        penalty = max(0.0, weight - self.capacity) * 10.0
        return float(-(value - penalty))

    def dp_optimal(self):
        """Exact DP solution (only feasible for small instances)."""
        n = self.n
        cap = int(self.capacity)
        w = self.weights.astype(int)
        v = self.values
        dp = np.zeros((n + 1, cap + 1))
        keep = np.zeros((n + 1, cap + 1), dtype=bool)
        for i in range(1, n + 1):
            for c in range(cap + 1):
                dp[i, c] = dp[i-1, c]
                if w[i-1] <= c and dp[i-1, c - w[i-1]] + v[i-1] > dp[i, c]:
                    dp[i, c] = dp[i-1, c - w[i-1]] + v[i-1]
                    keep[i, c] = True
        # backtrack
        sel = np.zeros(n, dtype=int)
        c = cap
        for i in range(n, 0, -1):
            if keep[i, c]:
                sel[i-1] = 1
                c -= w[i-1]
        return sel, float(dp[n, cap])
