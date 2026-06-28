"""
optlib/preprocessing/preprocessing.py
Data normalisation, scaling, and constraint utilities for optimisers.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple


# ── Scalers ────────────────────────────────────────────────────────────────

class MinMaxScaler:
    """Scale each feature to [0, 1].  Useful when bounds vary wildly."""

    def __init__(self):
        self._lo = self._hi = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self._lo = X.min(axis=0)
        self._hi = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        span = self._hi - self._lo
        span = np.where(span == 0, 1.0, span)
        return (X - self._lo) / span

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * (self._hi - self._lo) + self._lo

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScaler:
    """Zero-mean, unit-variance scaling."""

    def __init__(self, eps: float = 1e-8):
        self._mu = self._sigma = None
        self._eps = eps

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + self._eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self._sigma + self._mu

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ── Bounds wrapper ─────────────────────────────────────────────────────────

class BoundsNormalizer:
    """Map an arbitrary search space to [0,1]^d and back.

    Wraps an objective so the algorithm always sees a unit hypercube.
    """

    def __init__(self, bounds: np.ndarray):
        b        = np.asarray(bounds, dtype=float)
        self._lb = b[:, 0]
        self._ub = b[:, 1]
        self._span = self._ub - self._lb

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return (x - self._lb) / self._span

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        return self._lb + u * self._span

    def wrap(self, func: Callable) -> Callable:
        """Return a version of *func* that takes unit-cube inputs."""
        def wrapped(u):
            return func(self.from_unit(u))
        return wrapped

    @property
    def unit_bounds(self) -> np.ndarray:
        d = len(self._lb)
        return np.column_stack([np.zeros(d), np.ones(d)])


# ── Constraint handling ────────────────────────────────────────────────────

class PenaltyConstraintHandler:
    """Add penalty terms for constraint violations.

    Constraints should be expressed as g_i(x) <= 0.
    """

    def __init__(self, constraints: List[Callable], penalty: float = 1e4):
        self.constraints = constraints
        self.penalty     = penalty

    def penalise(self, x: np.ndarray, base_val: float) -> float:
        total = base_val
        for g in self.constraints:
            viol = max(0.0, float(g(x)))
            total += self.penalty * viol**2
        return total

    def wrap(self, func: Callable) -> Callable:
        def wrapped(x):
            return self.penalise(x, func(x))
        return wrapped

    def feasible(self, x: np.ndarray) -> bool:
        return all(float(g(x)) <= 1e-6 for g in self.constraints)


class AdaptivePenaltyHandler:
    """Penalty coefficient that grows with the iteration counter.
    Good for DE / GA where constraint satisfaction improves over time.
    """

    def __init__(self, constraints: List[Callable],
                 base_penalty: float = 1.0, growth: float = 1.01):
        self.constraints  = constraints
        self.base_penalty = base_penalty
        self.growth       = growth
        self._iter        = 0

    def step(self):
        """Call once per algorithm iteration."""
        self._iter += 1

    def current_penalty(self) -> float:
        return self.base_penalty * (self.growth ** self._iter)

    def penalise(self, x: np.ndarray, base_val: float) -> float:
        p = self.current_penalty()
        return base_val + p * sum(
            max(0.0, float(g(x)))**2 for g in self.constraints)

    def wrap(self, func: Callable) -> Callable:
        def wrapped(x):
            return self.penalise(x, func(x))
        return wrapped


# ── Utility ────────────────────────────────────────────────────────────────

def latin_hypercube_sample(n: int, d: int, bounds: np.ndarray,
                           seed: int | None = None) -> np.ndarray:
    """Generate an LHS design of *n* points in a *d*-dim space.

    Better space-filling than purely random initialisation.
    """
    rng = np.random.default_rng(seed)
    lb  = bounds[:, 0]
    ub  = bounds[:, 1]
    pts = np.empty((n, d))
    for j in range(d):
        perm      = rng.permutation(n)
        u         = (perm + rng.random(n)) / n
        pts[:, j] = lb[j] + u * (ub[j] - lb[j])
    return pts
