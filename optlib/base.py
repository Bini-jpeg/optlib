"""
optlib/base.py  —  Abstract base shared by every optimizer.

History now tracks (iteration, nfev, best, mean, std, elapsed) tuples so
convergence can be plotted vs either axis.  Every optimizer supports a
max_fev budget that terminates the run once exhausted.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ── Data containers ────────────────────────────────────────────────────────

@dataclass
class OptimizationHistory:
    """Per-iteration log.  nfev_values lets you plot vs evaluations, not iterations."""
    iterations:    List[int]   = field(default_factory=list)
    nfev_values:   List[int]   = field(default_factory=list)   # cumulative evals
    best_values:   List[float] = field(default_factory=list)
    mean_values:   List[float] = field(default_factory=list)
    std_values:    List[float] = field(default_factory=list)
    elapsed_times: List[float] = field(default_factory=list)
    extra:         List[Dict]  = field(default_factory=list)

    def record(self, iteration: int, best: float,
               mean: float = float('nan'), std: float = float('nan'),
               elapsed: float = 0.0, nfev: int = 0, **kv) -> None:
        self.iterations.append(int(iteration))
        self.nfev_values.append(int(nfev))
        self.best_values.append(float(best))
        self.mean_values.append(float(mean))
        self.std_values.append(float(std))
        self.elapsed_times.append(float(elapsed))
        self.extra.append(kv)

    def to_dict(self) -> Dict:
        return {
            "iterations":    self.iterations,
            "nfev_values":   self.nfev_values,
            "best_values":   self.best_values,
            "mean_values":   self.mean_values,
            "std_values":    self.std_values,
            "elapsed_times": self.elapsed_times,
        }

    def __len__(self) -> int:
        return len(self.iterations)


@dataclass
class OptimizationResult:
    """Unified result from every optimizer."""
    x:         np.ndarray          # best solution
    fun:       float               # best objective value
    nfev:      int                 # total function evaluations used
    nit:       int                 # total iterations run
    success:   bool
    message:   str
    history:   OptimizationHistory
    algorithm: str
    elapsed:   float               # wall-clock seconds

    def __repr__(self) -> str:
        arr = np.array2string(self.x, precision=4, max_line_width=60)
        return (
            f"OptimizationResult(\n"
            f"  algorithm = {self.algorithm}\n"
            f"  fun       = {self.fun:.6e}\n"
            f"  x         = {arr}\n"
            f"  nit/nfev  = {self.nit:,}/{self.nfev:,}\n"
            f"  elapsed   = {self.elapsed:.3f}s  "
            f"({self.nfev/max(self.elapsed,1e-9):.0f} eval/s)\n"
            f"  success   = {self.success}\n"
            f")"
        )


# ── Abstract base ──────────────────────────────────────────────────────────

class BaseOptimizer(ABC):
    """
    Common parent for every optimizer.

    Parameters
    ----------
    seed          : fixed RNG seed (None = non-deterministic).
    log_interval  : print one log line every N iterations; None = silent.
    store_history : collect per-iteration stats (disable for speed).
    """

    def __init__(
        self,
        seed:          Optional[int] = None,
        log_interval:  Optional[int] = None,
        store_history: bool = True,
    ) -> None:
        self.seed          = seed
        self.log_interval  = log_interval
        self.store_history = store_history
        self.rng           = np.random.default_rng(seed)
        self._name         = self.__class__.__name__

    # ── Bounds helpers ─────────────────────────────────────────────────────

    def _parse_bounds(self, bounds) -> Tuple[np.ndarray, np.ndarray]:
        b = np.asarray(bounds, dtype=float)
        if b.ndim == 1:
            b = b.reshape(-1, 2)
        lb, ub = b[:, 0], b[:, 1]
        if not np.all(lb < ub):
            raise ValueError("Every lower bound must be strictly < upper bound.")
        return lb, ub

    def _random_population(self, lb: np.ndarray, ub: np.ndarray, n: int) -> np.ndarray:
        return lb + self.rng.random((n, len(lb))) * (ub - lb)

    @staticmethod
    def _clip(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        return np.clip(x, lb, ub)

    @staticmethod
    def _bounce(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Reflective boundary repair — works for 1-D and 2-D arrays."""
        span = ub - lb
        lo   = (x - lb) % (2.0 * span)
        return np.clip(lb + np.where(lo <= span, lo, 2.0 * span - lo), lb, ub)

    # ── Budget / logging helpers ───────────────────────────────────────────

    def _budget_ok(self, nfev: int, max_fev: Optional[int]) -> bool:
        """Return True while budget is NOT exhausted (safe to continue)."""
        return max_fev is None or nfev < max_fev

    def _should_log(self, it: int) -> bool:
        return self.log_interval is not None and it % self.log_interval == 0

    def _log(self, it: int, best: float,
             nfev: int = 0, elapsed: float = 0.0, **kv) -> None:
        if not self._should_log(it):
            return
        parts = [f"[{self._name}] iter={it:>5d}  nfev={nfev:>7,}"
                 f"  best={best:.4e}  t={elapsed:.1f}s"]
        for k, v in kv.items():
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        print("  ".join(parts))

    # ── Interface ──────────────────────────────────────────────────────────

    @abstractmethod
    def optimize(
        self,
        func:     Callable[[np.ndarray], float],
        bounds:   Union[List[Tuple], np.ndarray],
        max_iter: int = 1000,
        max_fev:  Optional[int] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Minimise func over the hyper-rectangle defined by bounds.

        Stops when max_iter OR max_fev is reached, whichever comes first.
        """
        ...
