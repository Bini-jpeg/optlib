"""
optlib/logging_utils/logger.py
Structured, non-spammy run logger with optional CSV export.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class RunLogger:
    """
    Tracks and optionally prints structured optimisation progress.

    Design goals
    ------------
    * Zero output unless you opt-in (log_interval != None).
    * Exactly one line per logged iteration – no essay-length messages.
    * Optional CSV / txt export for post-run analysis.
    * Plays well with Jupyter (uses print, not logging module).

    Parameters
    ----------
    algorithm    : human-readable algorithm name.
    problem      : problem / function name (optional).
    log_interval : print every N iterations; None = silent.
    log_file     : path to write a log file (None = no file).
    verbose      : extra columns (sigma, pop_size, …) printed if True.
    """

    # Header template — fixed columns always present
    _COLS = ("iter", "best", "mean", "std", "elapsed_s")

    def __init__(
        self,
        algorithm:    str,
        problem:      str  = "",
        log_interval: Optional[int] = None,
        log_file:     Optional[str] = None,
        verbose:      bool = False,
    ):
        self.algorithm    = algorithm
        self.problem      = problem
        self.log_interval = log_interval
        self.log_file     = Path(log_file) if log_file else None
        self.verbose      = verbose
        self._start       = time.perf_counter()
        self._rows: list  = []
        self._header_printed = False

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fh  = open(self.log_file, "w", newline="")
            self._csv = csv.writer(self._fh)
        else:
            self._fh  = None
            self._csv = None

    # ── Public API ─────────────────────────────────────────────────────────

    def log(self, iteration: int, best: float,
            mean: float = float('nan'), std: float = float('nan'),
            **extra) -> None:
        """Record one iteration; print if logging is active."""
        elapsed = time.perf_counter() - self._start
        row = {
            "iter":      iteration,
            "best":      best,
            "mean":      mean,
            "std":       std,
            "elapsed_s": elapsed,
            **extra,
        }
        self._rows.append(row)

        if self._csv:
            if not self._header_printed:
                self._csv.writerow(list(row.keys()))
                self._header_printed = True
            self._csv.writerow([
                f"{v:.6e}" if isinstance(v, float) else v
                for v in row.values()
            ])

        if self.log_interval and iteration % self.log_interval == 0:
            self._print_row(iteration, best, mean, std, elapsed, extra)

    def summary(self, result) -> None:
        """Print a compact end-of-run summary."""
        tag = f"[{self.algorithm}]"
        if self.problem:
            tag += f" {self.problem}"
        print(f"\n{'─'*60}")
        print(f"{tag}  FINAL RESULT")
        print(f"  best fun = {result.fun:.6e}")
        print(f"  nit/nfev = {result.nit}/{result.nfev}")
        print(f"  elapsed  = {result.elapsed:.3f} s")
        print(f"{'─'*60}\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()

    def to_dict(self) -> Dict[str, list]:
        """Return all recorded rows as column lists."""
        if not self._rows:
            return {}
        keys = list(self._rows[0].keys())
        return {k: [r.get(k, float('nan')) for r in self._rows] for k in keys}

    # ── Internal ───────────────────────────────────────────────────────────

    def _print_row(self, it, best, mean, std, elapsed, extra):
        if not self._header_printed and sys.stdout.isatty():
            hdr  = f"{'iter':>6}  {'best':>12}  {'mean':>12}  {'std':>10}  {'sec':>7}"
            if self.verbose and extra:
                hdr += "  " + "  ".join(f"{k:>10}" for k in extra)
            print(f"\n[{self.algorithm}] {self.problem}")
            print(hdr)
            print("─" * len(hdr))
            self._header_printed = True

        line = (f"{it:>6d}  {best:>12.4e}  "
                f"{mean:>12.4e}  {std:>10.4e}  {elapsed:>7.2f}")
        if self.verbose and extra:
            for v in extra.values():
                line += f"  {v:>10.4g}" if isinstance(v, float) else f"  {str(v):>10}"
        print(line)
