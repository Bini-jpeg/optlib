"""
demos/demo_utils.py
Shared utilities for all demo scripts.

Design contract
---------------
Every demo runs algorithms with a FIXED EVALUATION BUDGET (max_fev).
Results are always compared by:
  1. Quality at equal nfev  (convergence vs nfev plot)
  2. Wall-clock time        (timing bar chart)
  3. Evaluations per second (throughput — algorithm efficiency)

A "function evaluation" means exactly one call to the objective.
Population-based algorithms naturally use pop_size × n_iterations calls.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PALETTE = list(mcolors.TABLEAU_COLORS.values())


# ── Single-run wrapper ─────────────────────────────────────────────────────

def run_one(algo, func: Callable, bounds, budget: int,
            max_iter: int = 100_000, **kwargs):
    """
    Run *algo* with a fixed evaluation budget.
    Returns OptimizationResult with .elapsed and .nfev populated.
    """
    return algo.optimize(func, bounds, max_iter=max_iter, max_fev=budget, **kwargs)


# ── Multi-run suite ────────────────────────────────────────────────────────

def run_suite(
    algo_factory: Callable[[int], Dict],   # seed → {name: algo}
    func:  Callable,
    bounds,
    budget:  int,
    n_runs:  int   = 5,
    seed0:   int   = 42,
    max_iter: int  = 100_000,
    verbose:  bool = True,
) -> Dict[str, Dict]:
    """
    Run every algorithm n_runs times with independent seeds.

    Returns
    -------
    results[algo_name] = {
        'finals':    [best_f per run],
        'elapsed':   [seconds per run],
        'nfev':      [actual evals per run],
        'histories': [OptimizationHistory per run],
        'throughput':[nfev/s per run],
    }
    """
    results: Dict[str, Dict] = {}
    algo_names = list(algo_factory(seed0).keys())

    for name in algo_names:
        results[name] = dict(finals=[], elapsed=[], nfev=[], histories=[], throughput=[])

    for run in range(n_runs):
        seed = seed0 + run * 17
        algos = algo_factory(seed)
        for name, algo in algos.items():
            if verbose:
                print(f"  {name:<18}  run {run+1}/{n_runs}  ", end="", flush=True)
            try:
                r = run_one(algo, func, bounds, budget=budget, max_iter=max_iter)
                results[name]['finals'].append(r.fun)
                results[name]['elapsed'].append(r.elapsed)
                results[name]['nfev'].append(r.nfev)
                results[name]['histories'].append(r.history)
                throughput = r.nfev / max(r.elapsed, 1e-9)
                results[name]['throughput'].append(throughput)
                if verbose:
                    print(f"best={r.fun:.4e}  nfev={r.nfev:>7,}"
                          f"  t={r.elapsed:.2f}s  ({throughput/1e3:.1f}k ev/s)")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                results[name]['finals'].append(float('inf'))
                results[name]['elapsed'].append(float('nan'))
                results[name]['nfev'].append(0)
                results[name]['throughput'].append(0.0)

    return results


# ── Console table ──────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Dict], title: str = "") -> None:
    """Print aligned summary: algo | median best | nfev | time | ev/s."""
    if title:
        print(f"\n{'═'*68}")
        print(f"  {title}")
    print(f"{'═'*68}")
    hdr = f"  {'Algorithm':<18}  {'Median Best':>12}  {'nfev':>7}  {'Time (s)':>9}  {'kev/s':>7}"
    print(hdr); print(f"  {'─'*64}")
    rows = []
    for name, r in results.items():
        med_f  = float(np.median(r['finals']))
        med_t  = float(np.nanmedian(r['elapsed']))
        med_n  = int(np.median(r['nfev']))
        med_tp = float(np.median(r['throughput'])) / 1e3
        rows.append((med_f, name, med_f, med_n, med_t, med_tp))
    for _, name, mf, mn, mt, mtp in sorted(rows):
        print(f"  {name:<18}  {mf:>12.4e}  {mn:>7,}  {mt:>9.3f}  {mtp:>7.1f}")
    print(f"{'═'*68}\n")


# ── Plot: convergence vs nfev ──────────────────────────────────────────────

def plot_convergence_nfev(
    results:  Dict[str, Dict],
    budget:   int,
    title:    str  = "Convergence vs Function Evaluations",
    log_y:    bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Median convergence curve (best value vs nfev) with IQR shading.
    x-axis = cumulative function evaluations — the only fair comparison axis.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (name, r) in enumerate(results.items()):
        color = PALETTE[idx % len(PALETTE)]
        # Build (nfev, best) traces, padded to budget length
        traces = []
        for h in r['histories']:
            if not h.nfev_values:
                continue
            xs = [0] + list(h.nfev_values)
            ys = [h.best_values[0]] + list(h.best_values)
            # Extend flat to budget
            xs.append(budget); ys.append(ys[-1])
            traces.append((np.array(xs), np.array(ys)))

        if not traces:
            continue

        # Interpolate all traces to a common nfev grid
        grid  = np.linspace(0, budget, 500)
        mat   = np.array([np.interp(grid, tr[0], tr[1]) for tr in traces])
        med   = np.median(mat, axis=0)
        p25   = np.percentile(mat, 25, axis=0)
        p75   = np.percentile(mat, 75, axis=0)

        ax.plot(grid, med, color=color, linewidth=2.0, label=name)
        ax.fill_between(grid, np.maximum(p25, 1e-15), p75,
                        alpha=0.12, color=color)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Function Evaluations (nfev)", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlim(0, budget)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── Plot: timing comparison ────────────────────────────────────────────────

def plot_timing(
    results:  Dict[str, Dict],
    title:    str = "Algorithm Timing",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel figure:
      Left  — median wall-clock time per algorithm (horizontal bars).
      Right — median evaluations/second (throughput).
    """
    names    = list(results.keys())
    med_time = [float(np.nanmedian(results[n]['elapsed']))   for n in names]
    med_tp   = [float(np.median(results[n]['throughput']))/1e3 for n in names]  # kev/s

    order = np.argsort(med_time)
    names_s   = [names[i]    for i in order]
    med_time_s = [med_time[i] for i in order]
    med_tp_s   = [med_tp[i]   for i in order]
    colors_s   = [PALETTE[i % len(PALETTE)] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.5*len(names)+3))

    # Left: time
    bars = ax1.barh(names_s, med_time_s, color=colors_s, edgecolor="white", alpha=0.85)
    ax1.bar_label(bars, fmt="%.2fs", padding=3, fontsize=8)
    ax1.set_xlabel("Median wall-clock time (s)", fontsize=11)
    ax1.set_title("Runtime", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Right: throughput
    bars2 = ax2.barh(names_s, med_tp_s, color=colors_s, edgecolor="white", alpha=0.85)
    ax2.bar_label(bars2, fmt="%.1f k/s", padding=3, fontsize=8)
    ax2.set_xlabel("Median throughput (k evaluations / s)", fontsize=11)
    ax2.set_title("Throughput (higher = faster per eval)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── Plot: quality vs time scatter ──────────────────────────────────────────

def plot_quality_vs_time(
    results:  Dict[str, Dict],
    title:    str = "Quality vs Time",
    log_y:    bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter: x = median time, y = median best value.
    Bottom-left corner = best (fast AND accurate).
    Each point is labelled with the algorithm name.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (name, r) in enumerate(results.items()):
        mt = float(np.nanmedian(r['elapsed']))
        mf = float(np.nanmedian(r['finals']))
        if np.isfinite(mt) and np.isfinite(mf):
            ax.scatter(mt, mf, s=120, color=PALETTE[idx % len(PALETTE)],
                       zorder=5, label=name)
            ax.annotate(name, (mt, mf), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Median wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Median best objective value", fontsize=12)
    ax.set_title(title + "\n(bottom-left = fast AND accurate)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
