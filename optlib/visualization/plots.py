"""
optlib/visualization/plots.py
All plotting utilities in one place.  Every function returns a Figure so
the caller can save or display it.  Matplotlib backend is non-interactive
by default (Agg) to work in headless/server environments.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # safe headless default

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec


# ── Colour palette ─────────────────────────────────────────────────────────

PALETTE = list(mcolors.TABLEAU_COLORS.values())

def _fig(w=10, h=6):
    return plt.figure(figsize=(w, h))


# ── 1. Convergence curves ──────────────────────────────────────────────────

def plot_convergence(
    results: Dict[str, "OptimizationResult"],   # name -> result
    title:   str  = "Convergence",
    log_y:   bool = True,
    show_std: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot best-value vs iteration for multiple algorithms.
    If the result has multiple runs, show mean ± std band.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, result) in enumerate(results.items()):
        h    = result.history
        iters = h.iterations
        best  = h.best_values
        color = PALETTE[idx % len(PALETTE)]

        if show_std and any(not math.isnan(v) for v in h.std_values):
            means = np.array(h.mean_values)
            stds  = np.array(h.std_values)
            ax.fill_between(iters,
                            np.maximum(means - stds, 1e-15),
                            means + stds,
                            alpha=0.15, color=color)

        ax.plot(iters, best, label=name, color=color, linewidth=1.8)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── 2. Algorithm comparison (box plot) ────────────────────────────────────

def plot_comparison_boxplot(
    data:      Dict[str, List[float]],   # algo_name -> [final values from N runs]
    title:     str = "Final Value Distribution",
    log_y:     bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side box plots for comparing final objective values.
    """
    names = list(data.keys())
    vals  = [data[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 6))
    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── 3. Fitness landscape (2-D functions only) ──────────────────────────────

def plot_landscape_2d(
    func,
    bounds:       Tuple,           # ((x_lo, x_hi), (y_lo, y_hi))
    trajectory:   Optional[np.ndarray] = None,   # (T, 2)
    population:   Optional[np.ndarray] = None,   # (N, 2)  final population
    resolution:   int  = 200,
    title:        str  = "Fitness Landscape",
    log_z:        bool = True,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Filled contour plot of a 2-D objective function with optional
    solution trajectory and final population scatter.
    """
    (xl, xh), (yl, yh) = bounds
    xs = np.linspace(xl, xh, resolution)
    ys = np.linspace(yl, yh, resolution)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array([[func(np.array([x, y])) for x in xs] for y in ys])

    fig, ax = plt.subplots(figsize=(8, 7))
    Z_plot  = np.log1p(ZZ - ZZ.min()) if log_z else ZZ
    cf = ax.contourf(XX, YY, Z_plot, levels=50, cmap="viridis", alpha=0.85)
    plt.colorbar(cf, ax=ax, label="log(1+f)" if log_z else "f(x)")
    ax.contour(XX, YY, Z_plot, levels=15, colors="white", linewidths=0.3, alpha=0.4)

    if trajectory is not None and len(trajectory) > 1:
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                "w-o", markersize=3, linewidth=1.2, label="trajectory", alpha=0.8)
        ax.plot(*trajectory[0],  "go", markersize=8, label="start")
        ax.plot(*trajectory[-1], "r*", markersize=12, label="end")

    if population is not None:
        ax.scatter(population[:, 0], population[:, 1],
                   c="yellow", s=20, alpha=0.6, label="population")

    ax.set_xlabel("x₁", fontsize=12); ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── 4. TSP tour visualisation ──────────────────────────────────────────────

def plot_tsp_tour(
    cities:    np.ndarray,
    tour:      np.ndarray,
    length:    float,
    title:     str = "TSP Tour",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    t   = list(tour) + [tour[0]]    # close the tour
    xs  = cities[t, 0]; ys = cities[t, 1]
    ax.plot(xs, ys, "b-o", markersize=7, linewidth=1.5, zorder=2)
    ax.scatter(cities[:, 0], cities[:, 1], s=80, c="red", zorder=3)
    for i, (x, y) in enumerate(cities):
        ax.text(x + 0.5, y + 0.5, str(i), fontsize=8)
    ax.set_title(f"{title}  (length = {length:.2f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── 5. Benchmark summary table ─────────────────────────────────────────────

def plot_benchmark_table(
    summary:   Dict[str, Dict[str, float]],  # {algo: {func: best_val}}
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Colour-coded table: rows = algorithms, columns = test functions.
    Green = best on that function, red = worst.
    """
    algos = list(summary.keys())
    funcs = list(next(iter(summary.values())).keys())
    data  = np.array([[summary[a].get(f, np.nan) for f in funcs] for a in algos])

    fig, ax = plt.subplots(figsize=(max(8, 1.4*len(funcs)), max(4, 0.5*len(algos))))
    ax.axis("off")

    col_colors = []
    for j in range(len(funcs)):
        col = data[:, j]
        best = np.nanmin(col)
        worst = np.nanmax(col)
        span = worst - best + 1e-30
        col_colors.append([(0.8 * (1 - (v-best)/span) + 0.2*1,
                            0.8 * ((v-best)/span) * 0.3 + 0.2,
                            0.2) for v in col])

    cell_colors = [["white" for _ in funcs] for _ in algos]
    for j, col in enumerate(col_colors):
        for i, rgb in enumerate(col):
            g = 1 - (data[i,j] - np.nanmin(data[:,j])) / (np.nanmax(data[:,j]) - np.nanmin(data[:,j]) + 1e-30)
            cell_colors[i][j] = (0.2 + 0.6*g, 0.7*g + 0.1, 0.2)

    tbl = ax.table(
        cellText   =[[f"{v:.3e}" if not np.isnan(v) else "—" for v in row] for row in data],
        rowLabels  = algos,
        colLabels  = [f[:12] for f in funcs],
        cellColours= cell_colors,
        loc        = "center",
        cellLoc    = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)

    ax.set_title("Benchmark Results  (green=best, red=worst per column)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── 6. Bayesian Optimisation — surrogate visualisation (1-D) ──────────────

def plot_bayesian_surrogate_1d(
    gp,
    X_obs:    np.ndarray,
    Y_obs:    np.ndarray,
    x_range:  Tuple[float, float],
    acq_fn:   Optional = None,
    title:    str = "GP Surrogate",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """1-D surrogate + acquisition overlay."""
    xs   = np.linspace(x_range[0], x_range[1], 300)[:, None]
    mu, std = gp.predict(xs)

    fig, axes = plt.subplots(2 if acq_fn else 1, 1,
                             figsize=(10, 8 if acq_fn else 5), sharex=True)
    ax = axes[0] if acq_fn else axes

    ax.plot(xs[:, 0], mu, "b-", label="GP mean", linewidth=2)
    ax.fill_between(xs[:, 0], mu - 2*std, mu + 2*std,
                    alpha=0.3, color="blue", label="±2σ")
    ax.scatter(X_obs[:, 0], Y_obs, c="red", s=60, zorder=5, label="observations")
    ax.set_ylabel("f(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=13, fontweight="bold")

    if acq_fn:
        acq = -acq_fn(xs)   # negate back to positive "desirability"
        axes[1].plot(xs[:, 0], acq, "g-", linewidth=2)
        axes[1].set_ylabel("Acquisition"); axes[1].set_xlabel("x")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── 7. Multi-run convergence with percentile bands ─────────────────────────

def plot_multirun_convergence(
    runs:     Dict[str, List[List[float]]],  # {algo: [[run1_vals], [run2_vals],...]}
    title:    str  = "Multi-Run Convergence",
    log_y:    bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, all_runs) in enumerate(runs.items()):
        color = PALETTE[idx % len(PALETTE)]
        max_len = max(len(r) for r in all_runs)
        # Pad with last value
        mat = np.array([r + [r[-1]] * (max_len - len(r)) for r in all_runs])
        med = np.median(mat, axis=0)
        p25 = np.percentile(mat, 25, axis=0)
        p75 = np.percentile(mat, 75, axis=0)
        x   = np.arange(1, max_len + 1)
        ax.plot(x, med, color=color, linewidth=2, label=name)
        ax.fill_between(x, np.maximum(p25, 1e-15), p75,
                        alpha=0.15, color=color)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Value (median ± IQR)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
