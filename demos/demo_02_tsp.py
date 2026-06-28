"""
Demo 2 — Travelling Salesman Problem (equal evaluation budget).

Budget : BUDGET = 60,000 tour-length evaluations per algorithm.

All four algorithms are compared on the same 25-city instance.
Convergence is plotted on the nfev x-axis — the only fair comparison axis.

Outputs (in outputs/demo02/):
  convergence_vs_nfev.png
  timing.png
  quality_vs_time.png
  tour_<algo>.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import math, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optlib.benchmarks.discrete_problems import TSP
from optlib.visualization.plots import plot_tsp_tour, PALETTE
from demos.demo_utils import print_summary_table, plot_timing, plot_quality_vs_time

OUTPUT   = Path("outputs/demo02"); OUTPUT.mkdir(parents=True, exist_ok=True)
SEED     = 42
N_CITIES = 25
BUDGET   = 60_000


# ── Budget-counting TSP wrapper ─────────────────────────────────────────────

class CountedTSP:
    """Wraps TSP, counts every tour_length() call, stops at budget."""

    class BudgetExhausted(Exception):
        pass

    def __init__(self, tsp: TSP, budget: int):
        self._tsp   = tsp
        self.budget = budget
        self.nfev   = 0
        self.best   = math.inf
        self.trace: list[tuple[int, float]] = []  # (nfev, best_length)

    @property
    def dist(self):   return self._tsp.dist
    @property
    def n(self):      return self._tsp.n
    @property
    def cities(self): return self._tsp.cities

    def tour_length(self, tour: np.ndarray) -> float:
        if self.nfev >= self.budget:
            raise CountedTSP.BudgetExhausted()
        L = self._tsp.tour_length(tour)
        self.nfev += 1
        if L < self.best:
            self.best = L
            self.trace.append((self.nfev, L))
        return L

    def nearest_neighbour(self, start: int = 0) -> np.ndarray:
        tour = self._tsp.nearest_neighbour(start)
        try:
            self.tour_length(tour)
        except CountedTSP.BudgetExhausted:
            pass
        return tour

    def two_opt_improve(self, tour: np.ndarray, max_passes: int = 100) -> np.ndarray:
        t = tour.copy(); n = self._tsp.n
        improved = True; passes = 0
        while improved and passes < max_passes:
            improved = False; passes += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    db = self._tsp.dist[t[i-1], t[i]]   + self._tsp.dist[t[j-1], t[j % n]]
                    da = self._tsp.dist[t[i-1], t[j-1]] + self._tsp.dist[t[i],   t[j % n]]
                    if da < db - 1e-10:
                        t[i:j] = t[i:j][::-1]
                        improved = True
            if improved:
                try:
                    self.tour_length(t)
                except CountedTSP.BudgetExhausted:
                    return t
        return t


# ── Algorithm implementations ────────────────────────────────────────────────

def ga_tsp(ctsp, pop_size=100, max_iter=600, p_mut=0.02, elite=5,
           seed=None, log_interval=None):
    rng = np.random.default_rng(seed); n = ctsp.n
    pop  = np.array([rng.permutation(n) for _ in range(pop_size)])
    try:
        fits = np.array([ctsp.tour_length(t) for t in pop])
    except CountedTSP.BudgetExhausted:
        fits = np.array([ctsp._tsp.tour_length(t) for t in pop])
    best_tour = pop[np.argmin(fits)].copy(); best_L = fits.min()

    def ox(p1, p2):
        a, b = sorted(rng.integers(0, n, 2))
        child = [-1] * n; child[a:b+1] = list(p1[a:b+1])
        used = set(child[a:b+1]); ptr = b + 1
        for g in np.roll(p2, -(b+1)):
            if g not in used:
                child[ptr % n] = g; used.add(g); ptr += 1
        return np.array(child, dtype=int)

    def mut(t):
        if rng.random() < p_mut * n:
            i, j = rng.integers(0, n, 2); t[i], t[j] = t[j], t[i]
        return t

    for it in range(1, max_iter + 1):
        order  = np.argsort(fits)
        elites = pop[order[:elite]].copy(); ef = fits[order[:elite]].copy()
        offs   = []
        while len(offs) < pop_size - elite:
            ix = rng.integers(0, pop_size, 6)
            p1 = ix[np.argmin(fits[ix[:3]])]; p2 = ix[3 + np.argmin(fits[ix[3:]])]
            offs.append(mut(ox(pop[p1], pop[p2])))
        offs = np.array(offs)
        try:
            of = np.array([ctsp.tour_length(t) for t in offs])
        except CountedTSP.BudgetExhausted:
            break
        pop  = np.vstack([elites, offs]); fits = np.concatenate([ef, of])
        idx  = np.argmin(fits)
        if fits[idx] < best_L:
            best_L = fits[idx]; best_tour = pop[idx].copy()
        if log_interval and it % log_interval == 0:
            print(f"  [GA-TSP]   iter={it:>4d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}")
    return best_tour, best_L


def sa_tsp(ctsp, max_iter=60_000, T0=None, alpha=0.9998,
           seed=None, log_interval=None):
    rng  = np.random.default_rng(seed); n = ctsp.n
    tour = ctsp.nearest_neighbour(); L = ctsp._tsp.tour_length(tour)
    T    = T0 or L * 0.15
    best_tour = tour.copy(); best_L = L
    for it in range(1, max_iter + 1):
        i, j = sorted(rng.integers(0, n, 2))
        if i == j:
            continue
        nt = tour.copy(); nt[i:j+1] = nt[i:j+1][::-1]
        try:
            nL = ctsp.tour_length(nt)
        except CountedTSP.BudgetExhausted:
            break
        delta = nL - L
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            tour = nt; L = nL
        if L < best_L:
            best_L = L; best_tour = tour.copy()
        T *= alpha
        if log_interval and it % log_interval == 0:
            print(f"  [SA-TSP]   iter={it:>5d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}  T={T:.3f}")
    return best_tour, best_L


def tabu_tsp(ctsp, max_iter=1000, tabu_tenure=15, max_moves=60,
             seed=None, log_interval=None):
    rng  = np.random.default_rng(seed); n = ctsp.n
    tour = ctsp.nearest_neighbour()
    tour = ctsp.two_opt_improve(tour, max_passes=3)
    L    = ctsp._tsp.tour_length(tour)
    best_tour = tour.copy(); best_L = L
    tabu: dict = {}; no_impr = 0

    all_pairs = [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]

    for it in range(1, max_iter + 1):
        pairs = [all_pairs[k] for k in rng.permutation(len(all_pairs))[:max_moves]]
        bc = None; bcL = math.inf; bm = None
        for (i, j) in pairs:
            cand = tour.copy(); cand[i:j+1] = cand[i:j+1][::-1]
            try:
                cL = ctsp.tour_length(cand)
            except CountedTSP.BudgetExhausted:
                return best_tour, best_L
            if (tabu.get((i, j), 0) < it or cL < best_L) and cL < bcL:
                bc = cand; bcL = cL; bm = (i, j)
        if bc is None:
            break
        tour = bc; L = bcL
        if bm:
            tabu[bm] = it + tabu_tenure
        if L < best_L:
            best_L = L; best_tour = tour.copy(); no_impr = 0
        else:
            no_impr += 1
        if no_impr > 150:
            tour = best_tour.copy(); L = best_L; no_impr = 0
        if log_interval and it % log_interval == 0:
            print(f"  [Tabu]     iter={it:>4d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}")
    return best_tour, best_L


def aco_tsp(ctsp, n_ants=25, max_iter=2400, alpha=1.0, beta=5.0, rho=0.1,
            seed=None, log_interval=None):
    from optlib.algorithms.discrete.ant_colony import AntColonyTSP
    # Wrap the counted TSP so ACO's tour_length calls go through the counter.
    # AntColonyTSP accepts a TSP-like object; monkey-patch tour_length.
    rng  = np.random.default_rng(seed)
    n    = ctsp.n
    dist = ctsp.dist
    eta  = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (dist + 1e-10))

    nn_L  = ctsp._tsp.tour_length(ctsp._tsp.nearest_neighbour())
    tau0  = 1.0 / (n * nn_L)
    tau   = np.full((n, n), tau0, dtype=float)
    tmax  = 1.0 / (rho * nn_L)
    tmin  = tmax / (2.0 * n)

    best_tour = ctsp._tsp.two_opt_improve(ctsp._tsp.nearest_neighbour())
    best_L    = ctsp._tsp.tour_length(best_tour)

    # Precompute η^β once (constant)
    eta_b = eta ** beta

    def build_tour(tau_a_row_i, rand_vals):
        """Build one tour; tau_a_row_i = τᵅ[current_city] updated per step."""
        visited = np.zeros(n, dtype=bool)
        tour    = np.empty(n, dtype=int)
        start   = int(rng.integers(0, n))
        tour[0] = start; visited[start] = True
        for step in range(1, n):
            cur  = tour[step - 1]
            row  = tau_a[cur] * eta_b[cur]
            row  = row * (~visited)
            total = row.sum()
            if total < 1e-300:
                nxt = int(np.where(~visited)[0][0])
            else:
                cs  = np.cumsum(row)
                r   = rand_vals[step - 1] * total
                nxt = int(np.searchsorted(cs, r))
                nxt = min(nxt, n - 1)
                while visited[nxt]:
                    nxt = (nxt + 1) % n
            tour[step] = nxt; visited[nxt] = True
        return tour

    for it in range(1, max_iter + 1):
        tau_a    = tau ** alpha          # precompute once per generation
        rand_all = rng.random((n_ants, n - 1))
        tours    = [build_tour(tau_a, rand_all[k]) for k in range(n_ants)]
        try:
            lengths = np.array([ctsp.tour_length(t) for t in tours])
        except CountedTSP.BudgetExhausted:
            break

        tau *= (1.0 - rho)
        kb   = int(np.argmin(lengths))
        dep  = tours[kb] if lengths[kb] < best_L else best_tour
        dL   = min(lengths[kb], best_L)
        for i in range(n):
            a, b = dep[i], dep[(i + 1) % n]
            tau[a, b] += 1.0 / dL
            tau[b, a] += 1.0 / dL
        tau = np.clip(tau, tmin, tmax)

        if lengths[kb] < best_L:
            best_L    = lengths[kb]
            best_tour = tours[kb].copy()
            best_tour = ctsp.two_opt_improve(best_tour, max_passes=3)
            best_L    = ctsp._tsp.tour_length(best_tour)

        if log_interval and it % log_interval == 0:
            print(f"  [ACO-MMAS] iter={it:>4d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}")
    return best_tour, best_L


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tsp  = TSP(n_cities=N_CITIES, seed=SEED)
    nn_L = tsp.tour_length(tsp.nearest_neighbour())
    print(f"TSP: {N_CITIES} cities  |  NN heuristic = {nn_L:.2f}")
    print(f"Fixed evaluation budget : {BUDGET:,} tour-length calls\n")

    POP, ANTS, MOVES = 100, 25, 60
    print(f"  GA      : pop={POP} × {BUDGET//POP} iters  = {POP*(BUDGET//POP):,} evals")
    print(f"  SA      : 1   × {BUDGET} iters  = {BUDGET:,} evals")
    print(f"  Tabu    : ~{MOVES} × {BUDGET//MOVES} iters  ≈ {MOVES*(BUDGET//MOVES):,} evals")
    print(f"  ACO     : {ANTS}  × {BUDGET//ANTS} iters  = {ANTS*(BUDGET//ANTS):,} evals\n")

    algo_configs = {
        "GA":       dict(fn=ga_tsp,   kw=dict(pop_size=POP,  max_iter=BUDGET//POP,       seed=SEED, log_interval=100)),
        "SA":       dict(fn=sa_tsp,   kw=dict(max_iter=BUDGET,                            seed=SEED, log_interval=15_000)),
        "Tabu":     dict(fn=tabu_tsp, kw=dict(max_iter=BUDGET//MOVES, max_moves=MOVES,    seed=SEED, log_interval=200)),
        "ACO-MMAS": dict(fn=aco_tsp,  kw=dict(n_ants=ANTS,  max_iter=BUDGET//ANTS,       seed=SEED, log_interval=500)),
    }

    # ── Single run per algorithm — reuse results for both table and plots ──
    run_results: dict[str, dict] = {}

    for name, cfg in algo_configs.items():
        print(f"\n[{name}]")
        ctsp    = CountedTSP(tsp, BUDGET)
        t0      = time.perf_counter()
        tour, L = cfg['fn'](ctsp, **cfg['kw'])
        elapsed = time.perf_counter() - t0
        run_results[name] = dict(
            tour      = tour,
            length    = L,
            nfev      = ctsp.nfev,
            elapsed   = elapsed,
            throughput= ctsp.nfev / max(elapsed, 1e-9),
            trace     = ctsp.trace,
            # Wrap in lists so demo_utils helpers work
            finals    = [L],
            elapsed_l = [elapsed],
            histories = [],
        )
        run_results[name]['throughput_l'] = [ctsp.nfev / max(elapsed, 1e-9)]
        print(f"  → best={L:.2f}  nfev={ctsp.nfev:,}  t={elapsed:.2f}s"
              f"  ({ctsp.nfev/max(elapsed,1e-9)/1e3:.1f}k ev/s)")

    # ── Console summary table ──────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  {'Algorithm':<12} {'Tour Len':>10} {'nfev':>8} {'Time (s)':>10} {'kev/s':>8}")
    print(f"  {'─'*58}")
    for name, r in sorted(run_results.items(), key=lambda kv: kv[1]['length']):
        print(f"  {name:<12} {r['length']:>10.2f} {r['nfev']:>8,}"
              f" {r['elapsed']:>10.3f} {r['throughput']/1e3:>8.1f}")
    print(f"{'═'*62}")

    # ── Convergence vs nfev ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (name, r) in enumerate(run_results.items()):
        trace = r['trace']
        xs = [0] + [t[0] for t in trace] + [BUDGET]
        ys = [nn_L] + [t[1] for t in trace] + [trace[-1][1] if trace else nn_L]
        ax.step(xs, ys, where="post",
                label=f"{name}  (best={r['length']:.1f}, {r['elapsed']:.1f}s)",
                color=PALETTE[idx % len(PALETTE)], linewidth=2)
    ax.axhline(nn_L, color="gray", linestyle=":", linewidth=1.2,
               label=f"NN heuristic ({nn_L:.1f})")
    ax.set_xlabel("Function Evaluations (nfev)", fontsize=12)
    ax.set_ylabel("Best Tour Length", fontsize=12)
    ax.set_title(f"TSP Convergence — Equal Budget ({BUDGET:,} evals, {N_CITIES} cities)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, BUDGET)
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "convergence_vs_nfev.png"), dpi=150)
    plt.close(fig)

    # ── Timing bars ────────────────────────────────────────────────────────
    names     = list(run_results.keys())
    times     = [run_results[n]['elapsed']       for n in names]
    tps       = [run_results[n]['throughput']/1e3 for n in names]
    order     = np.argsort(times)
    names_s   = [names[i]  for i in order]
    times_s   = [times[i]  for i in order]
    tps_s     = [tps[i]    for i in order]
    colors_s  = [PALETTE[i % len(PALETTE)] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    b1 = ax1.barh(names_s, times_s, color=colors_s, edgecolor="white", alpha=0.85)
    ax1.bar_label(b1, fmt="%.2fs", padding=3, fontsize=9)
    ax1.set_xlabel("Wall-clock time (s)"); ax1.set_title("Runtime", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    b2 = ax2.barh(names_s, tps_s, color=colors_s, edgecolor="white", alpha=0.85)
    ax2.bar_label(b2, fmt="%.1f k/s", padding=3, fontsize=9)
    ax2.set_xlabel("Throughput (k evaluations / s)")
    ax2.set_title("Throughput  (higher = faster per eval)", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    fig.suptitle("TSP — Timing Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "timing.png"), dpi=150)
    plt.close(fig)

    # ── Quality vs time scatter ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, r) in enumerate(run_results.items()):
        ax.scatter(r['elapsed'], r['length'], s=140,
                   color=PALETTE[idx % len(PALETTE)], zorder=5)
        ax.annotate(name, (r['elapsed'], r['length']),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Best Tour Length", fontsize=12)
    ax.set_title("TSP — Tour Quality vs Wall-Clock Time\n"
                 "(bottom-left = fast AND accurate)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(OUTPUT / "quality_vs_time.png"), dpi=150)
    plt.close(fig)

    # ── Tour images — reuse tours from the single run above ───────────────
    for name, r in run_results.items():
        fig = plot_tsp_tour(
            tsp.cities, r['tour'], r['length'],
            title=f"{name}  (nfev={r['nfev']:,}, t={r['elapsed']:.1f}s)",
            save_path=str(OUTPUT / f"tour_{name}.png"))
        plt.close(fig)

    print(f"\nDemo 2 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
