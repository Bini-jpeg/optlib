"""
Demo 2 — Travelling Salesman Problem (equal evaluation budget).

Budget : BUDGET = 60,000 tour-length evaluations per algorithm.
         One evaluation = one call to tsp.tour_length().

Comparison is plotted on the nfev x-axis so all algorithms are judged
at the same information cost regardless of their population/neighbourhood size.

Outputs (in outputs/demo02/):
  convergence_vs_nfev.png   — best tour length vs evaluations
  timing.png                — runtime + throughput bars
  quality_vs_time.png       — accuracy vs speed scatter
  tour_<algo>.png           — best route found
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
BUDGET   = 60_000    # tour-length evaluations per algorithm


# ── Budget-counting TSP wrapper ─────────────────────────────────────────────

class CountedTSP:
    """Wraps TSP and counts every tour_length() call; stops at budget."""

    class BudgetExhausted(Exception):
        pass

    def __init__(self, tsp: TSP, budget: int):
        self._tsp  = tsp
        self.budget = budget
        self.nfev   = 0
        self.best   = math.inf
        self.trace: list[tuple[int, float]] = []   # (nfev, best_length)

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
        try: self.tour_length(tour)
        except CountedTSP.BudgetExhausted: pass
        return tour

    def two_opt_improve(self, tour: np.ndarray, max_passes: int = 100) -> np.ndarray:
        t = tour.copy(); n = self._tsp.n; improved = True; passes = 0
        while improved and passes < max_passes:
            improved = False; passes += 1
            for i in range(1, n-1):
                for j in range(i+1, n):
                    if j-i == 1: continue
                    db = self._tsp.dist[t[i-1],t[i]]   + self._tsp.dist[t[j-1],t[j%n]]
                    da = self._tsp.dist[t[i-1],t[j-1]] + self._tsp.dist[t[i],  t[j%n]]
                    if da < db - 1e-10:
                        t[i:j] = t[i:j][::-1]; improved = True
            if improved:
                try: self.tour_length(t)
                except CountedTSP.BudgetExhausted: return t
        return t


# ── Algorithm implementations ────────────────────────────────────────────────

def ga_tsp(ctsp, pop_size=100, max_iter=600, p_mut=0.02, elite=5,
           seed=None, log_interval=None):
    rng = np.random.default_rng(seed); n = ctsp.n
    pop  = np.array([rng.permutation(n) for _ in range(pop_size)])
    try: fits = np.array([ctsp.tour_length(t) for t in pop])
    except CountedTSP.BudgetExhausted:
        fits = np.array([ctsp._tsp.tour_length(t) for t in pop])
    best_tour = pop[np.argmin(fits)].copy(); best_L = fits.min()

    def _ox(p1, p2):
        a,b = sorted(rng.integers(0,n,2)); child = [-1]*n
        child[a:b+1] = list(p1[a:b+1]); used = set(child[a:b+1]); ptr = b+1
        for g in np.roll(p2, -(b+1)):
            if g not in used: child[ptr%n]=g; used.add(g); ptr+=1
        return np.array(child, dtype=int)

    def _mut(t):
        if rng.random() < p_mut*n:
            i,j = rng.integers(0,n,2); t[i],t[j] = t[j],t[i]
        return t

    for it in range(1, max_iter+1):
        order = np.argsort(fits); elites=pop[order[:elite]].copy(); ef=fits[order[:elite]].copy()
        offs=[]
        while len(offs) < pop_size-elite:
            ix = rng.integers(0,pop_size,6)
            p1=ix[np.argmin(fits[ix[:3]])]; p2=ix[3+np.argmin(fits[ix[3:]])]
            offs.append(_mut(_ox(pop[p1],pop[p2])))
        offs = np.array(offs)
        try: of = np.array([ctsp.tour_length(t) for t in offs])
        except CountedTSP.BudgetExhausted: break
        pop=np.vstack([elites,offs]); fits=np.concatenate([ef,of])
        idx=np.argmin(fits)
        if fits[idx]<best_L: best_L=fits[idx]; best_tour=pop[idx].copy()
        if log_interval and it%log_interval==0:
            print(f"  [GA-TSP]   iter={it:>4d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}")
    return best_tour, best_L


def sa_tsp(ctsp, max_iter=60_000, T0=None, alpha=0.9998,
           seed=None, log_interval=None):
    rng = np.random.default_rng(seed); n = ctsp.n
    tour = ctsp.nearest_neighbour(); L = ctsp._tsp.tour_length(tour)
    T = T0 or L*0.15; best_tour=tour.copy(); best_L=L
    for it in range(1, max_iter+1):
        i,j = sorted(rng.integers(0,n,2))
        if i==j: continue
        nt=tour.copy(); nt[i:j+1]=nt[i:j+1][::-1]
        try: nL=ctsp.tour_length(nt)
        except CountedTSP.BudgetExhausted: break
        delta=nL-L
        if delta<0 or rng.random()<math.exp(-delta/max(T,1e-10)):
            tour=nt; L=nL
        if L<best_L: best_L=L; best_tour=tour.copy()
        T*=alpha
        if log_interval and it%log_interval==0:
            print(f"  [SA-TSP]   iter={it:>5d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}  T={T:.3f}")
    return best_tour, best_L


def tabu_tsp(ctsp, max_iter=1000, tabu_tenure=15, max_moves=60,
             seed=None, log_interval=None):
    rng = np.random.default_rng(seed); n = ctsp.n
    tour=ctsp.nearest_neighbour(); tour=ctsp.two_opt_improve(tour,max_passes=3)
    L=ctsp._tsp.tour_length(tour); best_tour=tour.copy(); best_L=L
    tabu: dict = {}; no_impr=0
    for it in range(1, max_iter+1):
        pairs = [ctsp._tsp.dist.shape[0]]   # dummy; generate below
        all_p = [(i,j) for i in range(1,n-1) for j in range(i+1,n)]
        pairs = [all_p[k] for k in rng.permutation(len(all_p))[:max_moves]]
        bc=None; bcL=math.inf; bm=None
        for (i,j) in pairs:
            cand=tour.copy(); cand[i:j+1]=cand[i:j+1][::-1]
            try: cL=ctsp.tour_length(cand)
            except CountedTSP.BudgetExhausted:
                return best_tour, best_L
            if (tabu.get((i,j),0)<it or cL<best_L) and cL<bcL:
                bc=cand; bcL=cL; bm=(i,j)
        if bc is None: break
        tour=bc; L=bcL
        if bm: tabu[bm]=it+tabu_tenure
        if L<best_L: best_L=L; best_tour=tour.copy(); no_impr=0
        else: no_impr+=1
        if no_impr>150: tour=best_tour.copy(); L=best_L; no_impr=0
        if log_interval and it%log_interval==0:
            print(f"  [Tabu]     iter={it:>4d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}")
    return best_tour, best_L


def aco_tsp(ctsp, n_ants=25, max_iter=2400, alpha=1.0, beta=5.0, rho=0.1,
            seed=None, log_interval=None):
    rng = np.random.default_rng(seed); n = ctsp.n
    dist=ctsp.dist; eta=1.0/(dist+np.eye(n)*1e-10)
    nn_L=ctsp._tsp.tour_length(ctsp._tsp.nearest_neighbour())
    tau=np.full((n,n), 1.0/(n*nn_L))
    tmax=1.0/(rho*nn_L); tmin=tmax/(2.0*n)
    best_tour=ctsp._tsp.two_opt_improve(ctsp._tsp.nearest_neighbour())
    best_L=ctsp._tsp.tour_length(best_tour)

    def build():
        vis=np.zeros(n,dtype=bool); tour=np.empty(n,dtype=int)
        s=int(rng.integers(0,n)); tour[0]=s; vis[s]=True
        for step in range(1,n):
            cur=tour[step-1]
            att=(tau[cur]*~vis)**alpha*(eta[cur]*~vis)**beta
            tot=att.sum()
            nxt=int(rng.choice(n,p=att/tot)) if tot>1e-300 \
                else int(rng.choice(np.where(~vis)[0]))
            tour[step]=nxt; vis[nxt]=True
        return tour

    for it in range(1, max_iter+1):
        tours=[build() for _ in range(n_ants)]
        try: lengths=np.array([ctsp.tour_length(t) for t in tours])
        except CountedTSP.BudgetExhausted: break
        tau*=(1-rho)
        kb=int(np.argmin(lengths))
        dep=tours[kb] if lengths[kb]<best_L else best_tour
        dL=min(lengths[kb],best_L)
        for i in range(n):
            a,b=dep[i],dep[(i+1)%n]; tau[a,b]+=1/dL; tau[b,a]+=1/dL
        tau=np.clip(tau,tmin,tmax)
        if lengths[kb]<best_L:
            best_L=lengths[kb]; best_tour=tours[kb].copy()
            best_tour=ctsp.two_opt_improve(best_tour,max_passes=3)
            best_L=ctsp._tsp.tour_length(best_tour)
        if log_interval and it%log_interval==0:
            print(f"  [ACO-MMAS] iter={it:>4d}  nfev={ctsp.nfev:>6,}  best={best_L:.2f}")
    return best_tour, best_L


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tsp   = TSP(n_cities=N_CITIES, seed=SEED)
    nn_L  = tsp.tour_length(tsp.nearest_neighbour())
    print(f"TSP: {N_CITIES} cities  |  NN heuristic = {nn_L:.2f}")
    print(f"Fixed evaluation budget : {BUDGET:,} tour-length calls\n")

    # Budget arithmetic
    POP, ANTS, MOVES = 100, 25, 60
    print(f"  GA      : pop={POP} × {BUDGET//POP} iters  = {POP*(BUDGET//POP):,} evals")
    print(f"  SA      : 1   × {BUDGET} iters  = {BUDGET:,} evals")
    print(f"  Tabu    : ~{MOVES} × {BUDGET//MOVES} iters  ≈ {MOVES*(BUDGET//MOVES):,} evals")
    print(f"  ACO     : {ANTS}  × {BUDGET//ANTS} iters  = {ANTS*(BUDGET//ANTS):,} evals\n")

    algo_configs = {
        "GA":        dict(fn=ga_tsp,   kw=dict(pop_size=POP,  max_iter=BUDGET//POP,  seed=SEED, log_interval=100)),
        "SA":        dict(fn=sa_tsp,   kw=dict(max_iter=BUDGET,                       seed=SEED, log_interval=15_000)),
        "Tabu":      dict(fn=tabu_tsp, kw=dict(max_iter=BUDGET//MOVES, max_moves=MOVES, seed=SEED, log_interval=200)),
        "ACO-MMAS":  dict(fn=aco_tsp,  kw=dict(n_ants=ANTS,  max_iter=BUDGET//ANTS,  seed=SEED, log_interval=500)),
    }

    # Pseudo-results dict for demo_utils plotting
    run_results: dict[str, dict] = {
        n: dict(finals=[], elapsed=[], nfev=[], histories=[], throughput=[], traces=[])
        for n in algo_configs
    }

    for name, cfg in algo_configs.items():
        print(f"\n[{name}]")
        ctsp = CountedTSP(tsp, BUDGET)
        t0   = time.perf_counter()
        tour, L = cfg['fn'](ctsp, **cfg['kw'])
        elapsed  = time.perf_counter() - t0
        run_results[name]['finals'].append(L)
        run_results[name]['elapsed'].append(elapsed)
        run_results[name]['nfev'].append(ctsp.nfev)
        run_results[name]['throughput'].append(ctsp.nfev / max(elapsed, 1e-9))
        run_results[name]['traces'].append(ctsp.trace)
        print(f"  → best={L:.2f}  nfev={ctsp.nfev:,}  t={elapsed:.2f}s"
              f"  ({ctsp.nfev/max(elapsed,1e-9)/1e3:.1f}k ev/s)")

    # ── Console table ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {'Algorithm':<12} {'Tour Len':>10} {'nfev':>8} {'Time (s)':>10} {'kev/s':>8}")
    print(f"  {'─'*56}")
    for name, r in sorted(run_results.items(), key=lambda kv: kv[1]['finals'][0]):
        print(f"  {name:<12} {r['finals'][0]:>10.2f} {r['nfev'][0]:>8,}"
              f" {r['elapsed'][0]:>10.3f} {r['throughput'][0]/1e3:>8.1f}")
    print(f"{'═'*60}")

    # ── Convergence vs nfev ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (name, r) in enumerate(run_results.items()):
        trace = r['traces'][0]
        xs = [0] + [t[0] for t in trace] + [BUDGET]
        ys = [nn_L] + [t[1] for t in trace] + [trace[-1][1] if trace else nn_L]
        ax.step(xs, ys, where="post", label=f"{name} ({r['finals'][0]:.1f})",
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
    fig.savefig(str(OUTPUT / "convergence_vs_nfev.png"), dpi=150); plt.close(fig)

    # ── Timing ────────────────────────────────────────────────────────────
    fig = plot_timing(run_results, title="TSP — Algorithm Timing Comparison",
                      save_path=str(OUTPUT / "timing.png")); plt.close(fig)

    fig = plot_quality_vs_time(run_results,
                               title="TSP — Tour Quality vs Wall-Clock Time",
                               log_y=False,
                               save_path=str(OUTPUT / "quality_vs_time.png")); plt.close(fig)

    # ── Tour plots ─────────────────────────────────────────────────────────
    # Re-run each with the same budget and save the best tour image
    for name, cfg in algo_configs.items():
        ctsp2 = CountedTSP(tsp, BUDGET)
        t2, L2 = cfg['fn'](ctsp2, **cfg['kw'])
        fig = plot_tsp_tour(tsp.cities, t2, L2,
                            title=f"{name}  (nfev={ctsp2.nfev:,})",
                            save_path=str(OUTPUT / f"tour_{name}.png"))
        plt.close(fig)

    print(f"\nDemo 2 complete — outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
