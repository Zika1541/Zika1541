"""
Full comparison runner: extracts all needed functions from the notebook
and runs the 7-way benchmark.
"""

import json, numpy as np, time

# ── Load and exec notebook cells ──────────────────────────────────────────────
NB = "/home/user/Zika1541/3_최종_코드_정리.ipynb"

with open(NB) as f:
    nb = json.load(f)

ns = {}  # shared namespace

# Execute the key code cells in order
CELL_ORDER = [4, 7, 9, 20]   # benchmarks, WO, WO2, Schwefel/Levy
for idx in CELL_ORDER:
    src = "".join(nb["cells"][idx]["source"])
    try:
        exec(src, ns)
    except Exception as e:
        print(f"  [warn] cell {idx}: {e}")

# Execute the 4 new algorithm cells (added at the end: 43, 44, 45, 46)
for idx in [43, 44, 45, 46]:
    src = "".join(nb["cells"][idx]["source"])
    try:
        exec(src, ns)
    except Exception as e:
        print(f"  [warn] cell {idx}: {e}")

# Pull names into local scope
rastrigin    = ns["rastrigin"];    rastrigin_grad = ns["rastrigin_grad"]
schwefel     = ns["schwefel"];     schwefel_grad  = ns["schwefel_grad"]
levy         = ns["levy"];         levy_grad      = ns["levy_grad"]
wo2          = ns["wo2"]
pso          = ns["pso"]
run_ce       = ns["run_ce"]
run_rime     = ns["run_rime"]
run_lshade   = ns["run_lshade"]

from scipy.optimize import differential_evolution
import cma


def run_de_bench(func, lower, upper, D=10, N=80, iterations=2000, seed=0):
    def f1d(x): return func(x[None, :])[0]
    result = differential_evolution(
        f1d, [(lower, upper)] * D,
        maxiter=max(1, iterations // N), popsize=15,
        seed=seed, tol=1e-12)
    return result.fun


def run_cmaes_bench(func, lower, upper, D=10, N=80, iterations=2000, seed=0):
    def f1d(x): return func(np.array(x)[None, :])[0]
    np.random.seed(seed)
    x0 = np.random.uniform(lower, upper, D)
    opts = cma.CMAOptions()
    opts["bounds"]    = [lower, upper]
    opts["maxfevals"] = N * iterations
    opts["verbose"]   = -9
    opts["seed"]      = seed
    es = cma.CMAEvolutionStrategy(x0, (upper - lower) / 4, opts)
    es.optimize(f1d)
    return es.result.fbest


ALGO_LIST = [
    ("WO2",     lambda f, g, lo, hi, D, N, it, s: wo2(f, g, lo, hi, D=D, N=N, iterations=it, seed=s)),
    ("PSO",     lambda f, g, lo, hi, D, N, it, s: pso(f, lo, hi, D=D, N=N, iterations=it, seed=s)),
    ("CE",      lambda f, g, lo, hi, D, N, it, s: run_ce(f, lo, hi, D=D, N=N, iterations=it, seed=s)),
    ("RIME",    lambda f, g, lo, hi, D, N, it, s: run_rime(f, lo, hi, D=D, N=N, iterations=it, seed=s)),
    ("L-SHADE", lambda f, g, lo, hi, D, N, it, s: run_lshade(f, lo, hi, D=D, N=N, iterations=it, seed=s)),
    ("DE",      lambda f, g, lo, hi, D, N, it, s: run_de_bench(f, lo, hi, D=D, N=N, iterations=it, seed=s)),
    ("CMA-ES",  lambda f, g, lo, hi, D, N, it, s: run_cmaes_bench(f, lo, hi, D=D, N=N, iterations=it, seed=s)),
]

BENCH = {
    "Rastrigin": (rastrigin, rastrigin_grad, -5.12,  5.12),
    "Schwefel":  (schwefel,  schwefel_grad, -500.0, 500.0),
    "Levy":      (levy,      levy_grad,     -10.0,  10.0),
}

N_RUNS = 30
DIMS   = [10, 30, 50]

all_results = {}

for fname, (func, gfunc, lo, hi) in BENCH.items():
    print("\n" + "=" * 90)
    print(f"  [{fname}]  ({N_RUNS} runs per config)")
    print("=" * 90)
    print(f"  {'D':>4} | {'Algorithm':>10} | {'mean':>13} {'std':>11} {'best':>11} | {'time/run':>9}")
    print(f"  {'-'*80}")

    for D in DIMS:
        N     = max(30, D * 4)
        iters = max(500, D * 100)
        D_res = {}

        for aname, afunc in ALGO_LIST:
            runs, times = [], []
            for run in range(N_RUNS):
                t0 = time.time()
                try:
                    v = afunc(func, gfunc, lo, hi, D, N, iters, run)
                except Exception as e:
                    v = float("nan")
                runs.append(v)
                times.append(time.time() - t0)

            arr = np.array(runs)
            D_res[aname] = arr
            print(f"  {D:>4} | {aname:>10} | {np.nanmean(arr):>13.4f} {np.nanstd(arr):>11.4f} "
                  f"{np.nanmin(arr):>11.4f} | {np.mean(times):>9.3f}s")

        print(f"  {'-'*80}")
        all_results[(fname, D)] = D_res

# ── Final ranking summary ──────────────────────────────────────────────────────
print()
print("=" * 90)
print("  FINAL RANKING  (낮을수록 좋음 — mean 기준, 함수×차원 평균 순위)")
print("=" * 90)
rank_accum = {a: [] for a, _ in ALGO_LIST}

for (fname, D), D_res in all_results.items():
    means = {a: np.nanmean(D_res[a]) for a in D_res}
    sorted_algos = sorted(means, key=means.get)
    for rank, a in enumerate(sorted_algos, 1):
        rank_accum[a].append(rank)

n_configs = len(all_results)
print(f"  {'Algorithm':>10} | {'Avg Rank':>9} | {'#1st':>6} | {'#2nd':>6}  (out of {n_configs} configs)")
print(f"  {'-'*50}")
summary = sorted(rank_accum.items(), key=lambda kv: np.mean(kv[1]))
for i, (a, ranks) in enumerate(summary, 1):
    wins   = sum(1 for r in ranks if r == 1)
    second = sum(1 for r in ranks if r == 2)
    print(f"  {a:>10} | {np.mean(ranks):>9.2f} | {wins:>6} | {second:>6}")

print()
print("알고리즘 설명:")
print("  CE      : Cross-Entropy Method (Rubinstein & Kroese 2004)")
print("  RIME    : Rime-Ice Optimization Algorithm (Su et al., Expert Syst. Appl. 2023)")
print("  L-SHADE : Linear pop-size reduction SHADE (Tanabe & Fukunaga 2014; 2024 variants)")
print("  DE      : Differential Evolution (scipy built-in, popsize=15)")
print("  CMA-ES  : Covariance Matrix Adaptation ES (PyCMA)")
print("  WO2     : Weather Optimization v2 (본 연구, 2026)")
print("  PSO     : Particle Swarm Optimization (baseline)")
