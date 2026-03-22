"""
Add CE, RIME, L-SHADE algorithms + comprehensive comparison to notebook.
Then run the comparison and print results.
"""

import json
import numpy as np
import subprocess, sys, os

NOTEBOOK = "/home/user/Zika1541/3_최종_코드_정리.ipynb"

# ── Algorithm implementation code ─────────────────────────────────────────────

CE_CODE = '''# ============================================================
# Cross-Entropy (CE) Method for Continuous Optimization
# Rubinstein & Kroese (2004); widely used baseline in EAs
# ============================================================

def run_ce(func, lower, upper, D=10, N=80, iterations=2000, seed=0,
           elite_frac=0.2, noise_decay=0.998):
    """
    Cross-Entropy Method:
      1. Sample from N(μ, σ²I)
      2. Evaluate & select elite fraction
      3. Update μ, σ from elites
      4. Noise floor prevents premature convergence
    """
    rng = np.random.default_rng(seed)
    n_elite = max(2, int(N * elite_frac))

    mu    = rng.uniform(lower, upper, D)
    sigma = np.full(D, (upper - lower) / 4.0)

    best_loss = float("inf")

    for t in range(iterations):
        X = mu + sigma * rng.standard_normal((N, D))
        X = np.clip(X, lower, upper)

        losses    = func(X)
        elite_idx = np.argpartition(losses, n_elite)[:n_elite]
        elites    = X[elite_idx]

        mu    = elites.mean(axis=0)
        sigma = elites.std(axis=0)

        # Noise floor: prevents sigma collapsing to 0
        noise_floor = (upper - lower) * 1e-4 * (noise_decay ** t) + 1e-8
        sigma = np.maximum(sigma, noise_floor)

        cur_best = losses[elite_idx].min()
        if cur_best < best_loss:
            best_loss = cur_best

    return best_loss
'''

RIME_CODE = '''# ============================================================
# RIME: Rime-Ice Optimization Algorithm (2023)
# Su et al., Expert Systems with Applications, 2023
# Inspired by rime-ice formation physics
# ============================================================

def run_rime(func, lower, upper, D=10, N=80, iterations=2000, seed=0, W=5.0):
    """
    RIME (2023):
      Soft-rime: attraction toward global best (exploration)
      Hard-rime: dimension-wise puncture with X_best (exploitation)
      Probability of hard-rime increases linearly over iterations.
      Greedy acceptance ensures monotonic improvement per particle.

    W: RIME-force coefficient (controls exploration radius).
    """
    rng  = np.random.default_rng(seed)
    span = upper - lower

    X       = rng.uniform(lower, upper, (N, D))
    fitness = func(X)

    best_idx = np.argmin(fitness)
    X_best   = X[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for t in range(1, iterations + 1):
        norm_t      = t / iterations                       # ∈ (0, 1]
        rime_factor = W * span / (2.0 * np.sqrt(t * iterations))

        # --- Soft-rime: move toward global best ---
        r1    = rng.random((N, D))
        X_new = X + rime_factor * (X_best[None, :] - X) * r1

        # --- Hard-rime puncture: copy X_best dims with prob = norm_t ---
        r2   = rng.random((N, D))
        mask = r2 < norm_t
        X_new = np.where(mask, X_best[None, :], X_new)

        X_new   = np.clip(X_new, lower, upper)
        fit_new = func(X_new)

        # Greedy selection
        improved = fit_new < fitness
        X        = np.where(improved[:, None], X_new, X)
        fitness  = np.where(improved, fit_new, fitness)

        cur_best = int(np.argmin(fitness))
        if fitness[cur_best] < best_fit:
            best_fit = float(fitness[cur_best])
            X_best   = X[cur_best].copy()

    return best_fit
'''

LSHADE_CODE = '''# ============================================================
# L-SHADE: Linear population-size reduction + SHADE
# Tanabe & Fukunaga (GECCO 2014); with recent (2024) tuning
#   • Success-History based Adaptive DE (SHADE)
#   • current-to-pBest/1/bin mutation
#   • Archive of replaced solutions for diversity
#   • Linear population shrinkage N_init → N_min
# ============================================================

def run_lshade(func, lower, upper, D=10, N=80, iterations=2000, seed=0, H=6):
    """
    L-SHADE (vectorized):
      F  ~ Cauchy(M_F, 0.1)   clipped to (0, 1]
      CR ~ Normal(M_CR, 0.1)  clipped to [0, 1]
      Mutation: x_pbest + F*(x_r1 - x_r2_or_archive)
      Budget: N * iterations function evaluations
    """
    rng    = np.random.default_rng(seed)
    N_init = N
    N_min  = max(4, D // 5)
    budget = N * iterations

    # Success-history memory
    M_F  = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k    = 0

    pop     = rng.uniform(lower, upper, (N_init, D))
    fitness = func(pop)
    best_fit = float(np.min(fitness))
    archive  = np.empty((0, D))
    evals    = N_init

    while evals < budget:
        N_cur = len(pop)

        # --- Linear population reduction ---
        N_new = max(N_min, int(N_min + (N_init - N_min) * (budget - evals) / budget))
        if N_new < N_cur:
            keep  = np.argpartition(fitness, N_new)[:N_new]
            pop   = pop[keep];  fitness = fitness[keep];  N_cur = N_new

        # --- Sample F & CR from history ---
        ri = rng.integers(0, H, N_cur)
        F  = np.clip(rng.standard_cauchy(N_cur) * 0.1 + M_F[ri],  0.01, 1.0)
        CR = np.clip(rng.standard_normal(N_cur) * 0.1 + M_CR[ri], 0.0,  1.0)

        # --- pBest selection ---
        p_n     = max(2, int(0.11 * N_cur))
        p_idx   = np.argpartition(fitness, p_n)[:p_n]
        xpbest  = pop[p_idx[rng.integers(p_n, size=N_cur)]]        # (N_cur, D)

        # --- r1: different from i (vectorized) ---
        shifts  = rng.integers(1, N_cur, N_cur)
        r1_idx  = (np.arange(N_cur) + shifts) % N_cur
        x_r1    = pop[r1_idx]

        # --- r2: from pop ∪ archive ---
        combined = np.vstack([pop, archive]) if len(archive) > 0 else pop
        r2_idx   = rng.integers(len(combined), size=N_cur)
        x_r2     = combined[r2_idx]

        # --- Mutation & crossover ---
        mutants = np.clip(pop + F[:, None] * (xpbest - pop) + F[:, None] * (x_r1 - x_r2),
                          lower, upper)
        j_rand  = rng.integers(D, size=N_cur)
        mask    = rng.random((N_cur, D)) < CR[:, None]
        mask[np.arange(N_cur), j_rand] = True
        trials  = np.where(mask, mutants, pop)

        # --- Evaluation ---
        f_trials = func(trials)
        evals   += N_cur
        if evals > budget:
            break

        # --- Selection ---
        improved = f_trials <= fitness

        # Archive replaced individuals
        if improved.any():
            archive = np.vstack([archive, pop[improved]]) if len(archive) > 0 else pop[improved].copy()
            if len(archive) > N_init:
                keep    = rng.choice(len(archive), N_init, replace=False)
                archive = archive[keep]

        S_F  = F[improved];  S_CR = CR[improved]
        S_d  = np.abs(f_trials[improved] - fitness[improved])

        pop     = np.where(improved[:, None], trials, pop)
        fitness = np.where(improved, f_trials, fitness)

        cur_min = float(np.min(fitness))
        if cur_min < best_fit:
            best_fit = cur_min

        # --- Update memory ---
        if len(S_F) > 0:
            w    = S_d / (S_d.sum() + 1e-12)
            M_F[k]  = np.dot(w, S_F**2) / (np.dot(w, S_F) + 1e-12)
            M_CR[k] = np.dot(w, S_CR)
            k = (k + 1) % H

    return best_fit
'''

COMPARE_CODE = '''# ============================================================
# 종합 비교: WO2 vs PSO vs CE vs RIME vs L-SHADE vs DE vs CMA-ES
# 최신 알고리즘 포함 7-way 비교 (Rastrigin, Schwefel, Levy)
# ============================================================

import time
from scipy.optimize import differential_evolution
import cma

def run_de_bench(func, lower, upper, D=10, N=80, iterations=2000, seed=0):
    def f1d(x): return func(x[None, :])[0]
    result = differential_evolution(
        f1d, [(lower, upper)] * D,
        maxiter=iterations // N, popsize=15,
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
    ("WO2",    lambda f,g,lo,hi,D,N,it,s: wo2(f,g,lo,hi,D=D,N=N,iterations=it,seed=s)),
    ("PSO",    lambda f,g,lo,hi,D,N,it,s: pso(f,lo,hi,D=D,N=N,iterations=it,seed=s)),
    ("CE",     lambda f,g,lo,hi,D,N,it,s: run_ce(f,lo,hi,D=D,N=N,iterations=it,seed=s)),
    ("RIME",   lambda f,g,lo,hi,D,N,it,s: run_rime(f,lo,hi,D=D,N=N,iterations=it,seed=s)),
    ("L-SHADE",lambda f,g,lo,hi,D,N,it,s: run_lshade(f,lo,hi,D=D,N=N,iterations=it,seed=s)),
    ("DE",     lambda f,g,lo,hi,D,N,it,s: run_de_bench(f,lo,hi,D=D,N=N,iterations=it,seed=s)),
    ("CMA-ES", lambda f,g,lo,hi,D,N,it,s: run_cmaes_bench(f,lo,hi,D=D,N=N,iterations=it,seed=s)),
]

def stage_comprehensive(n_runs=30, dims=None):
    """7-way 종합 비교: WO2, PSO, CE, RIME, L-SHADE, DE, CMA-ES"""
    if dims is None:
        dims = [10, 30, 50]

    BENCH = {
        "Rastrigin": (rastrigin, rastrigin_grad, -5.12,  5.12),
        "Schwefel":  (schwefel,  schwefel_grad, -500.0, 500.0),
        "Levy":      (levy,      levy_grad,     -10.0,  10.0),
    }

    all_results = {}

    for fname, (func, gfunc, lo, hi) in BENCH.items():
        print("=" * 90)
        print(f"  [{fname}]  ({n_runs} runs per config)")
        print("=" * 90)
        hdr = f"  {'D':>4} | {'Algorithm':>10} | {'mean':>13} {'std':>11} {'best':>11} | {'time/run':>9}"
        print(hdr)
        print(f"  {'-'*78}")

        for D in dims:
            N     = max(30, D * 4)
            iters = max(500, D * 100)
            D_res = {}

            for aname, afunc in ALGO_LIST:
                runs, times = [], []
                for run in range(n_runs):
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

            print(f"  {'-'*78}")
            all_results[(fname, D)] = D_res

    # ── Final ranking summary ──────────────────────────────────────────────
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

    print(f"  {'Algorithm':>10} | {'Avg Rank':>9} | {'Win Count':>9} (out of {len(all_results)} configs)")
    print(f"  {'-'*45}")
    summary = sorted(rank_accum.items(), key=lambda kv: np.mean(kv[1]))
    for i, (a, ranks) in enumerate(summary, 1):
        wins = sum(1 for r in ranks if r == 1)
        print(f"  {a:>10} | {np.mean(ranks):>9.2f} | {wins:>9}")

    print()
    print("알고리즘 설명:")
    print("  CE      : Cross-Entropy Method (Rubinstein & Kroese 2004)")
    print("  RIME    : Rime-Ice Optimization Algorithm (Su et al., Expert Syst. Appl. 2023)")
    print("  L-SHADE : Linear pop-size reduction SHADE (Tanabe & Fukunaga 2014; 2024 improvements)")
    print("  DE      : Differential Evolution (scipy, popsize=15)")
    print("  CMA-ES  : Covariance Matrix Adaptation ES (PyCMA)")

    return all_results

# stage_comprehensive(n_runs=30, dims=[10, 30, 50])
'''

# ── Insert cells into notebook ─────────────────────────────────────────────────

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

with open(NOTEBOOK) as f:
    nb = json.load(f)

# Add new cells at the end (before the last empty cell)
new_cells = [
    make_code_cell(CE_CODE),
    make_code_cell(RIME_CODE),
    make_code_cell(LSHADE_CODE),
    make_code_cell(COMPARE_CODE),
]

# Remove trailing empty cells, append new ones, keep structure
nb["cells"] = nb["cells"] + new_cells

with open(NOTEBOOK, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cells added to notebook successfully.")
print(f"Total cells: {len(nb['cells'])}")
