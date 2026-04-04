import json

with open('3_최종_코드_정리.ipynb') as f:
    nb = json.load(f)

wo2_code = r'''# ============================================================
# WO2: Weather Optimization v2
# 파라미터 4개: lr, beta, alpha, sigma
# 복잡도: O(N·D) per iteration
# ============================================================

def wo2(
    func,
    grad_func,
    lower=-5.12,
    upper=5.12,
    D=10,
    N=80,
    iterations=2000,
    # --- 4 algorithm parameters ---
    lr=0.02,        # base learning rate (cosine schedule)
    beta=0.9,       # momentum / EMA factor
    alpha=0.1,      # global attraction strength (응축)
    sigma=0.15,     # explosion radius as fraction of bound span (폭발)
    # --- standard setup ---
    seed=0,
    return_history=False,
):
    """
    WO2: Simplified Weather Optimization v2

    응축 (Condensation): momentum gradient descent + attraction to global best
    폭발 (Explosion): EMA-based energy tracking, stagnated particles explode near best

    Only 4 parameters, O(N*D) per iteration, no sorting/clustering.
    """
    rng = np.random.default_rng(seed)
    span = upper - lower

    # ---- Initialization ----
    x = rng.uniform(lower, upper, (N, D))
    v = np.zeros((N, D))                    # momentum velocities

    loss = func(x)
    loss_prev = loss.copy()

    energy = np.ones(N)                     # EMA of |delta_loss|, init=1

    best_idx = np.argmin(loss)
    x_best = x[best_idx].copy()
    best_loss = float(loss[best_idx])
    best_loss_prev = best_loss

    history = []

    for t in range(iterations):

        # === PHASE 1: CONDENSATION (응축) ===

        # Cosine learning rate schedule
        lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * t / iterations))

        # Gradients + norm clipping
        g = grad_func(x)                                        # (N, D)
        g_norm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-8
        g_clipped = g / np.maximum(g_norm, 1.0)                # clip to 1

        # Momentum update (heavy ball)
        v = beta * v - lr_t * g_clipped                        # (N, D)

        # Global attraction: pull toward current best (응축 은유)
        attract = alpha * (x_best - x)                         # (N, D)

        # Update positions
        x = np.clip(x + v + attract, lower, upper)

        # === ENERGY TRACKING ===
        loss = func(x)

        delta = np.abs(loss - loss_prev)                       # improvement per particle
        energy = beta * energy + (1.0 - beta) * delta          # EMA
        loss_prev = loss.copy()

        # Update global best
        idx_t = np.argmin(loss)
        if loss[idx_t] < best_loss:
            best_loss = float(loss[idx_t])
            x_best = x[idx_t].copy()

        # === PHASE 2: EXPLOSION (폭발) ===
        e_mean = np.mean(energy)

        stagnated = energy < (1.0 - sigma) * e_mean            # O(N)

        # Global stagnation safety: force explode if best hasn't improved
        # Uses existing beta (window ≈ 1/(1-beta)) and sigma — zero extra parameters
        window = max(1, int(1.0 / (1.0 - beta)))
        if t > 0 and (t % window) == 0 and best_loss >= best_loss_prev:
            n_force = max(1, int(sigma * N))
            force_idx = rng.choice(N, size=n_force, replace=False)
            stagnated[force_idx] = True
        best_loss_prev = best_loss

        n_explode = int(stagnated.sum())
        if n_explode > 0:
            noise = rng.standard_normal((n_explode, D))
            x_new = np.clip(x_best + sigma * span * noise, lower, upper)
            x[stagnated] = x_new
            v[stagnated] = 0.0
            energy[stagnated] = e_mean                         # reset to mean (방지 재폭발)

        if return_history:
            history.append({
                "iter":      t,
                "best_loss": best_loss,
                "n_explode": n_explode,
                "e_mean":    float(e_mean),
                "lr_t":      float(lr_t),
            })

    if return_history:
        return best_loss, history
    return best_loss
'''

benchmark_code = r'''import time
from scipy.stats import wilcoxon

def stage_wo2_benchmark(n_runs=30, dims=None):
    """WO vs WO2 vs PSO benchmark — Rastrigin, Schwefel, Levy"""
    if dims is None:
        dims = [10, 30, 50, 100]

    print("=" * 80)
    print("WO2 BENCHMARK: WO vs WO2 vs PSO")
    print("=" * 80)

    for fname, (func, gfunc, lo, hi) in MULTIMODAL_BENCHMARKS.items():
        print(f"\n  [{fname}]")
        header = f"  {'D':>4} | {'Algorithm':>10} | {'mean':>12} {'std':>10} {'best':>10} | {'avg_time':>8}"
        print(header)
        print(f"  {'-'*70}")

        for D in dims:
            N     = max(30, D * 4)
            iters = max(500, D * 100)
            wo_res, wo2_res, pso_res = [], [], []
            wo_t, wo2_t, pso_t = [], [], []

            for run in range(n_runs):
                t0 = time.time()
                bw, _ = weather_optimize(func, gfunc, lo, hi,
                                         D=D, N=N, iterations=iters, seed=run,
                                         storm_cap_ratio=0.15)
                wo_t.append(time.time() - t0)
                wo_res.append(bw)

                t0 = time.time()
                bw2 = wo2(func, gfunc, lo, hi, D=D, N=N, iterations=iters, seed=run)
                wo2_t.append(time.time() - t0)
                wo2_res.append(bw2)

                t0 = time.time()
                bp = pso(func, lo, hi, D=D, N=N, iterations=iters, seed=run)
                pso_t.append(time.time() - t0)
                pso_res.append(bp)

            wa, w2a, pa = np.array(wo_res), np.array(wo2_res), np.array(pso_res)

            try:
                stat, pval = wilcoxon(wa, w2a, alternative='greater')
                sig = "p<0.05 ✅" if pval < 0.05 else f"p={pval:.2f}"
            except Exception:
                sig = "n/a"

            improve_pct = (wa.mean() - w2a.mean()) / (wa.mean() + 1e-10) * 100

            print(f"  {D:>4} | {'WO':>10} | {wa.mean():>12.4f} {wa.std():>10.4f} {wa.min():>10.4f} | {np.mean(wo_t):>7.2f}s")
            print(f"  {D:>4} | {'WO2':>10} | {w2a.mean():>12.4f} {w2a.std():>10.4f} {w2a.min():>10.4f} | {np.mean(wo2_t):>7.2f}s  delta={improve_pct:+.1f}% {sig}")
            print(f"  {D:>4} | {'PSO':>10} | {pa.mean():>12.4f} {pa.std():>10.4f} {pa.min():>10.4f} | {np.mean(pso_t):>7.2f}s")
            print()

# Full run (30 seeds, D=[10,30,50,100]):
# stage_wo2_benchmark(n_runs=30)

# Quick test:
stage_wo2_benchmark(n_runs=5, dims=[10, 30])
'''

ablation_code = r'''def stage_wo2_ablation(n_runs=30, D=30):
    """
    Ablation: 4 variants to isolate each WO2 component
      A: Pure GD  (no momentum, no attract, global-mean explosion)
      B: +Momentum
      C: +Attraction (toward x_best)
      D: WO2 Full (+best-centered explosion)
    """
    print("=" * 70)
    print(f"WO2 ABLATION STUDY  (Rastrigin, D={D}, {n_runs} runs)")
    print("=" * 70)

    func, gfunc, lo, hi = MULTIMODAL_BENCHMARKS["Rastrigin"]
    N = max(30, D * 4)
    iters = max(500, D * 100)

    def run_variant(use_momentum, use_attract, best_centered_explode):
        results = []
        for run in range(n_runs):
            rng = np.random.default_rng(run)
            span = hi - lo
            x = rng.uniform(lo, hi, (N, D))
            v = np.zeros((N, D))
            lr, beta, alpha, sigma = 0.02, 0.9, 0.1, 0.15

            loss = func(x)
            loss_prev = loss.copy()
            energy = np.ones(N)
            best_idx = np.argmin(loss)
            x_best = x[best_idx].copy()
            best_loss = float(loss[best_idx])

            for t in range(iters):
                lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * t / iters))
                g = gfunc(x)
                g_norm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-8
                g_clipped = g / np.maximum(g_norm, 1.0)

                if use_momentum:
                    v = beta * v - lr_t * g_clipped
                    step = v.copy()
                else:
                    step = -lr_t * g_clipped

                if use_attract:
                    step = step + alpha * (x_best - x)

                x = np.clip(x + step, lo, hi)
                loss = func(x)

                delta = np.abs(loss - loss_prev)
                energy = beta * energy + (1.0 - beta) * delta
                loss_prev = loss.copy()

                idx_t = np.argmin(loss)
                if loss[idx_t] < best_loss:
                    best_loss = float(loss[idx_t])
                    x_best = x[idx_t].copy()

                e_mean = np.mean(energy)
                stagnated = energy < (1.0 - sigma) * e_mean
                n_explode = int(stagnated.sum())

                if n_explode > 0:
                    if best_centered_explode:
                        noise = rng.standard_normal((n_explode, D))
                        x_new = np.clip(x_best + sigma * span * noise, lo, hi)
                    else:
                        g_center = np.mean(x, axis=0)
                        g_std = np.std(x, axis=0) + 1e-8
                        x_new = np.clip(g_center + rng.normal(0, g_std, (n_explode, D)), lo, hi)
                    x[stagnated] = x_new
                    v[stagnated] = 0.0
                    energy[stagnated] = e_mean

            results.append(float(np.min(func(x))))
        return np.array(results)

    variants = [
        ("A: Pure GD",     False, False, False),
        ("B: +Momentum",   True,  False, False),
        ("C: +Attraction", True,  True,  False),
        ("D: WO2 Full",    True,  True,  True),
    ]

    print(f"  {'Variant':<20} | {'mean':>12} {'std':>10} {'best':>10} | vs A")
    print(f"  {'-'*62}")

    baseline = None
    for label, mom, attr, bce in variants:
        res = run_variant(mom, attr, bce)
        if baseline is None:
            baseline = res.mean()
        improve = (baseline - res.mean()) / (baseline + 1e-10) * 100
        print(f"  {label:<20} | {res.mean():>12.4f} {res.std():>10.4f} {res.min():>10.4f} | {improve:>+.1f}%")

# Full run:
# stage_wo2_ablation(n_runs=30, D=30)

# Quick test:
stage_wo2_ablation(n_runs=5, D=30)
'''

param_sweep_code = r'''def stage_wo2_param_sweep(n_runs=15, D=30):
    """Parameter sensitivity: sweep each of the 4 parameters, fix others at defaults."""
    print("=" * 70)
    print(f"WO2 PARAMETER SENSITIVITY  (Rastrigin, D={D}, {n_runs} runs)")
    print("=" * 70)

    func, gfunc, lo, hi = MULTIMODAL_BENCHMARKS["Rastrigin"]
    N = max(30, D * 4)
    iters = max(500, D * 100)

    defaults = dict(lr=0.02, beta=0.9, alpha=0.1, sigma=0.15)
    sweeps = {
        "lr":    [0.005, 0.01, 0.02, 0.05, 0.1],
        "beta":  [0.7, 0.8, 0.9, 0.95, 0.99],
        "alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
        "sigma": [0.05, 0.1, 0.15, 0.25, 0.4],
    }

    ref_results = [wo2(func, gfunc, lo, hi, D=D, N=N, iterations=iters, seed=s, **defaults)
                   for s in range(n_runs)]
    ref_mean = np.mean(ref_results)

    for param, values in sweeps.items():
        print(f"\n  [{param}]  (default={defaults[param]})")
        print(f"  {'value':>8} | {'mean':>12} {'std':>10} | {'vs default':>12}")
        print(f"  {'-'*52}")
        for val in values:
            kwargs = {**defaults, param: val}
            res = np.array([wo2(func, gfunc, lo, hi, D=D, N=N, iterations=iters, seed=s, **kwargs)
                            for s in range(n_runs)])
            delta = (res.mean() - ref_mean) / (ref_mean + 1e-10) * 100
            marker = " <- default" if val == defaults[param] else ""
            print(f"  {val:>8.4g} | {res.mean():>12.4f} {res.std():>10.4f} | {delta:>+10.1f}%{marker}")

# Full run:
# stage_wo2_param_sweep(n_runs=15, D=30)

# Quick test:
stage_wo2_param_sweep(n_runs=3, D=30)
'''

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# 코드 K — WO2 (Weather Optimization v2)\n", "\n", "파라미터 4개 (`lr`, `beta`, `alpha`, `sigma`), O(N·D) 복잡도. 모멘텀 + best-centered 폭발."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [wo2_code]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## WO vs WO2 vs PSO 벤치마크 (Rastrigin, Schwefel, Levy)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [benchmark_code]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## WO2 Ablation Study (components A→D)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ablation_code]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## WO2 Parameter Sensitivity"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [param_sweep_code]
    },
]

# Insert after cell 7 (the WO algorithm)
insert_at = 8
cells = nb['cells']
nb['cells'] = cells[:insert_at] + new_cells + cells[insert_at:]

with open('3_최종_코드_정리.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
print(f"Inserted {len(new_cells)} cells at position {insert_at}")
