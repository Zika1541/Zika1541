"""
CEC2021-style Benchmark — Colab 독립 실행 버전
================================================
함수  : F1 BentCigar / F2 Zakharov / F3 Rosenbrock / F4 Rastrigin / F5 Schaffer
알고리즘: WO2 · PSO · RIME · L-SHADE · DE
차원  : D = 10, 50, 100
실행  : 10 runs per config

사용법 (Colab):
  !pip install scipy -q
  # 이 파일을 업로드하거나 아래 전체 코드를 셀에 붙여넣기
  exec(open('cec2021_colab.py').read())
"""

# ──────────────────────────────────────────────────────────────────────────────
#  라이브러리
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from scipy.optimize import differential_evolution

# ──────────────────────────────────────────────────────────────────────────────
#  PSO
# ──────────────────────────────────────────────────────────────────────────────
def pso(func, lower, upper, D=10, N=80, iterations=2000, seed=0):
    rng = np.random.default_rng(seed)
    w, c1, c2 = 0.729, 1.494, 1.494
    pos       = rng.uniform(lower, upper, (N, D))
    vel       = rng.uniform(-(upper - lower), (upper - lower), (N, D)) * 0.1
    pbest_pos = pos.copy()
    pbest_val = func(pos)
    gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()
    gbest_val = float(np.min(pbest_val))
    for _ in range(iterations):
        r1 = rng.random((N, D)); r2 = rng.random((N, D))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, lower, upper)
        vals = func(pos)
        imp = vals < pbest_val
        pbest_pos[imp] = pos[imp]; pbest_val[imp] = vals[imp]
        if np.min(pbest_val) < gbest_val:
            gbest_val = float(np.min(pbest_val))
            gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()
    return gbest_val

# ──────────────────────────────────────────────────────────────────────────────
#  WO2 (Weather Optimization v2)
# ──────────────────────────────────────────────────────────────────────────────
def wo2(func, grad_func, lower=-5.12, upper=5.12, D=10, N=80, iterations=2000,
        lr=0.05, beta=0.9, alpha=0.05, sigma=0.15, seed=0, n_blocks=1, return_history=False):
    """WO2: 4-parameter simplified Weather Optimization (2026)
    n_blocks=1 : 기존 동작 (전체 D차원 폭발)
    n_blocks>1 : 블록 폭발 — D//n_blocks 차원만 랜덤화, 나머지는 x_best 고정
    """
    rng  = np.random.default_rng(seed)
    span = upper - lower
    window = max(10, int(round(1.0 / (1.0 - beta))))

    x          = rng.uniform(lower, upper, (N, D))
    v          = np.zeros((N, D))
    loss       = func(x)
    loss_prev  = loss.copy()
    energy     = np.ones(N)
    best_idx   = int(np.argmin(loss))
    x_best     = x[best_idx].copy()
    best_loss  = float(loss[best_idx])
    last_exp   = np.full(N, -window, dtype=int)
    n_explode  = max(1, int(sigma * N))
    history    = []

    for t in range(iterations):
        cos_val = 0.5 * (1.0 + np.cos(np.pi * t / iterations))
        lr_t    = lr * max(cos_val, 0.01)

        g       = grad_func(x)
        g_norm  = np.linalg.norm(g, axis=1, keepdims=True) + 1e-8
        g_clip  = g / np.maximum(g_norm, 1.0)
        v       = beta * v - lr_t * g_clip
        attract = alpha * (x_best - x)
        x       = np.clip(x + v + attract, lower, upper)

        loss    = func(x)
        delta   = np.abs(loss - loss_prev)
        energy  = beta * energy + (1.0 - beta) * delta
        loss_prev = loss.copy()

        idx_t = int(np.argmin(loss))
        if loss[idx_t] < best_loss:
            best_loss = float(loss[idx_t])
            x_best    = x[idx_t].copy()

        n_exp = 0
        if t % window == 0:
            eligible     = (t - last_exp) >= window
            eligible_idx = np.where(eligible)[0]
            if len(eligible_idx) >= 2:
                n_exp      = min(n_explode, len(eligible_idx))
                order      = np.argpartition(energy[eligible_idx], n_exp - 1)
                explode_idx = eligible_idx[order[:n_exp]]
                if n_blocks <= 1:
                    noise = rng.standard_normal((n_exp, D))
                    x[explode_idx] = np.clip(x_best + sigma * span * noise, lower, upper)
                else:
                    block_size = max(1, D // n_blocks)
                    free_dims  = rng.choice(D, size=block_size, replace=False)
                    x_new      = np.tile(x_best, (n_exp, 1))
                    noise      = rng.standard_normal((n_exp, block_size))
                    x_new[:, free_dims] = x_best[free_dims] + sigma * span * noise
                    x[explode_idx] = np.clip(x_new, lower, upper)
                v[explode_idx] = 0.0
                energy[explode_idx] = np.median(energy)
                last_exp[explode_idx] = t

        if return_history:
            history.append({"iter": t, "best_loss": best_loss, "n_explode": n_exp})

    return (best_loss, history) if return_history else best_loss

# ──────────────────────────────────────────────────────────────────────────────
#  RIME (Su et al. 2023)
# ──────────────────────────────────────────────────────────────────────────────
def run_rime(func, lower, upper, D=10, N=80, iterations=2000, seed=0, W=5.0):
    rng  = np.random.default_rng(seed)
    span = upper - lower
    X       = rng.uniform(lower, upper, (N, D))
    fitness = func(X)
    best_idx = int(np.argmin(fitness))
    X_best  = X[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for t in range(1, iterations + 1):
        norm_t      = t / iterations
        rime_factor = W * span / (2.0 * np.sqrt(t * iterations))
        r1    = rng.random((N, D))
        X_new = X + rime_factor * (X_best[None, :] - X) * r1
        r2    = rng.random((N, D))
        X_new = np.where(r2 < norm_t, X_best[None, :], X_new)
        X_new = np.clip(X_new, lower, upper)
        fit_new  = func(X_new)
        improved = fit_new < fitness
        X        = np.where(improved[:, None], X_new, X)
        fitness  = np.where(improved, fit_new, fitness)
        cur_best = int(np.argmin(fitness))
        if fitness[cur_best] < best_fit:
            best_fit = float(fitness[cur_best])
            X_best   = X[cur_best].copy()
    return best_fit

# ──────────────────────────────────────────────────────────────────────────────
#  L-SHADE (Tanabe & Fukunaga 2014)
# ──────────────────────────────────────────────────────────────────────────────
def run_lshade(func, lower, upper, D=10, N=80, iterations=2000, seed=0, H=6):
    rng    = np.random.default_rng(seed)
    N_init = N
    N_min  = max(4, D // 5)
    budget = N * iterations
    M_F    = np.full(H, 0.5)
    M_CR   = np.full(H, 0.5)
    k      = 0
    pop     = rng.uniform(lower, upper, (N_init, D))
    fitness = func(pop)
    best_fit = float(np.min(fitness))
    archive  = np.empty((0, D))
    evals    = N_init

    while evals < budget:
        N_cur = len(pop)
        N_new = max(N_min, int(N_min + (N_init - N_min) * (budget - evals) / budget))
        if N_new < N_cur:
            keep    = np.argpartition(fitness, N_new)[:N_new]
            pop     = pop[keep]; fitness = fitness[keep]; N_cur = N_new

        ri  = rng.integers(0, H, N_cur)
        F   = np.clip(rng.standard_cauchy(N_cur) * 0.1 + M_F[ri],  0.01, 1.0)
        CR  = np.clip(rng.standard_normal(N_cur) * 0.1 + M_CR[ri], 0.0,  1.0)

        p_n    = max(2, int(0.11 * N_cur))
        p_idx  = np.argpartition(fitness, p_n)[:p_n]
        xpbest = pop[p_idx[rng.integers(p_n, size=N_cur)]]

        shifts = rng.integers(1, N_cur, N_cur)
        x_r1   = pop[(np.arange(N_cur) + shifts) % N_cur]

        combined = np.vstack([pop, archive]) if len(archive) > 0 else pop
        x_r2     = combined[rng.integers(len(combined), size=N_cur)]

        mutants = np.clip(pop + F[:, None] * (xpbest - pop) + F[:, None] * (x_r1 - x_r2),
                          lower, upper)
        j_rand  = rng.integers(D, size=N_cur)
        mask    = rng.random((N_cur, D)) < CR[:, None]
        mask[np.arange(N_cur), j_rand] = True
        trials  = np.where(mask, mutants, pop)

        f_trials = func(trials)
        evals   += N_cur
        if evals > budget:
            break

        improved = f_trials <= fitness
        if improved.any():
            archive = np.vstack([archive, pop[improved]]) if len(archive) > 0 else pop[improved].copy()
            if len(archive) > N_init:
                archive = archive[rng.choice(len(archive), N_init, replace=False)]

        S_F = F[improved]; S_CR = CR[improved]
        S_d = np.abs(f_trials[improved] - fitness[improved])
        pop     = np.where(improved[:, None], trials, pop)
        fitness = np.where(improved, f_trials, fitness)
        if float(np.min(fitness)) < best_fit:
            best_fit = float(np.min(fitness))

        if len(S_F) > 0:
            w = S_d / (S_d.sum() + 1e-12)
            M_F[k]  = np.dot(w, S_F**2) / (np.dot(w, S_F) + 1e-12)
            M_CR[k] = np.dot(w, S_CR)
            k = (k + 1) % H

    return best_fit

# ──────────────────────────────────────────────────────────────────────────────
#  DE (scipy wrapper) — 고정 budget 기반
# ──────────────────────────────────────────────────────────────────────────────
def run_de(func, lower, upper, D=10, N=80, iterations=2000, seed=0):
    """DE via scipy — budget = N × iterations function evaluations."""
    budget   = N * iterations
    popsize  = 15                              # scipy 기본값 (pop = 15*D)
    maxiter  = max(10, budget // (popsize * D))

    def f1d(x):
        return float(func(np.asarray(x).reshape(1, -1))[0])

    res = differential_evolution(
        f1d, [(lower, upper)] * D,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-14, atol=1e-14,
        init="latinhypercube", polish=False,
    )
    return float(res.fun)

# ──────────────────────────────────────────────────────────────────────────────
#  CEC2021-style 벤치마크 함수 (F1–F5)
#  각 함수 = 고정 시드로 생성된 shift vector + QR rotation matrix
# ──────────────────────────────────────────────────────────────────────────────
def _rotation_matrix(D, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return Q

def _shift_vector(D, lo, hi, seed):
    rng = np.random.default_rng(seed)
    c   = (lo + hi) / 2.0
    s   = (hi - lo) * 0.4
    return rng.uniform(c - s, c + s, D)


class CEC2021Base:
    fid    = 0
    name   = "Base"
    bounds = (-100.0, 100.0)

    def __init__(self, D):
        self.D  = D
        lo, hi  = self.bounds
        seed    = self.fid * 10000 + D
        self.o  = _shift_vector(D, lo, hi, seed)
        self.R  = _rotation_matrix(D, seed + 1)

    def _z(self, X):
        return (X - self.o) @ self.R.T

    def grad(self, X):
        """Zero gradient — WO2 black-box 모드로 동작."""
        return np.zeros_like(X)

    def __call__(self, X):
        raise NotImplementedError


class F1_BentCigar(CEC2021Base):
    """단봉(unimodal), 고도 비등방성 — 축방향 탐색 어려움."""
    fid    = 1
    name   = "F1-BentCigar"
    bounds = (-100.0, 100.0)

    def __call__(self, X):
        z = self._z(X)
        return z[:, 0] ** 2 + 1e6 * np.sum(z[:, 1:] ** 2, axis=1)


class F2_Zakharov(CEC2021Base):
    """단봉, bowl+plate 혼합 구조."""
    fid    = 2
    name   = "F2-Zakharov"
    bounds = (-10.0, 10.0)

    def __call__(self, X):
        z  = self._z(X)
        i  = np.arange(1, z.shape[1] + 1, dtype=float)
        s1 = np.sum(z ** 2, axis=1)
        s2 = np.sum(0.5 * i * z, axis=1)
        return s1 + s2 ** 2 + s2 ** 4


class F3_Rosenbrock(CEC2021Base):
    """다봉, 좁고 구불구불한 valley."""
    fid    = 3
    name   = "F3-Rosenbrock"
    bounds = (-50.0, 50.0)

    def __call__(self, X):
        z = self._z(X) + 1.0
        return np.sum(
            100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2,
            axis=1,
        )


class F4_Rastrigin(CEC2021Base):
    """다봉, ~10^D 지역 최솟값 — 탈출 난이도 높음."""
    fid    = 4
    name   = "F4-Rastrigin"
    bounds = (-5.12, 5.12)

    def __call__(self, X):
        z = self._z(X)
        return 10 * z.shape[1] + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z), axis=1)


class F5_ExpandedSchaffer(CEC2021Base):
    """다봉, sin 기반 복잡 지형 — 지역 최솟값 밀도 높음."""
    fid    = 5
    name   = "F5-Schaffer"
    bounds = (-100.0, 100.0)

    @staticmethod
    def _f6(x, y):
        r2  = x ** 2 + y ** 2
        num = np.sin(np.sqrt(r2 + 1e-12)) ** 2 - 0.5
        den = (1.0 + 0.001 * r2) ** 2
        return 0.5 + num / den

    def __call__(self, X):
        z      = self._z(X)
        N, D   = z.shape
        result = np.zeros(N)
        for i in range(D - 1):
            result += self._f6(z[:, i], z[:, i + 1])
        result += self._f6(z[:, D - 1], z[:, 0])
        return result


# ──────────────────────────────────────────────────────────────────────────────
#  실험 설정
# ──────────────────────────────────────────────────────────────────────────────
N_RUNS      = 10
DIMS        = [10, 50, 100]
FUNC_CLASSES = [F1_BentCigar, F2_Zakharov, F3_Rosenbrock, F4_Rastrigin, F5_ExpandedSchaffer]
FUNC_NAMES  = [cls.name for cls in FUNC_CLASSES]

ALGO_LIST = [
    ("WO2",
     lambda func, D, N, it, s:
         wo2(func, func.grad, func.bounds[0], func.bounds[1],
             D=D, N=N, iterations=it, seed=s)),
    ("WO2-Block2",
     lambda func, D, N, it, s:
         wo2(func, func.grad, func.bounds[0], func.bounds[1],
             D=D, N=N, iterations=it, seed=s, n_blocks=2)),
    ("WO2-Block4",
     lambda func, D, N, it, s:
         wo2(func, func.grad, func.bounds[0], func.bounds[1],
             D=D, N=N, iterations=it, seed=s, n_blocks=4)),
    ("PSO",
     lambda func, D, N, it, s:
         pso(func, func.bounds[0], func.bounds[1],
             D=D, N=N, iterations=it, seed=s)),
    ("RIME",
     lambda func, D, N, it, s:
         run_rime(func, func.bounds[0], func.bounds[1],
                  D=D, N=N, iterations=it, seed=s)),
    ("L-SHADE",
     lambda func, D, N, it, s:
         run_lshade(func, func.bounds[0], func.bounds[1],
                    D=D, N=N, iterations=it, seed=s)),
    ("DE",
     lambda func, D, N, it, s:
         run_de(func, func.bounds[0], func.bounds[1],
                D=D, N=N, iterations=it, seed=s)),
]

# ──────────────────────────────────────────────────────────────────────────────
#  메인 벤치마크 루프
# ──────────────────────────────────────────────────────────────────────────────
all_results = {}   # (fname, D, aname) → np.array(N_RUNS)

for D in DIMS:
    N     = max(30, D * 2)
    iters = 1000

    bench = [(cls.name, cls(D)) for cls in FUNC_CLASSES]

    print(f"\n{'='*95}")
    print(f"  D = {D:3d}  |  population N = {N}  |  iterations = {iters}  |  runs = {N_RUNS}")
    print(f"{'='*95}")
    print(f"  {'Function':>15} | {'Algorithm':>10} | "
          f"{'mean':>14} {'std':>12} {'best':>12} | {'time/run':>9}")
    print(f"  {'-'*85}")

    for fname, func in bench:
        for aname, afunc in ALGO_LIST:
            runs, times = [], []
            for run in range(N_RUNS):
                t0 = time.time()
                try:
                    v = afunc(func, D, N, iters, run)
                except Exception as e:
                    print(f"    [ERR] {aname} D={D} run={run}: {e}")
                    v = float("nan")
                runs.append(v)
                times.append(time.time() - t0)

            arr = np.array(runs)
            all_results[(fname, D, aname)] = arr
            print(
                f"  {fname:>15} | {aname:>10} | "
                f"{np.nanmean(arr):>14.4e} {np.nanstd(arr):>12.4e} "
                f"{np.nanmin(arr):>12.4e} | {np.mean(times):>9.3f}s"
            )
        print(f"  {'-'*85}")

# ──────────────────────────────────────────────────────────────────────────────
#  최종 순위 요약
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*95}")
print("  FINAL RANKING  (낮을수록 좋음 — mean 기준)")
print(f"{'='*95}")

rank_accum = {a: [] for a, _ in ALGO_LIST}

for D in DIMS:
    for fname in FUNC_NAMES:
        means = {aname: np.nanmean(all_results[(fname, D, aname)])
                 for aname, _ in ALGO_LIST}
        for rank, a in enumerate(sorted(means, key=means.get), 1):
            rank_accum[a].append(rank)

n_configs = len(DIMS) * len(FUNC_NAMES)
print(f"  {'Algorithm':>10} | {'Avg Rank':>9} | {'#1st':>6} | {'#2nd':>6}"
      f"  (총 {n_configs} configs: {len(FUNC_NAMES)} functions × {len(DIMS)} dims)")
print(f"  {'-'*60}")
for a, ranks in sorted(rank_accum.items(), key=lambda kv: np.mean(kv[1])):
    wins   = sum(1 for r in ranks if r == 1)
    second = sum(1 for r in ranks if r == 2)
    print(f"  {a:>10} | {np.mean(ranks):>9.2f} | {wins:>6} | {second:>6}")

print()
print("함수 설명:")
print("  F1-BentCigar : Shifted+Rotated Bent Cigar  — 단봉, 고도 비등방성")
print("  F2-Zakharov  : Shifted+Rotated Zakharov    — 단봉, bowl+plate")
print("  F3-Rosenbrock: Shifted+Rotated Rosenbrock  — 다봉, 구불구불한 valley")
print("  F4-Rastrigin : Shifted+Rotated Rastrigin   — 다봉, 10^D 지역 최솟값")
print("  F5-Schaffer  : Expanded Schaffer F6        — 다봉, sin 기반 복잡 지형")
print()
print("알고리즘 설명:")
print("  WO2     : Weather Optimization v2 (본 연구, 2026)")
print("  PSO     : Particle Swarm Optimization (기본 베이스라인)")
print("  RIME    : Rime-Ice Optimization (Su et al. 2023)")
print("  L-SHADE : Linear pop-reduction SHADE (Tanabe & Fukunaga 2014)")
print("  DE      : Differential Evolution (scipy, popsize=15)")
