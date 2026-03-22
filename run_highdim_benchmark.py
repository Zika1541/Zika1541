"""
High-Dimensional CEC-style Benchmark Runner
WO2가 유리하다고 예상되는 고차원·어려운 함수 집중 평가

Functions : F1 Bent Cigar, F3 Rosenbrock, F4 Rastrigin,
            F5 Schaffer, F6 Levy, F7 Ackley
Algorithms: WO2, PSO, RIME, L-SHADE, DE
Dimensions: D = 100, 300
Runs      : D=100 → 10회,  D=300 → 5회

FES 예산(평가횟수) = 10000 * D  (CEC 표준)
"""

import json
import numpy as np
import time
from scipy.optimize import differential_evolution

# ── Load notebook and extract algorithms ──────────────────────────────────────
NB = "/home/user/Zika1541/3_최종_코드_정리.ipynb"

with open(NB) as f:
    nb = json.load(f)

ns = {}
for idx in [4, 7, 9]:         # PSO(4), WO(7), WO2(9)
    try:
        exec("".join(nb["cells"][idx]["source"]), ns)
    except Exception as e:
        print(f"  [warn] cell {idx}: {e}")

for idx in [43, 44, 45, 46]:  # CE(43), RIME(44), L-SHADE(45), extras(46)
    try:
        exec("".join(nb["cells"][idx]["source"]), ns)
    except Exception as e:
        print(f"  [warn] cell {idx}: {e}")

wo2        = ns["wo2"]
pso        = ns["pso"]
run_rime   = ns["run_rime"]
run_lshade = ns["run_lshade"]

# ── Rotation matrix & shift vector helpers ────────────────────────────────────

def _rotation_matrix(D, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)
    return Q

def _shift_vector(D, lo, hi, seed):
    rng = np.random.default_rng(seed)
    center = (lo + hi) / 2.0
    span   = (hi - lo) * 0.4
    return rng.uniform(center - span, center + span, D)

# ── Benchmark function base ───────────────────────────────────────────────────

class CEC2021Base:
    fid    = 0
    name   = "Base"
    bounds = (-100.0, 100.0)

    def __init__(self, D):
        self.D = D
        lo, hi = self.bounds
        seed   = self.fid * 10000 + D
        self.o = _shift_vector(D, lo, hi, seed)
        self.R = _rotation_matrix(D, seed + 1)

    def _z(self, X):
        return (X - self.o) @ self.R.T

    def grad(self, X):
        return np.zeros_like(X)

    def __call__(self, X):
        raise NotImplementedError


# ── 7 challenging functions ───────────────────────────────────────────────────

class F1_BentCigar(CEC2021Base):
    """단봉, 고도 비등방성 (조건수 10^6) — 방향 탐색 능력 테스트"""
    fid = 1; name = "F1-BentCigar"; bounds = (-100.0, 100.0)

    def __call__(self, X):
        z = self._z(X)
        return z[:, 0] ** 2 + 1e6 * np.sum(z[:, 1:] ** 2, axis=1)


class F3_Rosenbrock(CEC2021Base):
    """다봉 valley — 좁은 curved valley, 고차원에서 난도 급상승"""
    fid = 3; name = "F3-Rosenbrock"; bounds = (-50.0, 50.0)

    def __call__(self, X):
        z = self._z(X) + 1.0
        return np.sum(
            100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2,
            axis=1,
        )


class F4_Rastrigin(CEC2021Base):
    """다봉, ~10^D 지역 최솟값 — 고차원 multimodal 대표 함수"""
    fid = 4; name = "F4-Rastrigin"; bounds = (-5.12, 5.12)

    def __call__(self, X):
        z = self._z(X)
        D = z.shape[1]
        return 10 * D + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z), axis=1)


class F5_ExpandedSchaffer(CEC2021Base):
    """다봉, 연쇄 sin² 지형 — 고차원에서 연속 multimodal 구조"""
    fid = 5; name = "F5-Schaffer"; bounds = (-100.0, 100.0)

    @staticmethod
    def _f6(x, y):
        r2  = x ** 2 + y ** 2
        return 0.5 + (np.sin(np.sqrt(r2 + 1e-12)) ** 2 - 0.5) / (1.0 + 0.001 * r2) ** 2

    def __call__(self, X):
        z = self._z(X)
        N, D = z.shape
        result = np.zeros(N)
        for i in range(D - 1):
            result += self._f6(z[:, i], z[:, i + 1])
        result += self._f6(z[:, D - 1], z[:, 0])
        return result


class F6_Levy(CEC2021Base):
    """다봉, sin 주기 지형 — 고차원 multimodal, WO2 파동 위상 구조에 유리"""
    fid = 6; name = "F6-Levy"; bounds = (-10.0, 10.0)

    def __call__(self, X):
        z = self._z(X)
        w = 1.0 + (z - 1.0) / 4.0
        D = w.shape[1]
        term1  = np.sin(np.pi * w[:, 0]) ** 2
        term_i = np.sum(
            (w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2),
            axis=1,
        )
        term_n = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        return term1 + term_i + term_n


class F7_Ackley(CEC2021Base):
    """다봉, exp+cos 혼합 지형 — 전역 최솟값 주변에 무수한 지역 최솟값"""
    fid = 7; name = "F7-Ackley"; bounds = (-32.768, 32.768)

    def __call__(self, X):
        z = self._z(X)
        D = z.shape[1]
        a, b, c = 20.0, 0.2, 2 * np.pi
        sum_sq  = np.sum(z ** 2, axis=1) / D
        sum_cos = np.sum(np.cos(c * z), axis=1) / D
        return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e


# ── DE wrapper ────────────────────────────────────────────────────────────────

def run_de(func, lo, hi, D, N, iterations, seed):
    budget  = N * iterations
    popsize = max(5, N // max(D, 1))
    maxiter = max(1, budget // (popsize * D))

    def f1d(x):
        return float(func(np.asarray(x).reshape(1, -1))[0])

    res = differential_evolution(
        f1d, [(lo, hi)] * D,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-14, atol=1e-14,
        init="latinhypercube",
    )
    return res.fun


# ── Experiment setup ──────────────────────────────────────────────────────────

FUNC_CLASSES = [
    F1_BentCigar, F3_Rosenbrock, F4_Rastrigin,
    F5_ExpandedSchaffer, F6_Levy, F7_Ackley,
]
FUNC_NAMES = [cls.name for cls in FUNC_CLASSES]

# D=100: runs=10,  D=300: runs=5  (시간 절약)
DIM_CONFIG = {
    100: {"n_runs": 10},
    300: {"n_runs": 5},
}

ALGO_LIST = [
    ("WO2",
     lambda func, D, N, it, s:
         wo2(func, func.grad, func.bounds[0], func.bounds[1],
             D=D, N=N, iterations=it, seed=s)),
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

# ── Main benchmark loop ───────────────────────────────────────────────────────

all_results = {}

for D, cfg in DIM_CONFIG.items():
    N_RUNS = cfg["n_runs"]
    # CEC 표준: FES = 10000 * D
    FES   = 10000 * D
    N     = max(30, D * 2)       # population size
    iters = max(1, FES // N)     # iterations to match FES budget

    bench = [(cls.name, cls(D)) for cls in FUNC_CLASSES]

    print(f"\n{'='*100}")
    print(f"  D = {D:3d}  |  N = {N}  |  iterations = {iters}  |"
          f"  FES ≈ {N*iters:,}  |  runs = {N_RUNS}")
    print(f"{'='*100}")
    print(f"  {'Function':>16} | {'Algorithm':>10} | "
          f"{'mean':>14} {'std':>12} {'best':>12} | {'time/run':>9}")
    print(f"  {'-'*90}")

    for fname, func in bench:
        for aname, afunc in ALGO_LIST:
            runs, times = [], []
            for run in range(N_RUNS):
                t0 = time.time()
                try:
                    v = afunc(func, D, N, iters, run)
                except Exception as e:
                    v = float("nan")
                    print(f"    [err] {fname}/{aname}/run{run}: {e}")
                runs.append(v)
                times.append(time.time() - t0)

            arr = np.array(runs)
            all_results[(fname, D, aname)] = arr
            print(
                f"  {fname:>16} | {aname:>10} | "
                f"{np.nanmean(arr):>14.4e} {np.nanstd(arr):>12.4e} "
                f"{np.nanmin(arr):>12.4e} | {np.mean(times):>9.3f}s"
            )
        print(f"  {'-'*90}")

# ── Ranking summary ───────────────────────────────────────────────────────────

print(f"\n{'='*100}")
print("  FINAL RANKING  (낮을수록 좋음 — mean 기준, 함수×차원 평균 순위)")
print(f"{'='*100}")

rank_accum = {a: [] for a, _ in ALGO_LIST}

for D in DIM_CONFIG:
    for fname in FUNC_NAMES:
        means = {
            aname: np.nanmean(all_results[(fname, D, aname)])
            for aname, _ in ALGO_LIST
        }
        for rank, a in enumerate(sorted(means, key=means.get), 1):
            rank_accum[a].append(rank)

n_configs = len(DIM_CONFIG) * len(FUNC_NAMES)
print(f"  {'Algorithm':>10} | {'Avg Rank':>9} | {'#1st':>6} | {'#2nd':>6}"
      f"  (총 {n_configs} configs: {len(FUNC_NAMES)} functions × {len(DIM_CONFIG)} dims)")
print(f"  {'-'*65}")

summary = sorted(rank_accum.items(), key=lambda kv: np.mean(kv[1]))
for a, ranks in summary:
    wins   = sum(1 for r in ranks if r == 1)
    second = sum(1 for r in ranks if r == 2)
    print(f"  {a:>10} | {np.mean(ranks):>9.2f} | {wins:>6} | {second:>6}")

print()
print("함수 설명:")
print("  F1-BentCigar : 단봉, 조건수 10^6  — 방향 탐색 능력 핵심 테스트")
print("  F3-Rosenbrock: 다봉 valley         — 좁은 curved valley, D 클수록 난도↑")
print("  F4-Rastrigin : 다봉, ~10^D 지역최솟값 — 고차원 multimodal 대표")
print("  F5-Schaffer  : 다봉 sin² 연쇄 지형 — 고차원 연속 multimodal")
print("  F6-Levy      : 다봉 sin 주기 지형  — WO2 파동 위상 구조에 유리")
print("  F7-Ackley    : 다봉 exp+cos 혼합   — 전역 주변 무수한 지역 최솟값")
print()
print("알고리즘:")
print("  WO2     : Weather Optimization v2 (본 연구)")
print("  PSO     : Particle Swarm Optimization")
print("  RIME    : Rime-Ice Optimization (Su et al. 2023)")
print("  L-SHADE : Linear pop-reduction SHADE (Tanabe & Fukunaga 2014)")
print("  DE      : Differential Evolution (scipy)")
