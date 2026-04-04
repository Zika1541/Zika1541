"""
CEC2021-style Benchmark Runner

Functions : F1 Bent Cigar, F2 Zakharov, F3 Rosenbrock, F4 Rastrigin, F5 Schaffer
Algorithms: WO2, PSO, RIME, L-SHADE, DE
Dimensions: D = 10, 50, 100
Runs      : 10 per configuration

CEC2021 스타일 – 각 함수에 고정 시드로 생성된 shift vector + rotation matrix 적용
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
    """Generate random orthogonal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)
    return Q

def _shift_vector(D, lo, hi, seed):
    """Random shift vector within 40% of bounds from center."""
    rng = np.random.default_rng(seed)
    center = (lo + hi) / 2.0
    span   = (hi - lo) * 0.4
    return rng.uniform(center - span, center + span, D)

# ── CEC2021-style benchmark function classes ───────────────────────────────────

class CEC2021Base:
    """Shifted + Rotated benchmark function base class."""
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
        """Apply shift and rotation: z = (X - o) @ R^T  →  shape (N, D)"""
        return (X - self.o) @ self.R.T

    def grad(self, X):
        """Zero gradient (black-box mode for WO2 condensation phase)."""
        return np.zeros_like(X)

    def __call__(self, X):
        raise NotImplementedError


class F1_BentCigar(CEC2021Base):
    """F1: Shifted+Rotated Bent Cigar — unimodal, highly ill-conditioned."""
    fid    = 1
    name   = "F1-BentCigar"
    bounds = (-100.0, 100.0)

    def __call__(self, X):
        z = self._z(X)
        return z[:, 0] ** 2 + 1e6 * np.sum(z[:, 1:] ** 2, axis=1)


class F2_Zakharov(CEC2021Base):
    """F2: Shifted+Rotated Zakharov — unimodal, plate + bowl structure."""
    fid    = 2
    name   = "F2-Zakharov"
    bounds = (-10.0, 10.0)

    def __call__(self, X):
        z  = self._z(X)
        D  = z.shape[1]
        i  = np.arange(1, D + 1, dtype=float)
        s1 = np.sum(z ** 2, axis=1)
        s2 = np.sum(0.5 * i * z, axis=1)
        return s1 + s2 ** 2 + s2 ** 4


class F3_Rosenbrock(CEC2021Base):
    """F3: Shifted+Rotated Rosenbrock — multimodal valley, global opt at z=1."""
    fid    = 3
    name   = "F3-Rosenbrock"
    bounds = (-50.0, 50.0)

    def __call__(self, X):
        z = self._z(X) + 1.0          # shift so optimum aligns with z=1
        return np.sum(
            100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2,
            axis=1,
        )


class F4_Rastrigin(CEC2021Base):
    """F4: Shifted+Rotated Rastrigin — highly multimodal, ~10^D local optima."""
    fid    = 4
    name   = "F4-Rastrigin"
    bounds = (-5.12, 5.12)

    def __call__(self, X):
        z = self._z(X)
        D = z.shape[1]
        return 10 * D + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z), axis=1)


class F5_ExpandedSchaffer(CEC2021Base):
    """F5: Expanded Schaffer F6 — multimodal, sinusoidal landscape."""
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
        z  = self._z(X)
        N, D = z.shape
        result = np.zeros(N)
        for i in range(D - 1):
            result += self._f6(z[:, i], z[:, i + 1])
        result += self._f6(z[:, D - 1], z[:, 0])
        return result


# ── DE wrapper (scipy) ────────────────────────────────────────────────────────

def run_de(func, lo, hi, D, N, iterations, seed):
    """Differential Evolution via scipy — budget = N * iterations evaluations."""
    budget  = N * iterations
    popsize = max(5, N // max(D, 1))   # population = popsize * D agents
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

N_RUNS    = 10
DIMS      = [10, 50, 100]
FUNC_NAMES = ["F1-BentCigar", "F2-Zakharov", "F3-Rosenbrock",
              "F4-Rastrigin", "F5-Schaffer"]

FUNC_CLASSES = [F1_BentCigar, F2_Zakharov, F3_Rosenbrock,
                F4_Rastrigin, F5_ExpandedSchaffer]

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

all_results = {}   # key: (fname, D, aname) → np.array of N_RUNS values

for D in DIMS:
    N     = max(30, D * 2)
    iters = 1000

    # Instantiate functions once per D (fixed shift/rotation)
    bench = [(cls.name if hasattr(cls, "name") else cls.__name__, cls(D))
             for cls in FUNC_CLASSES]

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

# ── Final ranking summary ─────────────────────────────────────────────────────

print(f"\n{'='*95}")
print("  FINAL RANKING  (낮을수록 좋음 — mean 기준, 함수×차원 평균 순위)")
print(f"{'='*95}")

rank_accum = {a: [] for a, _ in ALGO_LIST}

for D in DIMS:
    for fname in FUNC_NAMES:
        means = {
            aname: np.nanmean(all_results[(fname, D, aname)])
            for aname, _ in ALGO_LIST
        }
        for rank, a in enumerate(sorted(means, key=means.get), 1):
            rank_accum[a].append(rank)

n_configs = len(DIMS) * len(FUNC_NAMES)
print(f"  {'Algorithm':>10} | {'Avg Rank':>9} | {'#1st':>6} | {'#2nd':>6}"
      f"  (총 {n_configs} configs: {len(FUNC_NAMES)} functions × {len(DIMS)} dims)")
print(f"  {'-'*60}")

summary = sorted(rank_accum.items(), key=lambda kv: np.mean(kv[1]))
for i, (a, ranks) in enumerate(summary, 1):
    wins   = sum(1 for r in ranks if r == 1)
    second = sum(1 for r in ranks if r == 2)
    print(f"  {a:>10} | {np.mean(ranks):>9.2f} | {wins:>6} | {second:>6}")

print()
print("CEC2021-style 벤치마크 함수 설명:")
print("  F1-BentCigar : Shifted+Rotated Bent Cigar — 단봉(unimodal), 고도 비등방성")
print("  F2-Zakharov  : Shifted+Rotated Zakharov   — 단봉, bowl+plate 구조")
print("  F3-Rosenbrock: Shifted+Rotated Rosenbrock — 다봉(multimodal), 좁은 valley")
print("  F4-Rastrigin : Shifted+Rotated Rastrigin  — 다봉, ~10^D 지역 최솟값")
print("  F5-Schaffer  : Expanded Schaffer F6       — 다봉, sin 기반 복잡 지형")
print()
print("알고리즘 설명:")
print("  WO2     : Weather Optimization v2 (본 연구, 2026)")
print("  PSO     : Particle Swarm Optimization (기본 베이스라인)")
print("  RIME    : Rime-Ice Optimization Algorithm (Su et al. 2023)")
print("  L-SHADE : Linear pop-reduction SHADE (Tanabe & Fukunaga 2014)")
print("  DE      : Differential Evolution (scipy built-in)")
