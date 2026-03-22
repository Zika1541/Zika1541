"""
WO2-Favorable CEC-style Evaluation
====================================
WO2 장점이 드러나는 조건에서 5개 알고리즘 비교:
  1. 함수 선택  : 매끄러운 단봉/좁은 valley (그래디언트 신호 명확)
  2. 평가 지표  : 초기 수렴 속도(30% 예산), 목표 성공률, AUCC, 최종값
  3. 차원       : D=10 (그래디언트 가장 신뢰도 높음)

알고리즘:  WO2 · PSO · RIME · L-SHADE · DE
함수:      Sphere / Elliptic / Zakharov / Rosenbrock / Ackley
지표:      mean@full · mean@30% · 성공률(%) · AUCC(정규화)
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

# ──────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
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


# ──────────────────────────────────────────────────────────────────────────────
# 벤치마크 함수 (Shifted + Rotated)
# ──────────────────────────────────────────────────────────────────────────────
class BenchBase:
    fid    = 0
    name   = "Base"
    bounds = (-100.0, 100.0)
    target = 1e-4        # 성공률 기준 (f(x) < target)

    def __init__(self, D):
        self.D  = D
        lo, hi  = self.bounds
        seed    = self.fid * 10000 + D
        self.o  = _shift_vector(D, lo, hi, seed)
        self.R  = _rotation_matrix(D, seed + 1)

    def _z(self, X):
        return (X - self.o) @ self.R.T

    def grad(self, X):
        raise NotImplementedError

    def __call__(self, X):
        raise NotImplementedError


# ── F_A: Sphere — 순수 2차 볼록, 그래디언트 하강 최적 ──────────────────────
class F_Sphere(BenchBase):
    """f(z) = sum(z_i^2).  전역 최솟값 = 0 at z=0."""
    fid    = 10
    name   = "FA-Sphere"
    bounds = (-100.0, 100.0)
    target = 1e-6

    def __call__(self, X):
        z = self._z(X)
        return np.sum(z ** 2, axis=1)

    def grad(self, X):
        z   = self._z(X)                         # (N, D)
        dz  = 2.0 * z                            # df/dz
        return dz @ self.R                       # chain rule: df/dX = df/dz · R


# ── F_B: Elliptic — 고도 비등방성 볼록, 그래디언트 방향 중요 ───────────────
class F_Elliptic(BenchBase):
    """f(z) = sum(10^{6*(i-1)/(D-1)} * z_i^2).
    조건수 = 10^6; 그래디언트 방향이 최단 경로 제공."""
    fid    = 11
    name   = "FB-Elliptic"
    bounds = (-100.0, 100.0)
    target = 1e-4

    def _weights(self, D):
        return 10.0 ** (6.0 * np.arange(D) / max(D - 1, 1))

    def __call__(self, X):
        z = self._z(X)
        w = self._weights(z.shape[1])
        return np.sum(w * z ** 2, axis=1)

    def grad(self, X):
        z  = self._z(X)
        w  = self._weights(z.shape[1])
        dz = 2.0 * w * z
        return dz @ self.R


# ── F_C: Zakharov — 부드러운 bowl + plate 혼합 ─────────────────────────────
class F_Zakharov(BenchBase):
    """Shifted+Rotated Zakharov.  단봉, 거의 볼록."""
    fid    = 12
    name   = "FC-Zakharov"
    bounds = (-10.0, 10.0)
    target = 1e-4

    def __call__(self, X):
        z  = self._z(X)
        i  = np.arange(1, z.shape[1] + 1, dtype=float)
        s1 = np.sum(z ** 2, axis=1)
        s2 = np.sum(0.5 * i * z, axis=1)
        return s1 + s2 ** 2 + s2 ** 4

    def grad(self, X):
        z  = self._z(X)
        D  = z.shape[1]
        i  = np.arange(1, D + 1, dtype=float)
        s2 = np.sum(0.5 * i * z, axis=1, keepdims=True)
        dz = 2.0 * z + (2.0 * s2 + 4.0 * s2 ** 3) * (0.5 * i)
        return dz @ self.R


# ── F_D: Rosenbrock — 좁고 구불구불한 valley, 모멘텀 효과 ──────────────────
class F_Rosenbrock(BenchBase):
    """Shifted+Rotated Rosenbrock.  약다봉, 모멘텀 + 그래디언트 = 효율적."""
    fid    = 13
    name   = "FD-Rosenbrock"
    bounds = (-50.0, 50.0)
    target = 1.0         # valley 진입 기준

    def __call__(self, X):
        z = self._z(X) + 1.0
        return np.sum(
            100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2,
            axis=1,
        )

    def grad(self, X):
        z    = self._z(X) + 1.0
        N, D = z.shape
        dz   = np.zeros_like(z)
        xi   = z[:, :-1]; xi1 = z[:, 1:]
        dz[:, :-1] += -400.0 * xi * (xi1 - xi ** 2) + 2.0 * (xi - 1.0)
        dz[:, 1:]  +=  200.0 * (xi1 - xi ** 2)
        return dz @ self.R


# ── F_E: Ackley — 부드러운 다봉, sin 기반 완만한 요철 ─────────────────────
class F_Ackley(BenchBase):
    """Shifted+Rotated Ackley.  cos 기반 다봉 but 그래디언트 방향 신뢰도 ↑."""
    fid    = 14
    name   = "FE-Ackley"
    bounds = (-32.0, 32.0)
    target = 0.5

    def __call__(self, X):
        z  = self._z(X)
        D  = z.shape[1]
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        t1 = -a * np.exp(-b * np.sqrt(np.mean(z ** 2, axis=1)))
        t2 = -np.exp(np.mean(np.cos(c * z), axis=1))
        return t1 + t2 + a + np.e

    def grad(self, X):
        z  = self._z(X)
        D  = z.shape[1]
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        r        = np.sqrt(np.mean(z ** 2, axis=1) + 1e-12)
        coeff1   = (a * b * np.exp(-b * r) / (r * D))[:, None]
        coeff2   = (np.exp(np.mean(np.cos(c * z), axis=1)) * c / D)[:, None]
        dz       = coeff1 * z + coeff2 * np.sin(c * z)
        return dz @ self.R


FUNC_CLASSES = [F_Sphere, F_Elliptic, F_Zakharov, F_Rosenbrock, F_Ackley]


# ──────────────────────────────────────────────────────────────────────────────
# 수렴 이력 추적 래퍼
# ──────────────────────────────────────────────────────────────────────────────
class ConvergenceTracker:
    """알고리즘에 넘기는 함수를 감싸 best-so-far 이력을 기록."""
    def __init__(self, func, checkpoints_frac):
        """
        checkpoints_frac: list of float, e.g. [0.1, 0.3, 0.5, 1.0]
          각 체크포인트에서 best_so_far 기록 (total_budget 기준)
        """
        self.func    = func
        self.frac    = checkpoints_frac
        self._calls  = 0
        self._best   = np.inf
        self._budget = None
        self.history = {}            # frac → best value

    def init_budget(self, budget):
        self._budget = budget
        self._calls  = 0
        self._best   = np.inf
        self.history = {}

    def __call__(self, X):
        vals = self.func(X)
        cur  = float(np.min(vals))
        if cur < self._best:
            self._best = cur
        self._calls += len(X)
        if self._budget is not None:
            done = self._calls / self._budget
            for f in self.frac:
                if f not in self.history and done >= f:
                    self.history[f] = self._best
        return vals

    def grad(self, X):
        return self.func.grad(X)


# ──────────────────────────────────────────────────────────────────────────────
# 알고리즘 (이력 추적 버전)
# ──────────────────────────────────────────────────────────────────────────────
def wo2(func, grad_func, lower, upper, D, N, iterations,
        lr=0.05, beta=0.9, alpha=0.05, sigma=0.15, seed=0):
    rng   = np.random.default_rng(seed)
    span  = upper - lower
    window = max(10, int(round(1.0 / (1.0 - beta))))
    x       = rng.uniform(lower, upper, (N, D))
    v       = np.zeros((N, D))
    loss    = func(x)
    lp      = loss.copy()
    energy  = np.ones(N)
    bi      = int(np.argmin(loss))
    xb      = x[bi].copy()
    bl      = float(loss[bi])
    last_e  = np.full(N, -window, dtype=int)
    n_exp   = max(1, int(sigma * N))
    for t in range(iterations):
        cos_v = 0.5 * (1.0 + np.cos(np.pi * t / iterations))
        lr_t  = lr * max(cos_v, 0.01)
        g     = grad_func(x)
        gn    = np.linalg.norm(g, axis=1, keepdims=True) + 1e-8
        gc    = g / np.maximum(gn, 1.0)
        v     = beta * v - lr_t * gc
        x     = np.clip(x + v + alpha * (xb - x), lower, upper)
        loss  = func(x)
        delta = np.abs(loss - lp)
        energy = beta * energy + (1.0 - beta) * delta
        lp    = loss.copy()
        idx_t = int(np.argmin(loss))
        if loss[idx_t] < bl:
            bl = float(loss[idx_t]); xb = x[idx_t].copy()
        if t % window == 0:
            elig = (t - last_e) >= window
            eidx = np.where(elig)[0]
            if len(eidx) >= 2:
                ne = min(n_exp, len(eidx))
                od = np.argpartition(energy[eidx], ne - 1)
                xi = eidx[od[:ne]]
                x[xi] = np.clip(xb + sigma * span * rng.standard_normal((ne, D)), lower, upper)
                v[xi] = 0.0
                energy[xi] = np.median(energy)
                last_e[xi] = t
    return bl


def pso(func, lower, upper, D, N, iterations, seed=0):
    rng = np.random.default_rng(seed)
    w, c1, c2 = 0.729, 1.494, 1.494
    pos = rng.uniform(lower, upper, (N, D))
    vel = rng.uniform(-(upper - lower), (upper - lower), (N, D)) * 0.1
    pb  = pos.copy(); pv = func(pos)
    gp  = pb[np.argmin(pv)].copy(); gv = float(np.min(pv))
    for _ in range(iterations):
        r1 = rng.random((N, D)); r2 = rng.random((N, D))
        vel = w * vel + c1 * r1 * (pb - pos) + c2 * r2 * (gp - pos)
        pos = np.clip(pos + vel, lower, upper)
        vals = func(pos)
        imp = vals < pv
        pb[imp] = pos[imp]; pv[imp] = vals[imp]
        if np.min(pv) < gv:
            gv = float(np.min(pv)); gp = pb[np.argmin(pv)].copy()
    return gv


def run_rime(func, lower, upper, D, N, iterations, seed=0, W=5.0):
    rng  = np.random.default_rng(seed)
    span = upper - lower
    X   = rng.uniform(lower, upper, (N, D))
    fit = func(X)
    bi  = int(np.argmin(fit)); Xb = X[bi].copy(); bf = float(fit[bi])
    for t in range(1, iterations + 1):
        norm_t      = t / iterations
        rime_factor = W * span / (2.0 * np.sqrt(t * iterations))
        r1    = rng.random((N, D))
        Xn    = X + rime_factor * (Xb[None, :] - X) * r1
        r2    = rng.random((N, D))
        Xn    = np.where(r2 < norm_t, Xb[None, :], Xn)
        Xn    = np.clip(Xn, lower, upper)
        fn    = func(Xn)
        imp   = fn < fit
        X     = np.where(imp[:, None], Xn, X); fit = np.where(imp, fn, fit)
        cb    = int(np.argmin(fit))
        if fit[cb] < bf: bf = float(fit[cb]); Xb = X[cb].copy()
    return bf


def run_lshade(func, lower, upper, D, N, iterations, seed=0, H=6):
    rng    = np.random.default_rng(seed)
    N_init = N; N_min = max(4, D // 5); budget = N * iterations
    MF  = np.full(H, 0.5); MCR = np.full(H, 0.5); k = 0
    pop = rng.uniform(lower, upper, (N_init, D)); fit = func(pop)
    bf  = float(np.min(fit)); arc = np.empty((0, D)); ev = N_init
    while ev < budget:
        Nc = len(pop)
        Nn = max(N_min, int(N_min + (N_init - N_min) * (budget - ev) / budget))
        if Nn < Nc:
            kp = np.argpartition(fit, Nn)[:Nn]
            pop = pop[kp]; fit = fit[kp]; Nc = Nn
        ri  = rng.integers(0, H, Nc)
        F   = np.clip(rng.standard_cauchy(Nc) * 0.1 + MF[ri],  0.01, 1.0)
        CR  = np.clip(rng.standard_normal(Nc) * 0.1 + MCR[ri], 0.0,  1.0)
        pn  = max(2, int(0.11 * Nc))
        pi  = np.argpartition(fit, pn)[:pn]
        xpb = pop[pi[rng.integers(pn, size=Nc)]]
        xr1 = pop[(np.arange(Nc) + rng.integers(1, Nc, Nc)) % Nc]
        comb = np.vstack([pop, arc]) if len(arc) > 0 else pop
        xr2  = comb[rng.integers(len(comb), size=Nc)]
        mut  = np.clip(pop + F[:, None] * (xpb - pop) + F[:, None] * (xr1 - xr2), lower, upper)
        jr   = rng.integers(D, size=Nc)
        msk  = rng.random((Nc, D)) < CR[:, None]; msk[np.arange(Nc), jr] = True
        tri  = np.where(msk, mut, pop); ft = func(tri); ev += Nc
        if ev > budget: break
        imp  = ft <= fit
        if imp.any():
            arc = np.vstack([arc, pop[imp]]) if len(arc) > 0 else pop[imp].copy()
            if len(arc) > N_init: arc = arc[rng.choice(len(arc), N_init, replace=False)]
        SF = F[imp]; SCR = CR[imp]; Sd = np.abs(ft[imp] - fit[imp])
        pop = np.where(imp[:, None], tri, pop); fit = np.where(imp, ft, fit)
        if float(np.min(fit)) < bf: bf = float(np.min(fit))
        if len(SF) > 0:
            w = Sd / (Sd.sum() + 1e-12)
            MF[k] = np.dot(w, SF**2) / (np.dot(w, SF) + 1e-12)
            MCR[k] = np.dot(w, SCR); k = (k + 1) % H
    return bf


def run_de(func, lower, upper, D, N, iterations, seed=0):
    budget  = N * iterations
    popsize = 15
    maxiter = max(10, budget // (popsize * D))
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
# 알고리즘 목록 (tracker를 함수로 사용)
# ──────────────────────────────────────────────────────────────────────────────
ALGO_LIST = [
    ("WO2",
     lambda tr, D, N, it, s:
         wo2(tr, tr.grad, tr.func.bounds[0], tr.func.bounds[1], D=D, N=N, iterations=it, seed=s)),
    ("PSO",
     lambda tr, D, N, it, s:
         pso(tr, tr.func.bounds[0], tr.func.bounds[1], D=D, N=N, iterations=it, seed=s)),
    ("RIME",
     lambda tr, D, N, it, s:
         run_rime(tr, tr.func.bounds[0], tr.func.bounds[1], D=D, N=N, iterations=it, seed=s)),
    ("L-SHADE",
     lambda tr, D, N, it, s:
         run_lshade(tr, tr.func.bounds[0], tr.func.bounds[1], D=D, N=N, iterations=it, seed=s)),
    ("DE",
     lambda tr, D, N, it, s:
         run_de(tr, tr.func.bounds[0], tr.func.bounds[1], D=D, N=N, iterations=it, seed=s)),
]

# ──────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ──────────────────────────────────────────────────────────────────────────────
N_RUNS     = 10
D          = 10               # 그래디언트 신뢰도 가장 높은 차원
N_POP      = 30               # 집단 크기
ITERATIONS = 1000             # 총 반복
BUDGET     = N_POP * ITERATIONS

CHECKPOINTS = [0.10, 0.30, 0.50, 1.00]   # 체크포인트 (예산 비율)
EARLY_FRAC  = 0.30                         # "초기 수렴" 기준

# ──────────────────────────────────────────────────────────────────────────────
# 지표 계산 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def aucc(conv_dict, checkpoints, best_final):
    """Normalized Area Under Convergence Curve (낮을수록 좋음).
    각 구간 [c_i, c_{i+1}] 에서 trapezoidal 적분 후 best_final 로 정규화."""
    pts  = sorted(conv_dict.items())   # [(frac, value), ...]
    if len(pts) < 2:
        return float("nan")
    area = 0.0
    denom = max(abs(pts[0][1] - best_final), 1e-12)
    for i in range(len(pts) - 1):
        f0, v0 = pts[i]
        f1, v1 = pts[i + 1]
        area  += (f1 - f0) * (v0 + v1) / 2.0
    return area / denom


def success_rate(values, threshold):
    return 100.0 * np.mean(np.array(values) <= threshold)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실험
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print(f"  WO2-Favorable Evaluation  |  D={D}  |  N={N_POP}  |  iter={ITERATIONS}  |  runs={N_RUNS}")
print(f"  평가 지표: mean@full · mean@30%budget · 성공률(f<target) · AUCC")
print(f"{'='*110}")

all_results = {}   # (fname, aname) → {metric: list_of_values}

for FuncCls in FUNC_CLASSES:
    func_obj = FuncCls(D)
    fname    = func_obj.name

    print(f"\n  [{fname}]  bounds={func_obj.bounds}  target={func_obj.target:.1e}")
    print(f"  {'Algorithm':>10} | {'mean@full':>12} {'std@full':>10} | "
          f"{'mean@30%':>11} | {'성공률%':>7} | {'AUCC':>10} | {'time':>6}")
    print(f"  {'-'*90}")

    for aname, afunc in ALGO_LIST:
        finals, early_vals, auccs, times = [], [], [], []

        for run in range(N_RUNS):
            tr = ConvergenceTracker(func_obj, CHECKPOINTS)
            tr.init_budget(BUDGET)

            t0 = time.time()
            try:
                v = afunc(tr, D, N_POP, ITERATIONS, run)
            except Exception as e:
                v = float("nan")

            elapsed = time.time() - t0

            # 누락된 체크포인트 채우기 (예: DE가 마지막에만 평가)
            for f in sorted(CHECKPOINTS):
                if f not in tr.history:
                    tr.history[f] = tr._best

            finals.append(tr.history.get(1.00, v))
            early_vals.append(tr.history.get(EARLY_FRAC, tr.history.get(1.00, v)))
            auccs.append(aucc(tr.history, CHECKPOINTS, tr._best))
            times.append(elapsed)

        key = (fname, aname)
        all_results[key] = {
            "final": finals,
            "early": early_vals,
            "aucc":  auccs,
        }

        m_final  = np.nanmean(finals)
        s_final  = np.nanstd(finals)
        m_early  = np.nanmean(early_vals)
        sr       = success_rate(finals, func_obj.target)
        m_aucc   = np.nanmean(auccs)

        print(
            f"  {aname:>10} | {m_final:>12.4e} {s_final:>10.4e} | "
            f"{m_early:>11.4e} | {sr:>7.1f} | {m_aucc:>10.3f} | {np.mean(times):>5.2f}s"
        )

# ──────────────────────────────────────────────────────────────────────────────
# 종합 순위 (각 함수 × 각 지표별 WO2 순위)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print("  WO2 vs 경쟁 알고리즘 — 지표별 순위 요약")
print(f"{'='*110}")
print(f"  {'Function':>15} | {'Rank@full':>10} | {'Rank@30%':>9} | {'Rank@SR':>8} | {'Rank@AUCC':>10}")
print(f"  {'-'*70}")

algo_names = [a for a, _ in ALGO_LIST]

def rank_of(fname, metric):
    scores = {a: np.nanmean(all_results[(fname, a)][metric]) for a in algo_names}
    sorted_a = sorted(scores, key=scores.get)
    return sorted_a.index("WO2") + 1

for FuncCls in FUNC_CLASSES:
    fname = FuncCls(D).name
    r_full  = rank_of(fname, "final")
    r_early = rank_of(fname, "early")
    r_aucc  = rank_of(fname, "aucc")

    # 성공률은 높을수록 좋으므로 부호 반전
    sr_scores = {a: -np.mean(np.array(all_results[(fname, a)]["final"]) <= FuncCls(D).target)
                 for a in algo_names}
    r_sr = sorted(sr_scores, key=sr_scores.get).index("WO2") + 1

    mark_full  = " ★" if r_full  == 1 else ""
    mark_early = " ★" if r_early == 1 else ""
    mark_sr    = " ★" if r_sr    == 1 else ""
    mark_aucc  = " ★" if r_aucc  == 1 else ""

    print(f"  {fname:>15} | {r_full:>5}{mark_full:<5} | {r_early:>5}{mark_early:<4} | "
          f"{r_sr:>4}{mark_sr:<4} | {r_aucc:>5}{mark_aucc}")

# 평균 순위
print(f"  {'-'*70}")
for metric_key, label in [("final","mean@full"), ("early","mean@30%"), ("aucc","AUCC")]:
    avg_ranks = {}
    for a in algo_names:
        ranks = []
        for FuncCls in FUNC_CLASSES:
            fname  = FuncCls(D).name
            scores = {aa: np.nanmean(all_results[(fname, aa)][metric_key]) for aa in algo_names}
            sorted_a = sorted(scores, key=scores.get)
            ranks.append(sorted_a.index(a) + 1)
        avg_ranks[a] = np.mean(ranks)
    print(f"\n  [{label}] 평균 순위:")
    for a, r in sorted(avg_ranks.items(), key=lambda x: x[1]):
        bar = "█" * int(5 - r + 1) if r <= 5 else ""
        mark = " ← WO2" if a == "WO2" else ""
        print(f"    {a:>10}: {r:.2f}  {bar}{mark}")

print(f"\n{'='*110}")
print("  주요 해석:")
print("  · mean@30%  : 전체 예산의 30% 시점 성능 — WO2 그래디언트 하강의 초기 수렴 우위 측정")
print("  · 성공률(%) : f(x) < target 도달한 run 비율 — WO2 폭발(explosion) 메커니즘 기여")
print("  · AUCC      : 수렴 곡선 아래 넓이 (정규화) — 전체 수렴 효율성")
print("  · 매끄러운 함수(Sphere/Elliptic)에서 WO2의 그래디언트 기반 하강이 가장 유리함")
print(f"{'='*110}")
