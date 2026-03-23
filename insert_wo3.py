import json

with open('3_최종_코드_정리.ipynb') as f:
    nb = json.load(f)

wo3_code = r'''# ============================================================
# WO3: Weather Optimization v3
# 파라미터 4개: lr, beta, alpha, sigma  (WO2와 동일, 추가 없음)
# WO2 대비 수식 변경 3가지:
#   1. 개인 최적 기억 (Personal Best): attract += alpha_t * (p_best - x)
#   2. 감쇠 alpha: alpha_t = alpha * max(1 - 0.5*t/T, 0.5)
#   3. 차분 벡터 폭발 (Differential Explosion): x_new = x_best + diff + noise
# ============================================================

def wo3(
    func,
    grad_func,
    lower=-5.12,
    upper=5.12,
    D=10,
    N=80,
    iterations=2000,
    # --- 4 algorithm parameters (WO2와 동일) ---
    lr=0.02,        # base learning rate (cosine schedule)
    beta=0.9,       # momentum / EMA factor
    alpha=0.1,      # attraction strength (응축) — 전역 + 개인 각각 적용
    sigma=0.15,     # explosion radius as fraction of bound span (폭발)
    # --- standard setup ---
    seed=0,
    return_history=False,
):
    """
    WO3: Weather Optimization v3

    WO2의 응축(Condensation)·폭발(Explosion) 구조를 유지하면서
    파라미터 추가 없이 3가지 수식을 개선:

    [변경 1 — 개인 최적 기억 (Personal Best)]
      WO2: attract = alpha * (x_best - x)
      WO3: attract = alpha_t * (x_best - x) + alpha_t * (p_best - x)
      → 각 입자가 자신의 역대 최저 손실 위치를 기억 (조기 수렴 방지)

    [변경 2 — 감쇠하는 alpha (Adaptive Alpha)]
      WO2: alpha 고정
      WO3: alpha_t = alpha * max(1 - 0.5*t/T, 0.5)  → 초기 1.0배, 후기 0.5배
      → 초기 빠른 수렴, 후기 정밀 탐색 (탐색/탐색 균형 자동 조절)

    [변경 3 — 차분 벡터 폭발 (Differential Explosion)]
      WO2: x_new = x_best + sigma*span*N(0,I)
      WO3: diff = x[r1] - x[r2]  (활성 입자 중 무작위 2개)
           x_new = x_best + diff + 0.5*sigma*span*N(0,I)
      → 현재 집단의 탐색 방향을 폭발에 활용 (이미 탐색된 영역 재탐색 방지)

    날씨 은유:
      개인 기억 = 각 공기덩어리의 역대 최저기압 위치 보유
      감쇠 alpha = 폭풍 성숙 시 와류 반경 축소 (tightening vortex)
      차분 폭발 = 두 활성 기압계의 바람 전단(wind shear) 방향으로 뇌우 전개
    """
    rng = np.random.default_rng(seed)
    span = upper - lower

    # ---- Initialization ----
    x = rng.uniform(lower, upper, (N, D))
    v = np.zeros((N, D))                    # momentum velocities

    loss = func(x)
    loss_prev = loss.copy()

    # [신규] Personal best 초기화
    p_best = x.copy()                       # 개인 최적 위치
    p_loss = loss.copy()                    # 개인 최적 손실값

    energy = np.ones(N)                     # EMA of |delta_loss|, init=1

    best_idx = np.argmin(loss)
    x_best = x[best_idx].copy()
    best_loss = float(loss[best_idx])
    best_loss_prev = best_loss

    history = []

    for t in range(iterations):

        # === PHASE 1: CONDENSATION (응축) ===

        # Cosine learning rate schedule [WO2와 동일]
        lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * t / iterations))

        # [신규] Adaptive alpha: 초기 alpha → 후기 0.5*alpha (선형 감쇠)
        alpha_t = alpha * max(1.0 - 0.5 * t / iterations, 0.5)

        # Gradients + norm clipping
        g = grad_func(x)                                        # (N, D)
        g_norm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-8
        g_clipped = g / np.maximum(g_norm, 1.0)                # clip to 1

        # Momentum update (heavy ball) [WO2와 동일]
        v = beta * v - lr_t * g_clipped                        # (N, D)

        # [신규] 전역 인력 + 개인 인력 (personal best 추가)
        attract = alpha_t * (x_best - x) + alpha_t * (p_best - x)  # (N, D)

        # Update positions
        x = np.clip(x + v + attract, lower, upper)

        # === ENERGY TRACKING ===
        loss = func(x)

        # [신규] Personal best 갱신 (탐욕적)
        improved = loss < p_loss
        p_best[improved] = x[improved]
        p_loss[improved] = loss[improved]

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

        # Global stagnation safety [WO2와 동일]
        window = max(1, int(1.0 / (1.0 - beta)))
        if t > 0 and (t % window) == 0 and best_loss >= best_loss_prev:
            n_force = max(1, int(sigma * N))
            force_idx = rng.choice(N, size=n_force, replace=False)
            stagnated[force_idx] = True
        best_loss_prev = best_loss

        n_explode = int(stagnated.sum())
        if n_explode > 0:
            # [신규] 차분 벡터: 활성 입자 중 무작위 2개에서 방향 추출
            active_idx = np.where(~stagnated)[0]
            if len(active_idx) >= 2:
                r1, r2 = rng.choice(active_idx, size=2, replace=False)
                diff = x[r1] - x[r2]                           # (D,) 방향 벡터
            else:
                diff = np.zeros(D)

            noise = rng.standard_normal((n_explode, D))
            x_new = np.clip(
                x_best + diff + 0.5 * sigma * span * noise,    # [신규: diff 추가]
                lower, upper
            )
            x[stagnated] = x_new
            v[stagnated] = 0.0
            # [신규] 폭발 후 개인 최적 리셋 (새 위치에서 재출발)
            p_best[stagnated] = x_new
            p_loss[stagnated] = loss[stagnated]
            energy[stagnated] = e_mean                         # reset to mean

        if return_history:
            history.append({
                "iter":      t,
                "best_loss": best_loss,
                "n_explode": n_explode,
                "e_mean":    float(e_mean),
                "lr_t":      float(lr_t),
                "alpha_t":   float(alpha_t),
            })

    if return_history:
        return best_loss, history
    return best_loss
'''

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 코드 L — WO3 (Weather Optimization v3)\n",
            "\n",
            "파라미터 4개 (`lr`, `beta`, `alpha`, `sigma`) — WO2와 동일, 추가 없음.\n",
            "\n",
            "WO2 대비 수식 3가지 변경:\n",
            "1. **개인 최적 기억**: `attract += alpha_t * (p_best - x)`\n",
            "2. **감쇠 alpha**: `alpha_t = alpha * max(1 - 0.5*t/T, 0.5)`\n",
            "3. **차분 벡터 폭발**: `x_new = x_best + (x[r1]-x[r2]) + 0.5*σ*span*N(0,I)`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [wo3_code]
    },
]

# WO2 code is at cell 9; insert WO3 right after (position 10)
insert_at = 10
cells = nb['cells']
nb['cells'] = cells[:insert_at] + new_cells + cells[insert_at:]

with open('3_최종_코드_정리.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
print(f"Inserted {len(new_cells)} cells at position {insert_at}")
print(f"WO3 code cell is now at index {insert_at + 1}")
print(f"Former cells 43-46 (CE/RIME/L-SHADE/extras) are now at {insert_at+1+33}-{insert_at+1+36}")
