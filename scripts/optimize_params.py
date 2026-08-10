#!/usr/bin/env python3
"""
参数优化：在 6 组极端池上网格搜索最优 (c, decay, alpha)。

输出: minimax 最优参数 ← 最差池上表现最好的那组

用法: python3 scripts/optimize_params.py
"""

import sys, math, random, json
from itertools import product
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bandit import UCBBandit


# ═══════════════════════════════════════════
# 模拟模型（带真实错误）
# ═══════════════════════════════════════════
class MockModel:
    def __init__(self, name, quality, tokens=3000, max_tokens=65536,
                 quota_exhausted=False):
        self.name = name; self.tokens = tokens
        self.max_tokens = max_tokens
        self.quota_exhausted = quota_exhausted
        self.quality = quality

    def get_q(self, tt):
        return self.quality[tt] if isinstance(self.quality, dict) else self.quality

    def call(self, tt, request_max_tokens=65536):
        if self.quota_exhausted:
            return False, 0, 0, "403"
        if request_max_tokens > self.max_tokens:
            return False, 0, 0, "400"
        q = self.get_q(tt)
        r = q + random.gauss(0, 0.02)
        r = max(0.01, min(1.0, r))
        t = self.tokens + random.randint(-300, 300)
        return True, r, t, None


# ═══════════════════════════════════════════
# 6 组极端测试池
# ═══════════════════════════════════════════
POOLS = {
    "同质池": [
        MockModel("A", 0.90, 3000), MockModel("B", 0.89, 3100),
        MockModel("C", 0.91, 2900), MockModel("D", 0.88, 3200),
        MockModel("E", 0.90, 3000),
    ],
    "异构池": [
        MockModel("best",  0.95, 3000), MockModel("mid-A", 0.60, 2000),
        MockModel("mid-B", 0.55, 2500), MockModel("weak", 0.40, 1500),
    ],
    "翻转池_A降B升": [
        MockModel("A", 0.90, 3000), MockModel("B", 0.60, 2000),
        MockModel("C", 0.70, 2500),
    ],
    "冷启池_3+2": [
        MockModel("A", 0.90, 3000), MockModel("B", 0.80, 2000),
        MockModel("C", 0.70, 2500),
        MockModel("new-best", 0.95, 2800),  # 第 100 轮加入
        MockModel("new-weak", 0.50, 1500),
    ],
    "故障池": [
        MockModel("good-A", 0.92, 3000), MockModel("good-B", 0.88, 2500),
        MockModel("dead-403", 0.99, 1000, quota_exhausted=True),
        MockModel("small-400", 0.93, 2000, max_tokens=4096),
    ],
    "多任务池_coding强vs_chat强": [
        MockModel("coder", {"coding": 0.95, "chat": 0.30, "other": 0.60}, 4000),
        MockModel("chatter", {"coding": 0.30, "chat": 0.95, "other": 0.60}, 2500),
        MockModel("balanced", {"coding": 0.65, "chat": 0.65, "other": 0.65}, 3000),
    ],
}

POOL_TASKS = {
    "同质池":  ["chat"] * 150 + ["coding"] * 50,
    "异构池":  ["chat"] * 200,
    "翻转池_A降B升": ["chat"] * 100 + ["coding"] * 100,  # 中途变任务
    "冷启池_3+2": ["chat"] * 70 + ["coding"] * 70 + ["chat"] * 60,
    "故障池":  ["chat"] * 80 + ["coding"] * 80,
    "多任务池_coding强vs_chat强": ["coding" if i % 2 == 0 else "chat" for i in range(200)],
}

# 翻转池：第 100 轮 A 从 0.90 降为 0.40，B 升到 0.90
POOL_CHANGES = {
    "翻转池_A降B升": {100: {"A": 0.40, "B": 0.90}},
}

# 冷启池：第 100 轮加入 2 个新模型（index 3,4）
POOL_ADD_AT = {
    "冷启池_3+2": 100,
}


# ═══════════════════════════════════════════
# 单次仿真
# ═══════════════════════════════════════════
def simulate(pool_name, params):
    """返回 cumulative_regret, optimal_rate, 选模分布"""
    models_raw = POOLS[pool_name]
    models = [MockModel(m.name, m.quality, m.tokens, m.max_tokens,
                        m.quota_exhausted) for m in models_raw]
    task_types = POOL_TASKS[pool_name]
    rounds = len(task_types)
    changes = POOL_CHANGES.get(pool_name, {})
    add_at = POOL_ADD_AT.get(pool_name, None)

    bandit = UCBBandit(c=params["c"], alpha=params["alpha"],
                       base_reward=100.0)
    # 临时覆盖 DECAY 常量
    import bandit as bm
    bm.DECAY = params.get("decay", 0.95)

    active_models = models[:3] if add_at else models[:]
    if add_at:
        hidden = models[3:]

    exhausted = set()
    selections = []
    rewards = []

    for t in range(rounds):
        # 质量变化
        if t in changes:
            for name, new_q in changes[t].items():
                for m in models:
                    if m.name == name:
                        m.quality = new_q

        # 新模型加入
        if add_at and t == add_at:
            active_models.extend(hidden)

        tt = task_types[t]
        avail = [m for m in active_models if m.name not in exhausted]
        candidates = [{"model": m.name, "base_url": "", "api_key": ""}
                      for m in avail]

        selected = bandit.select(candidates, task_type=tt)
        model = next(m for m in avail if m.name == selected["model"])

        success, reward, tokens, err = model.call(tt)

        if err == "403":
            exhausted.add(model.name)
        elif err == "400":
            bandit.update(model.name, False, 0, tt)
        elif success:
            norm_r = (max(100.0 - params["alpha"] * tokens, 0.1)) / 100.0
            bandit.update(model.name, True, int(tokens), tt)

        selections.append(selected["model"])
        rewards.append(reward if success else 0.0)

    # 指标
    regret = 0
    optimal_count = 0
    for t in range(rounds):
        tt = task_types[t]
        best_name = max(active_models, key=lambda m: m.get_q(tt)).name
        if selections[t] == best_name:
            optimal_count += 1
        regret += rewards[t]

    return regret, optimal_count / rounds * 100, selections


# ═══════════════════════════════════════════
# 网格搜索
# ═══════════════════════════════════════════
def grid_search():
    c_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    decay_vals = [0.85, 0.90, 0.95, 0.97, 0.99]
    alpha_vals = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    # 先固定 decay=0.95, alpha=5e-4, 扫 c
    print("=" * 70)
    print("  Phase 1: 固定 decay=0.95 alpha=5e-4，扫描 c")
    print("=" * 70)
    print(f"  {'c':<8} {'平均Regret':>10} {'最佳率':>7} {'最差池':>12}  {'最差Regret':>10}")
    print(f"  {'-'*55}")

    c_results = {}
    for c in c_vals:
        params = {"c": c, "decay": 0.95, "alpha": 5e-4}
        pool_regrets = {}
        pool_rates = {}
        for pool_name in POOLS:
            regret, rate, _ = simulate(pool_name, params)
            pool_regrets[pool_name] = regret
            pool_rates[pool_name] = rate

        avg_regret = sum(pool_regrets.values()) / len(pool_regrets)
        worst_pool = max(pool_regrets, key=pool_regrets.get)
        worst_regret = pool_regrets[worst_pool]
        avg_rate = sum(pool_rates.values()) / len(pool_rates)

        marker = ""
        c_results[c] = (avg_regret, worst_regret, avg_rate)
        print(f"  {c:<8} {avg_regret:>10.1f} {avg_rate:>6.1f}% "
              f"{worst_pool:>12}  {worst_regret:>10.1f}  {marker}")

    # 找 minimax 最优 c
    best_c = min(c_results, key=lambda c: c_results[c][1])
    print(f"\n  → minimax 最优 c = {best_c} (最差池 Regret 最低)")

    # Phase 2: 用最优 c 扫 decay
    print(f"\n{'='*70}")
    print(f"  Phase 2: 固定 c={best_c} alpha=5e-4，扫描 decay")
    print(f"{'='*70}")
    print(f"  {'decay':<8} {'平均Regret':>10} {'最佳率':>7} {'最差池':>12}  {'最差Regret':>10}")

    decay_results = {}
    for decay in decay_vals:
        params = {"c": best_c, "decay": decay, "alpha": 5e-4}
        pool_regrets = {}
        pool_rates = {}
        for pool_name in POOLS:
            regret, rate, _ = simulate(pool_name, params)
            pool_regrets[pool_name] = regret
            pool_rates[pool_name] = rate

        avg_regret = sum(pool_regrets.values()) / len(pool_regrets)
        worst_pool = max(pool_regrets, key=pool_regrets.get)
        worst_regret = pool_regrets[worst_pool]
        avg_rate = sum(pool_rates.values()) / len(pool_rates)

        decay_results[decay] = (avg_regret, worst_regret, avg_rate)
        print(f"  {decay:<8} {avg_regret:>10.1f} {avg_rate:>6.1f}% "
              f"{worst_pool:>12}  {worst_regret:>10.1f}")

    best_decay = min(decay_results, key=lambda d: decay_results[d][1])
    print(f"\n  → minimax 最优 decay = {best_decay}")

    # Phase 3: 用最优 c, decay 扫 alpha
    print(f"\n{'='*70}")
    print(f"  Phase 3: 固定 c={best_c} decay={best_decay}，扫描 alpha")
    print(f"{'='*70}")
    print(f"  {'alpha':<10} {'平均Regret':>10} {'最佳率':>7} {'最差池':>12}  {'最差Regret':>10}")

    alpha_results = {}
    for alpha in alpha_vals:
        params = {"c": best_c, "decay": best_decay, "alpha": alpha}
        pool_regrets = {}
        pool_rates = {}
        for pool_name in POOLS:
            regret, rate, _ = simulate(pool_name, params)
            pool_regrets[pool_name] = regret
            pool_rates[pool_name] = rate

        avg_regret = sum(pool_regrets.values()) / len(pool_regrets)
        worst_pool = max(pool_regrets, key=pool_regrets.get)
        worst_regret = pool_regrets[worst_pool]
        avg_rate = sum(pool_rates.values()) / len(pool_rates)

        alpha_results[alpha] = (avg_regret, worst_regret, avg_rate)
        print(f"  {alpha:<10.0e} {avg_regret:>10.1f} {avg_rate:>6.1f}% "
              f"{worst_pool:>12}  {worst_regret:>10.1f}")

    best_alpha = min(alpha_results, key=lambda a: alpha_results[a][1])
    print(f"\n  → minimax 最优 alpha = {best_alpha:.0e}")

    # ── 最终推荐 ──
    print(f"\n{'='*70}")
    print(f"  minimax 最优参数")
    print(f"{'='*70}")
    print(f"  c     = {best_c}")
    print(f"  decay = {best_decay}")
    print(f"  alpha = {best_alpha:.0e}")
    print(f"\n  更新方法:")
    print(f"    bandit.py 第27行: DECAY = {best_decay}")
    print(f"    bandit.py 第79行: UCBBandit(c={best_c}, alpha={best_alpha:.0e})")
    print(f"    或在 config.yaml 的 bandit 段设置 ucb_c: {best_c}, alpha: {best_alpha:.0e}")


if __name__ == "__main__":
    random.seed(42)
    grid_search()
