#!/usr/bin/env python3
"""
MC 参数搜索：自动调优 smart-router UCB bandit 超参数。

搜索空间 5 维：(c, alpha, decay, prior_pulls, prior_blend_min)
场景：固定池 + 中途换池（含新模型冷启动 + 老模型下线）
筛选：竞速(sequential halving) + 成功率约束 + 收敛速度排名
精调：Nelder-Mead 确定性重放
验证：多 scenario 交叉

设计原则：
- 用所有真实模型组成一个混合池（10 模型、token 跨度 1k-21k），
  给搜索足够的区分度。搜出来的参数适用于实际两个池。
- 部分模型注入随机失败率（5-10%），让成功率约束有意义——
  否则 100% 成功场景下所有参数都能通过过滤。
- 换池场景：模拟余额耗尽（模型下线）+ 新模型加入（冷启动），
  检验 decay 和 prior 参数是否够快恢复。

用法:
  cd ~/.hermes/plugins/smart-router
  python3 scripts/monte_carlo_search.py
"""

import sys, os, math, random, copy, json, time, hashlib
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ── 导入真实 bandit 模块 ──────────────────────────────────────────
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))
import bandit


# ═══════════════════════════════════════════════════════════════════
# 0. 模型数据
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelProfile:
    name: str
    avg_tokens: float
    success_rate: float = 1.0     # 真实成功率


def load_real_models() -> List[ModelProfile]:
    """从 bandit JSON 提取所有模型，注入合理的随机失败率。"""
    models: Dict[str, dict] = {}
    for pool_file in (_PLUGIN_DIR / "data").glob("bandit_*.json"):
        data = json.loads(pool_file.read_text())
        for model_name, stats in data.get("stats", {}).items():
            overall = stats.get("overall", {})
            pulls = overall.get("pulls", 0)
            if pulls <= 0:
                continue
            models[model_name] = {
                "name": model_name,
                "avg_tokens": overall["total_tokens"] / pulls,
            }

    # 注入随机失败率（模拟真实 MaaS 偶尔的 429/超时）
    failure_rates = {
        # 少数模型有不稳定因素
        # 用名字哈希保证确定性
    }
    profiles = []
    for i, (name, m) in enumerate(sorted(models.items())):
        name_hash = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
        fr = 0.05 + (name_hash % 6) / 100 if name_hash % 3 == 0 else 0.0
        profiles.append(ModelProfile(
            name=m["name"],
            avg_tokens=m["avg_tokens"],
            success_rate=1.0 - fr,
        ))
    return profiles


# ═══════════════════════════════════════════════════════════════════
# 1. 参数空间
# ═══════════════════════════════════════════════════════════════════

PARAM_BOUNDS = {
    "c":                (10**-1.3, 10**0.3),     # [0.05, 2.0]
    "alpha":            (10**-5.0, 10**-1.0),     # [1e-5, 0.1]
    "decay":            (0.90, 0.999),
    "prior_pulls":      (1, 10),
    "prior_blend_min":  (2, 20),
}

SUCCESS_RATE_FLOOR = 0.95
MAX_CONVERGENCE_ROUND = 150
CONVERGENCE_WINDOW = 20
CONVERGENCE_THRESHOLD = 0.7

# task_type 列表
ALL_TASK_TYPES = ["chat", "coding", "reasoning", "writing", "analysis",
                  "translation", "other"]


def sample_params(n: int) -> List[Dict]:
    """log 尺度采样（c/alpha 对数均匀，其余线性均匀）。"""
    samples = []
    for _ in range(n):
        b = PARAM_BOUNDS
        samples.append({
            "c": 10 ** random.uniform(math.log10(b["c"][0]),
                                      math.log10(b["c"][1])),
            "alpha": 10 ** random.uniform(math.log10(b["alpha"][0]),
                                          math.log10(b["alpha"][1])),
            "decay": random.uniform(*b["decay"]),
            "prior_pulls": round(random.uniform(*b["prior_pulls"])),
            "prior_blend_min": round(random.uniform(*b["prior_blend_min"])),
        })
    return samples


# ═══════════════════════════════════════════════════════════════════
# 2. 场景生成
# ═══════════════════════════════════════════════════════════════════

# 从真实 classifications.jsonl 提取 task_type 权重（一次性加载）
_TASK_TYPE_WEIGHTS = None


def _load_task_weights() -> List[float]:
    """从 classifications.jsonl 提取 task_type 分布，返回权重列表。
    无数据时回退到均匀分布。
    """
    global _TASK_TYPE_WEIGHTS
    if _TASK_TYPE_WEIGHTS is not None:
        return _TASK_TYPE_WEIGHTS

    p = _PLUGIN_DIR / "data" / "classifications.jsonl"
    if not p.exists():
        _TASK_TYPE_WEIGHTS = [1.0 / len(ALL_TASK_TYPES)] * len(ALL_TASK_TYPES)
        return _TASK_TYPE_WEIGHTS

    from collections import Counter
    import json
    cnt = Counter()
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
                tt = r.get("task_type", "other")
                if tt in ALL_TASK_TYPES:
                    cnt[tt] += 1
            except Exception:
                pass

    total = sum(cnt.values())
    if total == 0:
        _TASK_TYPE_WEIGHTS = [1.0 / len(ALL_TASK_TYPES)] * len(ALL_TASK_TYPES)
    else:
        _TASK_TYPE_WEIGHTS = [cnt.get(tt, 0) / total for tt in ALL_TASK_TYPES]

    return _TASK_TYPE_WEIGHTS


def _gen_task_sequence(rounds: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    weights = _load_task_weights()
    return rng.choices(ALL_TASK_TYPES, weights=weights, k=rounds)


def scenario_fixed_pool(models: List[ModelProfile],
                        rounds: int, seed: int):
    """固定池：N 个模型从头到尾不变。"""
    tasks = _gen_task_sequence(rounds, seed)
    return tasks, [models for _ in range(rounds)]


def scenario_pool_change(models: List[ModelProfile],
                         rounds: int, seed: int):
    """中途换池：前 1/3 轮用初始池，之后替换 2 个旧模型 + 加入 2 个新模型。

    模拟真实场景：余额耗尽（旧模型下线）+ 新模型冷启动。
    新模型 token 量级与旧池不同，检验 decay/prior 恢复速度。
    """
    rng = random.Random(seed)
    tasks = _gen_task_sequence(rounds, seed + 1000)  # 不同 seed 避免与固定池完全相同

    swap_round = rounds // 3

    # 选 2 个要淘汰的旧模型（成功率最低的）
    sorted_by_fr = sorted(models, key=lambda m: m.success_rate)
    to_remove = {sorted_by_fr[0].name, sorted_by_fr[1].name}

    # 构建新模型
    new_models = [
        ModelProfile(name="new-cheap",
                     avg_tokens=rng.uniform(1500, 5000),
                     success_rate=rng.uniform(0.93, 0.98)),
        ModelProfile(name="new-expensive",
                     avg_tokens=rng.uniform(25000, 40000),
                     success_rate=rng.uniform(0.90, 0.96)),
    ]

    old_pool = models
    new_pool = [m for m in models if m.name not in to_remove] + new_models

    pool_snapshots = []
    for i in range(rounds):
        if i < swap_round:
            pool_snapshots.append(old_pool)
        else:
            pool_snapshots.append(new_pool)

    return tasks, pool_snapshots


# ═══════════════════════════════════════════════════════════════════
# 3. 单次仿真
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    total_tokens: float                 # 所有成功调用的 token 累计
    success_count: int
    failure_count: int
    selection_counts: Counter = field(default_factory=Counter)
    convergence_round: Optional[int] = None   # None = 未收敛
    tokens_to_convergence: float = 0.0        # 收敛前的 token 累计
    rounds: int = 0


def _patch_constants(params: Dict) -> Dict:
    old = {
        "DECAY": bandit.DECAY,
        "PRIOR_PULLS": bandit.PRIOR_PULLS,
        "PRIOR_BLEND_MIN": bandit.PRIOR_BLEND_MIN,
    }
    bandit.DECAY = params["decay"]
    bandit.PRIOR_PULLS = params["prior_pulls"]
    bandit.PRIOR_BLEND_MIN = params["prior_blend_min"]
    return old


def _restore_constants(old: Dict):
    bandit.DECAY = old["DECAY"]
    bandit.PRIOR_PULLS = old["PRIOR_PULLS"]
    bandit.PRIOR_BLEND_MIN = old["PRIOR_BLEND_MIN"]


def _find_best_model(current_models: List[ModelProfile]) -> Optional[str]:
    """在当前候选池里找 bandit 视角的最优模型。

    目标：最小化 avg_tokens / success_rate（期望每成功调用的 token 成本）。
    这与 UCB 的 avg_reward = (100 - alpha*avg_tokens)/100 * success_rate 一致——
    便宜且稳定的模型 = 高 reward。
    """
    best = None
    best_cost = float("inf")
    for m in current_models:
        if m.success_rate <= 0:
            continue
        cost_per_success = m.avg_tokens / m.success_rate
        if cost_per_success < best_cost:
            best_cost = cost_per_success
            best = m.name
    return best


def _simulate_one(params: Dict, models: List[ModelProfile],
                  tasks: list, pool_snapshots: list) -> SimResult:
    """单次仿真：固定 seed → 跑完任务序列 → 返回统计。"""
    bb = bandit.UCBBandit(c=params["c"], alpha=params["alpha"],
                          base_reward=100.0)

    model_map = {m.name: m for m in models}
    # 合成模型也加入 map
    all_models_seen = dict(model_map)

    total_tokens = 0.0
    success_count = 0
    failure_count = 0
    selection_counts = Counter()
    recent_selections: List[str] = []
    convergence_round = None
    tokens_to_convergence = 0.0

    for i, tt in enumerate(tasks):
        pool_models = pool_snapshots[i]
        # 更新 model_map（换池后可能有新模型）
        for m in pool_models:
            if m.name not in all_models_seen:
                all_models_seen[m.name] = m

        candidates = [{"model": m.name, "base_url": "", "api_key": ""}
                      for m in pool_models]

        selected = bb.select(candidates, task_type=tt)
        if not selected:
            failure_count += 1
            continue

        sel_name = selected["model"]
        selection_counts[sel_name] += 1
        recent_selections.append(sel_name)
        if len(recent_selections) > CONVERGENCE_WINDOW:
            recent_selections.pop(0)

        # 取模型数据
        model = all_models_seen.get(sel_name)
        if model is None:
            # 不应发生，但兜底
            tokens = 10000
            sr = 1.0
        else:
            base_tokens = model.avg_tokens
            tokens = max(10, int(base_tokens + random.gauss(0, base_tokens * 0.05)))
            sr = model.success_rate

        success = random.random() < sr

        bb.update(sel_name, success, tokens, tt)

        if success:
            success_count += 1
            total_tokens += tokens
        else:
            failure_count += 1

        # 收敛检测：对当前池的最优模型，看 20 轮窗口内是否 >70% 选它
        if convergence_round is None and len(recent_selections) == CONVERGENCE_WINDOW:
            best = _find_best_model(pool_models)
            if best:
                best_count = sum(1 for s in recent_selections if s == best)
                if best_count / CONVERGENCE_WINDOW >= CONVERGENCE_THRESHOLD:
                    convergence_round = i + 1
                    tokens_to_convergence = total_tokens

    if convergence_round is None:
        tokens_to_convergence = total_tokens

    return SimResult(
        total_tokens=total_tokens,
        success_count=success_count,
        failure_count=failure_count,
        selection_counts=selection_counts,
        convergence_round=convergence_round,
        tokens_to_convergence=tokens_to_convergence,
        rounds=len(tasks),
    )


def evaluate_params(params: Dict, models: List[ModelProfile],
                    rounds: int, seeds: List[int],
                    scenario_fn) -> List[SimResult]:
    """对一组参数跑多个 seed，返回所有结果。"""
    results = []
    old_const = _patch_constants(params)
    try:
        for seed in seeds:
            tasks, snapshots = scenario_fn(models, rounds, seed)
            r = _simulate_one(params, models, tasks, snapshots)
            results.append(r)
    finally:
        _restore_constants(old_const)
    return results


# ═══════════════════════════════════════════════════════════════════
# 4. 竞速筛选（Sequential Halving）
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CandidateEval:
    params: Dict
    results_fixed: List[SimResult] = field(default_factory=list)
    results_change: List[SimResult] = field(default_factory=list)
    total_seeds_run: int = 0

    @property
    def _all_results(self) -> List[SimResult]:
        return self.results_fixed + self.results_change

    @property
    def mean_success_rate(self) -> float:
        total_succ = sum(r.success_count for r in self._all_results)
        total_rounds = sum(r.rounds for r in self._all_results)
        return total_succ / total_rounds if total_rounds > 0 else 0.0

    @property
    def mean_conv_round(self) -> float:
        """平均收敛轮次（未收敛的用 rounds 代替）。"""
        vals = [r.convergence_round or r.rounds for r in self._all_results]
        return sum(vals) / len(vals) if vals else float("inf")

    @property
    def mean_ttc(self) -> float:
        """平均 tokens-to-convergence。"""
        vals = [r.tokens_to_convergence for r in self._all_results]
        return sum(vals) / len(vals) if vals else float("inf")

    @property
    def lcb_ttc(self) -> float:
        """置信下界：mean - 1.5*SE。越小越好，且惩罚不确定性。"""
        vals = [r.tokens_to_convergence for r in self._all_results]
        n = len(vals)
        if n < 2:
            return vals[0] if vals else float("inf")
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n)
        return max(0.0, mean - 1.5 * se)


def _is_feasible(c: CandidateEval) -> bool:
    if c.mean_success_rate < SUCCESS_RATE_FLOOR:
        return False
    if c.mean_conv_round > MAX_CONVERGENCE_ROUND:
        return False
    return True


def racing_search(params_list: List[Dict],
                  models: List[ModelProfile],
                  rounds: int,
                  budget_per_param: int,
                  seeds: List[int]) -> List[CandidateEval]:
    """Sequential halving：每轮砍掉后 50%，幸存者 seed 数翻倍。"""
    candidates = [CandidateEval(params=p) for p in params_list]
    survivors = list(candidates)
    seeds_per_round = 2
    seed_idx = 0
    used_seeds = 0

    while len(survivors) > 5 and used_seeds < budget_per_param * 2:
        this_round_seeds = min(seeds_per_round, len(seeds) - seed_idx)
        if this_round_seeds <= 0:
            break
        batch = seeds[seed_idx:seed_idx + this_round_seeds]
        seed_idx += this_round_seeds

        for c in survivors:
            c.results_fixed.extend(
                evaluate_params(c.params, models, rounds, batch,
                                scenario_fixed_pool))
            c.results_change.extend(
                evaluate_params(c.params, models, rounds, batch,
                                scenario_pool_change))
            c.total_seeds_run += this_round_seeds * 2

        used_seeds += this_round_seeds * len(survivors) * 2

        # 过滤不可行点
        pre_count = len(survivors)
        survivors = [c for c in survivors if _is_feasible(c)]
        if len(survivors) < 5 and pre_count >= 5:
            # 约束太紧 → 保留成功率最高的前 10 个
            print(f"  [racing] ⚠ 仅 {len(survivors)} 个候选通过约束 "
                  f"(共 {pre_count})，保留 top-10 按成功率")
            all_evaluated = [c for c in candidates if c.total_seeds_run > 0]
            all_evaluated.sort(key=lambda c: -c.mean_success_rate)
            survivors = all_evaluated[:10]
        if not survivors:
            break

        # 按 LCB + 收敛速度综合排序
        survivors.sort(key=lambda c: (
            c.mean_conv_round,        # 先比收敛速度
            c.lcb_ttc,               # 再比 token 下界
        ))

        cut_at = max(len(survivors) // 2, 5)
        survivors = survivors[:cut_at]

        seeds_per_round *= 2

        best = survivors[0]
        print(f"  [racing] survivors={len(survivors)}, "
              f"seeds_per={this_round_seeds}, "
              f"best_LCB={best.lcb_ttc:,.0f}, "
              f"best_conv={best.mean_conv_round:.0f}, "
              f"best_rate={best.mean_success_rate*100:.1f}%")

    return survivors


# ═══════════════════════════════════════════════════════════════════
# 5. Nelder-Mead 局部精调
# ═══════════════════════════════════════════════════════════════════

def _pack(params: Dict) -> List[float]:
    """映射到无界空间。"""
    return [
        math.log10(params["c"]),
        math.log10(params["alpha"]),
        math.log((params["decay"] - 0.90) / (0.999 - params["decay"])),
        float(params["prior_pulls"]),
        float(params["prior_blend_min"]),
    ]


def _unpack(x: List[float]) -> Dict:
    b = PARAM_BOUNDS
    decay_raw = 1.0 / (1.0 + math.exp(-x[2]))
    return {
        "c": max(b["c"][0], min(b["c"][1], 10 ** x[0])),
        "alpha": max(b["alpha"][0], min(b["alpha"][1], 10 ** x[1])),
        "decay": max(b["decay"][0], min(b["decay"][1],
                                        0.90 + decay_raw * 0.099)),
        "prior_pulls": max(1, min(10, round(x[3]))),
        "prior_blend_min": max(2, min(20, round(x[4]))),
    }


def _det_eval(params: Dict, models: List[ModelProfile],
              rounds: int, seed: int, scenario_fn) -> float:
    """确定性评估：返回 avg tokens_to_convergence（跨两个场景）。"""
    old_const = _patch_constants(params)
    try:
        tasks_f, snap_f = scenario_fixed_pool(models, rounds, seed)
        tasks_c, snap_c = scenario_pool_change(models, rounds, seed + 1)
        r1 = _simulate_one(params, models, tasks_f, snap_f)
        r2 = _simulate_one(params, models, tasks_c, snap_c)
    finally:
        _restore_constants(old_const)
    return (r1.tokens_to_convergence + r2.tokens_to_convergence) / 2


def _nm_iteration(simplex: List[Tuple[List[float], float]],
                  models: List[ModelProfile], rounds: int, seed: int,
                  scenario_fn) -> List[Tuple[List[float], float]]:
    """一次 Nelder-Mead 迭代。"""
    simplex.sort(key=lambda p: p[1])
    n = len(simplex) - 1

    # 去掉最差点的重心
    centroid = [0.0] * n
    for i in range(n):
        for j in range(n):
            centroid[j] += simplex[i][0][j]
    centroid = [c / n for c in centroid]

    worst_val = simplex[-1][1]

    # 反射
    alpha_nm = 1.0
    reflected = [centroid[j] + alpha_nm * (centroid[j] - simplex[-1][0][j])
                 for j in range(n)]
    f_ref = _det_eval(_unpack(reflected), models, rounds, seed, scenario_fn)

    if f_ref < simplex[0][1]:
        # 扩展
        expanded = [centroid[j] + 2.0 * (reflected[j] - centroid[j])
                    for j in range(n)]
        f_exp = _det_eval(_unpack(expanded), models, rounds, seed, scenario_fn)
        simplex[-1] = (expanded, f_exp) if f_exp < f_ref else (reflected, f_ref)
    elif f_ref < simplex[-2][1]:
        simplex[-1] = (reflected, f_ref)
    else:
        if f_ref < worst_val:
            # 外部收缩
            contracted = [centroid[j] + 0.5 * (reflected[j] - centroid[j])
                          for j in range(n)]
            f_cont = _det_eval(_unpack(contracted), models, rounds, seed,
                               scenario_fn)
            if f_cont < f_ref:
                simplex[-1] = (contracted, f_cont)
            else:
                _shrink(simplex, models, rounds, seed, scenario_fn)
        else:
            # 内部收缩
            contracted = [centroid[j] - 0.5 * (reflected[j] - centroid[j])
                          for j in range(n)]
            f_cont = _det_eval(_unpack(contracted), models, rounds, seed,
                               scenario_fn)
            if f_cont < worst_val:
                simplex[-1] = (contracted, f_cont)
            else:
                _shrink(simplex, models, rounds, seed, scenario_fn)
    return simplex


def _shrink(simplex, models, rounds, seed, scenario_fn):
    """全收缩：除最优点外全部向最优点靠拢。"""
    best = simplex[0][0]
    for i in range(1, len(simplex)):
        shrink = [best[j] + 0.5 * (simplex[i][0][j] - best[j])
                  for j in range(len(best))]
        f_shrink = _det_eval(_unpack(shrink), models, rounds, seed,
                             scenario_fn)
        simplex[i] = (shrink, f_shrink)


def nm_refine(start_params: Dict, models: List[ModelProfile],
              rounds: int, seed: int, max_iter: int = 30,
              scenario_fn=scenario_fixed_pool) -> Tuple[Dict, float]:
    """Nelder-Mead 局部精调。"""
    x0 = _pack(start_params)
    n = len(x0)

    simplex = []
    f0 = _det_eval(start_params, models, rounds, seed, scenario_fn)
    simplex.append((x0[:], f0))

    for j in range(n):
        xj = x0[:]
        xj[j] *= 1.1 if xj[j] != 0 else 0.1
        fj = _det_eval(_unpack(xj), models, rounds, seed, scenario_fn)
        simplex.append((xj, fj))

    prev_best = simplex[0][1]
    for _ in range(max_iter):
        simplex = _nm_iteration(simplex, models, rounds, seed, scenario_fn)
        simplex.sort(key=lambda p: p[1])
        new_best = simplex[0][1]
        if abs(prev_best - new_best) < 1e-6 * max(abs(new_best), 1.0):
            break
        prev_best = new_best

    simplex.sort(key=lambda p: p[1])
    best_x, best_f = simplex[0]
    return _unpack(best_x), best_f


# ═══════════════════════════════════════════════════════════════════
# 6. 交叉验证
# ═══════════════════════════════════════════════════════════════════

def cross_validate(candidates: List[Dict], models: List[ModelProfile],
                   rounds: int, seeds: List[int]) -> List[dict]:
    """对最终候选在多个 scenario 上评估。"""
    scenario_fns = {
        "fixed": scenario_fixed_pool,
        "pool_change": scenario_pool_change,
    }
    results = []
    for params in candidates:
        row = {"params": params, "scenarios": {}}
        all_ttc = []
        for sname, sfn in scenario_fns.items():
            s_results = evaluate_params(params, models, rounds, seeds, sfn)
            ttc_vals = [r.tokens_to_convergence for r in s_results]
            conv_vals = [r.convergence_round or rounds for r in s_results]
            succ_vals = [r.success_count / r.rounds * 100 for r in s_results]
            row["scenarios"][sname] = {
                "mean_ttc": sum(ttc_vals) / len(ttc_vals),
                "mean_conv": sum(conv_vals) / len(conv_vals),
                "success_rate": sum(r.success_count for r in s_results)
                / sum(r.rounds for r in s_results) * 100,
            }
            all_ttc.extend(ttc_vals)
        row["avg_ttc"] = sum(all_ttc) / len(all_ttc)
        row["worst_ttc"] = max(all_ttc)
        results.append(row)
    return results


# ═══════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  smart-router MC 参数搜索")
    print("=" * 60)

    # ── 加载模型 ──
    models = load_real_models()
    print(f"\n[0] 模型数据: {len(models)} 个模型（混合池）")
    for m in sorted(models, key=lambda x: x.avg_tokens):
        tag = " ⚡不稳定" if m.success_rate < 0.98 else ""
        print(f"    {m.name:<40} avg_tokens={m.avg_tokens:,.0f}  "
              f"成功率={m.success_rate*100:.0f}%{tag}")

    print(f"\n  任务分布（来自 classifications.jsonl）：")
    weights = _load_task_weights()
    for tt, w in zip(ALL_TASK_TYPES, weights):
        bar = "█" * int(w * 40)
        print(f"    {tt:<14} {w*100:5.1f}%  {bar}")

    ROUNDS = 500
    TOTAL_BUDGET = 400  # 总 seed 预算
    MASTER_SEED = 42

    random.seed(MASTER_SEED)
    all_seeds = list(range(1000, 1200))

    # ── 采样 ──
    N_SAMPLES = 200
    print(f"\n[1] 采样 {N_SAMPLES} 组参数（log 空间）...")
    params_list = sample_params(N_SAMPLES)
    for i in range(3):
        p = params_list[i]
        print(f"    样本 {i+1}: c={p['c']:.4f}, alpha={p['alpha']:.6f}, "
              f"decay={p['decay']:.4f}, pp={p['prior_pulls']}, "
              f"pbm={p['prior_blend_min']}")

    # ── 竞速筛选 ──
    print(f"\n[2] 竞速筛选（sequential halving, budget={TOTAL_BUDGET}）...")
    t0 = time.time()
    survivors = racing_search(params_list, models, ROUNDS,
                              TOTAL_BUDGET, all_seeds)
    elapsed = time.time() - t0
    print(f"    耗时: {elapsed:.1f}s, 幸存: {len(survivors)} 个候选")

    print(f"\n    Top-5:")
    for i, c in enumerate(survivors[:5]):
        p = c.params
        print(f"    #{i+1}: c={p['c']:.4f}, alpha={p['alpha']:.6f}, "
              f"decay={p['decay']:.4f}, pp={p['prior_pulls']}, "
              f"pbm={p['prior_blend_min']}, "
              f"LCB={c.lcb_ttc:,.0f}, conv={c.mean_conv_round:.0f}, "
              f"rate={c.mean_success_rate*100:.1f}%")

    # ── NM 精调 ──
    NM_CANDIDATES = min(3, len(survivors))
    print(f"\n[3] NM 精调 top-{NM_CANDIDATES}（确定性重放, seed=42）...")
    nm_seed = 42
    refined = []
    for i in range(NM_CANDIDATES):
        start = survivors[i].params
        best, best_f = nm_refine(start, models, ROUNDS, nm_seed,
                                 max_iter=30)
        p = best
        print(f"    NM#{i+1}:")
        print(f"      start: c={start['c']:.4f}, alpha={start['alpha']:.6f}, "
              f"decay={start['decay']:.4f}")
        print(f"      best:  c={p['c']:.4f}, alpha={p['alpha']:.6f}, "
              f"decay={p['decay']:.4f}, pp={p['prior_pulls']}, "
              f"pbm={p['prior_blend_min']}, f={best_f:,.0f}")
        refined.append(best)

        # 边界诊断：各参数使用不同的贴近判据
        # c/alpha 是 log 尺度 → 贴近下界 = val < lo * 2, 贴近上界 = val > hi / 2
        # decay 是线性 [0.90, 0.999] → 贴近下界 = val < 0.905, 贴近上界 = val > 0.995
        # prior_pulls / prior_blend_min 是整数 → 贴边界 = ±2
        for key, bounds in PARAM_BOUNDS.items():
            val = p[key]
            lo, hi = bounds
            at_lower = at_upper = False
            if key in ("c", "alpha"):
                at_lower = val < lo * 2.0
                at_upper = val > hi * 0.5
            elif key == "decay":
                at_lower = val < lo + 0.005
                at_upper = val > hi - 0.004
            else:
                at_lower = val <= lo + 1
                at_upper = val >= hi - 1
            if at_lower:
                print(f"      ⚠ {key}={val} 贴着下界 {lo}, 建议扩框")
            elif at_upper:
                print(f"      ⚠ {key}={val} 贴着上界 {hi}, 建议扩框")

    # ── 交叉验证 ──
    VAL_SEEDS = list(range(2000, 2020))
    print(f"\n[4] 交叉验证（{len(refined)} 候选 × 2 scenario × {len(VAL_SEEDS)} seeds）...")
    cv_results = cross_validate(refined, models, ROUNDS, VAL_SEEDS)

    print(f"\n    {'候选':<12} {'avg TTC':>10} {'worst':>10} "
          f"{'conv':>6} {'success':>8}")
    print(f"    {'-'*55}")
    for i, r in enumerate(cv_results):
        p = r["params"]
        sf = r["scenarios"]["fixed"]
        sc = r["scenarios"]["pool_change"]
        print(f"    NM#{i+1:<9} {r['avg_ttc']:>10,.0f} {r['worst_ttc']:>10,.0f} "
              f"{sf['mean_conv']:>6.0f} {sf['success_rate']:>7.1f}%")
        print(f"    {'':12} fixed: TTC={sf['mean_ttc']:,.0f}, "
              f"change: TTC={sc['mean_ttc']:,.0f}, conv={sc['mean_conv']:.0f}")

    # ── 最终推荐 ──
    # 选 worst_ttc 最低的（最稳健）
    best = min(cv_results, key=lambda r: r["avg_ttc"])
    bp = best["params"]
    print(f"\n{'='*60}")
    print(f"  推荐参数")
    print(f"{'='*60}")
    print(f"  c:                {bp['c']:.4f}")
    print(f"  alpha:            {bp['alpha']:.6f}")
    print(f"  decay:            {bp['decay']:.4f}")
    print(f"  prior_pulls:      {bp['prior_pulls']}")
    print(f"  prior_blend_min:  {bp['prior_blend_min']}")
    print(f"  avg TTC:          {best['avg_ttc']:,.0f}")
    print(f"  worst TTC:        {best['worst_ttc']:,.0f}")
    print(f"\n  [config.yaml]")
    print(f"  bandit.ucb_c: {bp['c']:.4f}")
    print(f"  bandit.alpha: {bp['alpha']:.6f}")
    print(f"\n  [bandit.py 模块常量 — 需手动编辑]")
    print(f"  DECAY = {bp['decay']:.4f}")
    print(f"  PRIOR_PULLS = {bp['prior_pulls']}")
    print(f"  PRIOR_BLEND_MIN = {bp['prior_blend_min']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
