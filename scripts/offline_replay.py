#!/usr/bin/env python3
"""
离线回放：用真实 classifications.jsonl 重放 UCB bandit，
对比"bandit 会选什么" vs "实际选了什么"，算 token 节省量。

三组对比：
  - replay:  新鲜 bandit(c=0.05, alpha=0.012) 重放 → 会选什么、花多少 token
  - actual:  实际记录里选了什么模型（用该模型历史 avg_tokens 估算消耗）
  - oracle:  总是选池子里 avg_tokens 最低的模型 → 理论下界
  - random:  均匀随机选 → 基线

还模拟一个"中途换池"场景：前 N/2 条用原池，后 N/2 条替换 2 个模型，
检验换池后 bandit 多快恢复。

用法:
  cd ~/.hermes/plugins/smart-router
  python3 scripts/offline_replay.py
"""

import sys, os, json, math, random
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))
import bandit


# ═══════════════════════════════════════════════════════════════════
# 0. 加载数据
# ═══════════════════════════════════════════════════════════════════

def load_records():
    """加载 classifications.jsonl，按时间排序。"""
    p = _PLUGIN_DIR / "data" / "classifications.jsonl"
    if not p.exists():
        print("无 classifications.jsonl")
        return []
    records = []
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
                records.append(r)
            except Exception:
                pass
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


def load_model_tokens():
    """从 bandit JSON 提取每个模型的 overall avg_tokens 和 per-task_type avg_tokens。"""
    tokens = {}
    for pool_file in (_PLUGIN_DIR / "data").glob("bandit_*.json"):
        data = json.loads(pool_file.read_text())
        for model, stats in data.get("stats", {}).items():
            overall = stats.get("overall", {})
            pulls = overall.get("pulls", 0)
            if pulls <= 0:
                continue
            task_tokens = {}
            for tt, ts in stats.items():
                if tt == "overall":
                    continue
                tp = ts.get("pulls", 0)
                if tp > 0:
                    task_tokens[tt] = ts["total_tokens"] / tp
            tokens[model] = {
                "avg_tokens": overall["total_tokens"] / pulls,
                "task_tokens": task_tokens,
            }
    return tokens


def load_pool_config():
    """从 config.yaml 读取当前池配置。"""
    import yaml
    config_path = Path.home() / ".hermes" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    sr = cfg.get("smart_model_routing", {})
    return {
        "simple": sr.get("simple_models", []),
        "complex": sr.get("complex_models", []),
    }


# ═══════════════════════════════════════════════════════════════════
# 1. 回放
# ═══════════════════════════════════════════════════════════════════

SIMPLE_TASKS = {"chat", "translation", "other"}
COMPLEX_TASKS = {"coding", "reasoning"}


def pool_for_task(task_type: str) -> str:
    if task_type in SIMPLE_TASKS:
        return "simple"
    return "complex"


@dataclass
class ReplayResult:
    model: str
    estimated_tokens: float
    success: bool


def estimate_tokens(model: str, task_type: str, model_tokens: dict) -> float:
    """估算某模型处理某 task_type 的 token 消耗。"""
    if model not in model_tokens:
        return 10000  # 未知模型，保守估计
    mt = model_tokens[model]
    if task_type in mt["task_tokens"]:
        return mt["task_tokens"][task_type]
    return mt["avg_tokens"]


def replay_bandit(records, pool_config, model_tokens, bandit_params):
    """用新鲜 bandit 重放所有记录。

    bandit_params: {"c": 0.05, "alpha": 0.012}
    返回: [(replay_result, actual_model), ...]
    """
    bb = bandit.UCBBandit(c=bandit_params["c"], alpha=bandit_params["alpha"],
                          base_reward=100.0)
    results = []

    for rec in records:
        tt = rec.get("task_type", "other")
        pool_key = pool_for_task(tt)
        models = pool_config.get(pool_key, [])

        if not models:
            results.append((None, rec.get("model_routed_to", "")))
            continue

        candidates = [{"model": m, "base_url": "", "api_key": ""}
                      for m in models]

        selected = bb.select(candidates, task_type=tt)
        if not selected:
            results.append((None, rec.get("model_routed_to", "")))
            continue

        sel_name = selected["model"]
        tokens = estimate_tokens(sel_name, tt, model_tokens)
        success = rec.get("routing_success", True)

        bb.update(sel_name, success, int(tokens), tt)

        results.append((
            ReplayResult(model=sel_name, estimated_tokens=tokens,
                         success=success),
            rec.get("model_routed_to", ""),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════
# 2. 对比
# ═══════════════════════════════════════════════════════════════════

def compute_baselines(records, pool_config, model_tokens):
    """计算 actual / oracle / random 三条基线。"""
    actual_tokens = []
    actual_models = Counter()
    actual_success = 0
    oracle_tokens = []
    random_tokens = []
    per_pool_models = {}

    rng = random.Random(42)

    for rec in records:
        tt = rec.get("task_type", "other")
        pool_key = pool_for_task(tt)
        models = pool_config.get(pool_key, [])

        # --- actual ---
        actual_model = rec.get("model_routed_to", "")
        tok = estimate_tokens(actual_model, tt, model_tokens)
        actual_tokens.append(tok)
        actual_models[actual_model] += 1
        if rec.get("routing_success", True):
            actual_success += 1

        # --- oracle: 总是选池子里最便宜的 ---
        if models:
            cheapest = min(models,
                          key=lambda m: estimate_tokens(m, tt, model_tokens))
            oracle_tokens.append(estimate_tokens(cheapest, tt, model_tokens))

        # --- random: 均匀随机选 ---
        if models:
            rand_model = rng.choice(models)
            random_tokens.append(estimate_tokens(rand_model, tt, model_tokens))

    return {
        "actual_total": sum(actual_tokens),
        "actual_avg": sum(actual_tokens) / len(actual_tokens) if actual_tokens else 0,
        "actual_success_rate": actual_success / len(records) * 100 if records else 0,
        "actual_models": actual_models,
        "oracle_total": sum(oracle_tokens),
        "oracle_avg": sum(oracle_tokens) / len(oracle_tokens) if oracle_tokens else 0,
        "random_total": sum(random_tokens),
        "random_avg": sum(random_tokens) / len(random_tokens) if random_tokens else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. 换池场景模拟
# ═══════════════════════════════════════════════════════════════════

def simulate_pool_change(records, pool_config, model_tokens, bandit_params):
    """模拟中途换池：前一半用原池，后一半换掉 2 个旧模型 + 加入 2 个新模型。

    看换池后多少轮新模型被选中的比例追上旧模型。
    """
    n = len(records)
    mid = n // 2

    # 构建换池后的配置
    changed_pool = {}
    for pool_key in ("simple", "complex"):
        old_models = list(pool_config.get(pool_key, []))
        if len(old_models) >= 4:
            # 去掉前 2 个，加 2 个新模型标记
            removed = old_models[:2]
            kept = old_models[2:]
            new = [f"new-{pool_key}-cheap", f"new-{pool_key}-expensive"]
            changed_pool[pool_key] = kept + new
            # 给新模型加 token 估算
            model_tokens[f"new-{pool_key}-cheap"] = {
                "avg_tokens": 3000,
                "task_tokens": {},
            }
            model_tokens[f"new-{pool_key}-expensive"] = {
                "avg_tokens": 30000,
                "task_tokens": {},
            }
        else:
            changed_pool[pool_key] = old_models

    bb = bandit.UCBBandit(c=bandit_params["c"], alpha=bandit_params["alpha"],
                          base_reward=100.0)

    new_model_selections = Counter()
    old_model_selections = Counter()

    for i, rec in enumerate(records):
        tt = rec.get("task_type", "other")
        pool_key = pool_for_task(tt)

        if i < mid:
            models = pool_config.get(pool_key, [])
        else:
            models = changed_pool.get(pool_key, [])

        if not models:
            continue

        candidates = [{"model": m, "base_url": "", "api_key": ""}
                      for m in models]
        selected = bb.select(candidates, task_type=tt)
        if not selected:
            continue

        sel_name = selected["model"]
        tokens = estimate_tokens(sel_name, tt, model_tokens)
        success = rec.get("routing_success", True)
        bb.update(sel_name, success, int(tokens), tt)

        if f"new-{pool_key}" in sel_name:
            new_model_selections[sel_name] += 1
        else:
            old_model_selections[sel_name] += 1

    return new_model_selections, old_model_selections, mid


# ═══════════════════════════════════════════════════════════════════
# 4. 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  离线回放 — Bandit 路由效果评估")
    print("=" * 60)

    records = load_records()
    if not records:
        print("无数据，先跑 batch_test.py")
        return

    model_tokens = load_model_tokens()
    pool_config = load_pool_config()

    print(f"\n  记录数: {len(records)}")
    print(f"  Simple 池: {pool_config['simple']}")
    print(f"  Complex 池: {pool_config['complex']}")

    bandit_params = {"c": 0.05, "alpha": 0.012}

    # ── 回放 ──
    print(f"\n  [1] 回放 bandit (c={bandit_params['c']}, alpha={bandit_params['alpha']})...")
    results = replay_bandit(records, pool_config, model_tokens, bandit_params)

    replay_total = 0
    replay_models = Counter()
    replay_success = 0
    match_count = 0  # bandit 和实际选了同一个模型的次数

    for replay_r, actual_model in results:
        if replay_r:
            replay_total += replay_r.estimated_tokens
            replay_models[replay_r.model] += 1
            if replay_r.success:
                replay_success += 1
            if replay_r.model == actual_model:
                match_count += 1

    replay_avg = replay_total / len(records) if records else 0
    replay_success_rate = replay_success / len(records) * 100 if records else 0

    # ── 基线 ──
    baselines = compute_baselines(records, pool_config, model_tokens)

    # ── 对比 ──
    print(f"\n  [2] Token 消耗对比（估算）")
    print(f"  {'':20} {'总计':>12} {'平均':>10} {'vs oracle':>10}")
    print(f"  {'-'*55}")
    print(f"  {'oracle（理论最优）':20} {baselines['oracle_total']:>12,.0f} "
          f"{baselines['oracle_avg']:>10,.0f}")
    print(f"  {'bandit 回放':20} {replay_total:>12,.0f} "
          f"{replay_avg:>10,.0f} "
          f"{'+'+str(int((replay_total/baselines['oracle_total']-1)*100))+'%' if baselines['oracle_total'] > 0 else '':>10}")
    print(f"  {'实际路由':20} {baselines['actual_total']:>12,.0f} "
          f"{baselines['actual_avg']:>10,.0f}")
    print(f"  {'随机基线':20} {baselines['random_total']:>12,.0f} "
          f"{baselines['random_avg']:>10,.0f}")

    savings_vs_actual = (1 - replay_total / baselines['actual_total']) * 100 \
        if baselines['actual_total'] > 0 else 0
    savings_vs_random = (1 - replay_total / baselines['random_total']) * 100 \
        if baselines['random_total'] > 0 else 0

    print(f"\n  bandit vs 实际: {savings_vs_actual:+.1f}% token")
    print(f"  bandit vs 随机: {savings_vs_random:+.1f}% token")

    # ── 模型选中分布 ──
    print(f"\n  [3] 模型选中分布")
    print(f"  {'模型':<30} {'bandit':>8} {'实际':>8}")
    print(f"  {'-'*48}")
    all_models = sorted(set(list(replay_models.keys()) +
                            list(baselines['actual_models'].keys())))
    for m in all_models:
        rp = replay_models.get(m, 0)
        ac = baselines['actual_models'].get(m, 0)
        print(f"  {m:<30} {rp:>8} {ac:>8}")

    print(f"\n  bandit 与实际选同一模型: {match_count}/{len(records)} "
          f"({match_count/len(records)*100:.0f}%)")

    # ── 成功率 ──
    print(f"\n  [4] 成功率")
    print(f"  bandit: {replay_success_rate:.1f}%")
    print(f"  实际:   {baselines['actual_success_rate']:.1f}%")

    # ── 换池模拟 ──
    print(f"\n  [5] 换池模拟（中途替换 2 个模型）")
    new_sel, old_sel, mid = simulate_pool_change(
        records, pool_config, model_tokens, bandit_params)

    total_after_mid = sum(new_sel.values()) + sum(old_sel.values())
    if total_after_mid > 0:
        new_ratio = sum(new_sel.values()) / total_after_mid * 100
        print(f"  换池后新模型占比: {new_ratio:.0f}%")
        print(f"  新模型被选次数:")
        for m, c in new_sel.most_common():
            print(f"    {m}: {c}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
