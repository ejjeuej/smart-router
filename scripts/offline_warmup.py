#!/usr/bin/env python3
"""
离线全信息预热（借鉴点 51, OrcaRouter）：冷启动先验的最终答案。

修正此前"没有偏好数据所以不能离线预热"的排除逻辑：
  预热不需要偏好数据——只需要
    ① 一组代表性 prompt（从 data/classifications.jsonl 历史日志抽）
    ② 把池子里每个模型各跑一遍（full-information，离线一次性 API 成本）
  → 每个模型得到一组四分量奖励（借鉴点 49）→ 平均奖励初始化 Q 值
     （比家族先验 PRIOR_PULLS=3 强得多，且是"自己池子的实测"不是猜测）

neff 语义（借鉴点 42）：预热样本数 n 直接作为等效伪观测数写入 pulls，
均值保持（直接写均值，不缩向零）；几何遗忘保证先验自然衰减。

质量信号（借鉴点 44）：成本信号看不见质量退化。默认 quality=0.5 中性
（奖励只反映成本/延迟差异）；可加 --judge-model 用 judge 模型给响应打
1-5 分充当质量信号，预热结果才有区分度。

用法:
  cd ~/.hermes/plugins/smart-router
  python3 scripts/offline_warmup.py --pool simple --n 20          # 跑 simple 池
  python3 scripts/offline_warmup.py --pool all --n 30 --dry-run  # 只打印计划
  python3 scripts/offline_warmup.py --pool complex --n 15 --judge-model deepseek-chat

参数:
  --pool simple|complex|all   预热哪个池（默认 all）
  --n N                       每个池抽多少条代表性 query（默认 20）
  --judge-model NAME          用该模型给响应打分（1-5 → 质量分），默认不打分
  --dry-run                   只打印计划，不实际调用模型
  --max-tokens N              预热请求的输出上限（默认 256，省钱）
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))

from bandit import UCBBandit, get_bandit, save_one, DEFAULT_PRICE  # noqa: E402

# 质量分：judge 打 1-5 分 → 质量分 (score-1)/4 ∈ [0,1]
_JUDGE_PROMPT = (
    "Rate the quality of the following assistant response to the user's "
    "request on a scale of 1 (useless/wrong) to 5 (excellent). "
    "Respond with ONLY a single digit.\n\n"
    "USER: {user}\n\nASSISTANT: {assistant}"
)


def strip_system_prefix(msg: str) -> str:
    """剥掉 Hermes 切模型时注入的 [System: ...] 前缀块。"""
    s = msg.strip()
    if s.startswith("[System:"):
        idx = s.find("]")
        if idx != -1:
            s = s[idx + 1:].strip()
    return s


def load_history_queries(plugin_dir: Path, n: int, seed: int = 42) -> list:
    """从 classifications.jsonl 抽代表性 query。

    策略：剥 [System:] 前缀 → 去重 → 按长度分桶（短≤30 / 中≤150 / 长>150）
    → 每桶尽量等量抽取 → 补足到 n 条。
    """
    path = plugin_dir / "data" / "classifications.jsonl"
    if not path.exists():
        print(f"[warmup] 无历史日志: {path}，请先用一段时间让日志积累")
        return []

    raw = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            msg = strip_system_prefix(r.get("user_message") or "")
            if len(msg) < 2 or len(msg) > 2000:
                continue
            raw.append(msg)

    # 去重（保序）
    seen = set()
    uniq = []
    for m in raw:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    if not uniq:
        print("[warmup] 历史日志没有可用 query")
        return []

    # 按长度分桶
    buckets = {"short": [], "mid": [], "long": []}
    for m in uniq:
        L = len(m)
        if L <= 30:
            buckets["short"].append(m)
        elif L <= 150:
            buckets["mid"].append(m)
        else:
            buckets["long"].append(m)

    rng = random.Random(seed)
    picked = []
    per_bucket = max(1, n // 3)
    for key in ("short", "mid", "long"):
        pool = buckets[key]
        rng.shuffle(pool)
        picked.extend(pool[:per_bucket])
        print(f"[warmup]   {key:<6} 桶: {len(pool):>4} 条候选, 抽 {min(per_bucket, len(pool))}")
    # 补足到 n
    if len(picked) < n:
        rest = [m for m in uniq if m not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    print(f"[warmup]   合计抽取 {len(picked)} 条代表性 query")
    return picked[:n]


def build_candidates(plugin_dir: Path, pool_key: str) -> list:
    """从 config 加载池子模型（含 endpoint/key）。

    返回 [{"model": name, "base_url": ..., "api_key": ...}]
    """
    sys.path.insert(0, str(plugin_dir))
    from config import load_router_config

    cfg = load_router_config()
    models = cfg.get(pool_key, [])
    providers = cfg.get("providers", {})
    candidates = []
    for m in models:
        entries = providers.get(m) or []
        if not entries:
            continue
        e = entries[0]
        key = e.get("api_key") or e.get("api_key_env")
        if not key:
            # api_key_env 指向环境变量名，需要再解析一次
            from config import get_api_key
            key = e.get("api_key") or get_api_key(e.get("api_key_env", ""))
        if not key or not e.get("base_url"):
            continue
        candidates.append({"model": m, "base_url": e["base_url"], "api_key": key})
    return candidates


def judge_quality(client, judge_model: str, user_msg: str, resp_text: str) -> float:
    """用 judge 模型给响应打分（1-5 → 质量分 [0,1]）。失败回退 0.5。"""
    try:
        r = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                user=user_msg[:1000], assistant=resp_text[:2000])}],
            max_tokens=4,
            temperature=0,
        )
        txt = (r.choices[0].message.content or "").strip()
        score = int(re.sub(r"\D", "", txt)[:1] or "3")
        score = max(1, min(5, score))
        return (score - 1) / 4.0
    except Exception:
        return 0.5


def warmup_pool(pool_key: str, queries: list, args, bandit: UCBBandit,
                cfg: dict) -> None:
    """对一个池跑离线预热，注入 bandit。"""
    candidates = build_candidates(Path(__file__).resolve().parent.parent, pool_key)
    if not candidates:
        print(f"[warmup] {pool_key}: 无可用候选（config 未发现 endpoint）")
        return

    print(f"\n[warmup] ═══ 预热 {pool_key}: {len(candidates)} 个模型 × {len(queries)} 条 query ═══")
    if args.dry_run:
        for c in candidates:
            print(f"  [dry-run] 会跑 {c['model']} × {len(queries)} 次")
        return

    from openai import OpenAI

    results = {}  # model → [reward, ...]
    token_totals = {}  # model → 累计 tokens
    for c in candidates:
        client = OpenAI(base_url=c["base_url"], api_key=c["api_key"], timeout=30.0)
        rewards = []
        tokens_sum = 0
        fail_count = 0
        for q in queries:
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=c["model"],
                    messages=[{"role": "user", "content": q}],
                    max_tokens=args.max_tokens,
                    temperature=0.3,
                )
                latency_ms = (time.time() - t0) * 1000
                total_tokens = resp.usage.total_tokens if resp.usage else 0
                tokens_sum += total_tokens

                quality = 0.5
                if args.judge_model:
                    resp_text = (resp.choices[0].message.content or "")[:2000]
                    quality = judge_quality(client, args.judge_model, q, resp_text)

                # 四分量奖励（借鉴点 49）—— 与生产 update() 同一套公式
                from bandit import LATENCY_MAX
                q_t = 1.0 * quality - 0.4 * bandit._log_norm_cost(
                    total_tokens * bandit._price_of(c["model"]) / 1000.0
                ) - 0.3 * min(1.0, latency_ms / LATENCY_MAX)
                rewards.append(max(q_t, 0.01))
            except Exception as e:
                fail_count += 1
                rewards.append(-0.5)  # 调用失败按 403 级惩罚
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        avg_tokens = tokens_sum / len(rewards) if rewards else 5000.0
        results[c["model"]] = avg
        token_totals[c["model"]] = avg_tokens
        print(f"  [warmup] {c['model']:<28} avg_reward={avg:+.3f} "
              f"avg_tokens={avg_tokens:.0f} 失败={fail_count}")

    # 注入 bandit
    for model, avg in results.items():
        bandit.inject_offline_prior(model, avg, n=len(queries),
                                    avg_tokens=token_totals.get(model, 0.0))
    save_one(pool_key)
    print(f"[warmup] ✓ 已注入 {pool_key}：pulls={len(queries)}（neff），"
          f"总模型 {len(results)} 个")


def main():
    ap = argparse.ArgumentParser(description="离线全信息预热（借鉴点 51）")
    ap.add_argument("--pool", choices=["simple", "complex", "all"], default="all")
    ap.add_argument("--n", type=int, default=20, help="每池抽取 query 数（默认 20）")
    ap.add_argument("--judge-model", default="", help="judge 模型名（可选）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    plugin_dir = Path(__file__).resolve().parent.parent
    queries = load_history_queries(plugin_dir, args.n)
    if not queries:
        sys.exit(1)

    sys.path.insert(0, str(plugin_dir))
    from config import load_router_config

    cfg = load_router_config()
    bandit_cfg = cfg.get("bandit", {})

    pools = ["simple_models", "complex_models"] if args.pool == "all" else \
        [f"{args.pool}_models"]
    for pool_key in pools:
        bandit = get_bandit(pool_key, bandit_cfg)
        warmup_pool(pool_key, queries, args, bandit, cfg)

    print("\n[warmup] 完成。重启 App 后 bandit 将用预热先验参与 UCB 评分；")
    print("         或直接查看 data/bandit_*.json 确认注入结果。")


if __name__ == "__main__":
    main()
