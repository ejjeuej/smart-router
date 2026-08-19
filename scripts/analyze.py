#!/usr/bin/env python3
"""smart-router 现状盘点 + 省钱估算（只读，零 API 成本）。

替代旧版 offline_replay / evaluate / bandit_report 等 8 个脚本。
直接读本插件 data/ 目录（脚本所在目录的上级），无需适配 HERMES_HOME。

用法:
    python scripts/analyze.py
"""
import json
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
BURN_IN = 20  # bandit burn-in 阈值


def load_jsonl(p):
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    cls = load_jsonl(DATA / "classifications.jsonl")
    print("=" * 64)
    print(f"路由记录: {len(cls)} 条")
    print("=" * 64)
    if cls:
        succ = sum(1 for r in cls if r.get("routing_success"))
        print(f"  成功率: {succ}/{len(cls)}")
        print(f"  复杂度: {dict(Counter(r.get('complexity') for r in cls))}")
        print(f"  模型:   {dict(Counter(r.get('model_actual') for r in cls))}")
        print(f"  任务:   {dict(Counter(r.get('task_type') for r in cls))}")
        print(f"  方法:   {dict(Counter(r.get('method') for r in cls))}")
        # 时间跨度
        ts = [r.get("timestamp", "") for r in cls if r.get("timestamp")]
        if ts:
            print(f"  时间跨度: {min(ts)[:19]} ~ {max(ts)[:19]} (UTC)")

    pools = {}
    for name, fn in [
        ("simple", "bandit_simple_models.json"),
        ("complex", "bandit_complex_models.json"),
    ]:
        p = DATA / fn
        if p.exists():
            pools[name] = json.loads(p.read_text(encoding="utf-8"))

    model_avg = {}  # model -> (avg_token_per_pull, pool_name)
    for name, d in pools.items():
        print()
        print("=" * 64)
        print(
            f"{name} 池  rounds={d.get('total_rounds')}  "
            f"cost_ema={d.get('cost_ema', 0):.6f}  lambda_t={d.get('lambda_t', 0):.4f}"
        )
        print("=" * 64)
        stats = d.get("stats", {})
        plays = d.get("plays", {})
        print(f"  {'模型':<20} {'pulls':>8} {'tokens':>10} {'avg/pull':>9} {'plays':>6} {'burn-in':>9}")
        for m, s in stats.items():
            ov = s.get("overall", {})
            pulls = ov.get("pulls", 0)
            toks = ov.get("total_tokens", 0)
            avg = toks / pulls if pulls else 0
            pl = plays.get(m, 0)
            model_avg[m] = (avg, name)
            bi = "PASS" if pl >= BURN_IN else f"{pl}/{BURN_IN}"
            print(f"  {m:<20} {pulls:8.1f} {toks:10.0f} {avg:9.0f} {pl:6d} {bi:>9}")
        # 各任务类型 token 分布（看模型在哪些任务上被用）
        for m, s in stats.items():
            tasks = {k: v for k, v in s.items() if k != "overall"}
            if tasks:
                parts = ", ".join(f"{t}:{v['total_tokens']:.0f}" for t, v in tasks.items())
                print(f"    {m} 任务token: {parts}")

    # 省钱估算
    print()
    print("=" * 64)
    print("省钱估算（粗略参考）")
    print("=" * 64)
    if not cls or not model_avg:
        print("  数据不足")
        return
    complex_avgs = {m: v[0] for m, v in model_avg.items() if v[1] == "complex"}
    if not complex_avgs:
        print("  complex 池无数据，无法估算")
        return
    baseline_model = max(complex_avgs, key=complex_avgs.get)
    baseline_avg = complex_avgs[baseline_model]
    print(f"  baseline 假设: 全用 {baseline_model} (avg {baseline_avg:.0f} tok/pull)")

    actual_tok = 0.0
    baseline_tok = 0.0
    n = 0
    for r in cls:
        m = r.get("model_actual")
        if not m or m not in model_avg:
            continue
        n += 1
        actual_tok += model_avg[m][0]
        baseline_tok += baseline_avg
    if baseline_tok and n:
        saved = baseline_tok - actual_tok
        print(f"  样本: {n} 条")
        print(f"  实际估算 token: {actual_tok:.0f}")
        print(f"  baseline token:  {baseline_tok:.0f}")
        print(f"  省: {saved:.0f} ({saved / baseline_tok * 100:.1f}%)")
        print("  WARNING: classifications 无 per-request token，用衰减均值估算，噪声大")
        print("           要精确数字必须做 A/B 实验（关插件 vs 开插件各跑同一批问题）")


if __name__ == "__main__":
    main()
