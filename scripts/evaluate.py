"""
老虎机算法离线评估脚本。

用法：
  python scripts/evaluate.py

指标：
  1. 成功率     — routing_success=true 的比例
  2. 模型分布   — 各模型被选中次数（看是否过度集中或分散）
  3. 分类准确率 — LLM 分类 vs 规则分类的占比、分类延迟
  4. 复杂度分布 — simple/medium/complex 的实际分布
  5. 任务类型分布 — 各类 task_type 占比
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def load_data():
    path = Path(__file__).resolve().parent.parent / "data" / "classifications.jsonl"
    if not path.exists():
        print(f"数据文件不存在: {path}")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def main():
    records = load_data()
    if not records:
        print("没有数据。")
        return

    total = len(records)
    success = [r for r in records if r.get("routing_success") is True]
    failure = [r for r in records if r.get("routing_success") is False]
    unknown = [r for r in records if r.get("routing_success") is None]

    # ── 1. 成功率 ──
    success_count = len(success)
    failure_count = len(failure)
    unknown_count = len(unknown)
    routed = success_count + failure_count
    success_rate = success_count / routed * 100 if routed > 0 else 0

    print("=" * 60)
    print("  老虎机算法离线评估报告")
    print("=" * 60)
    print(f"\n  总记录数: {total}")
    print(f"  路由成功: {success_count} ({success_rate:.1f}%)")
    print(f"  路由失败: {failure_count}")
    print(f"  未路由（开关关闭）: {unknown_count}")

    # ── 2. 模型分布 ──
    model_counter = Counter(r["model_routed_to"] for r in records if r.get("model_routed_to"))
    print(f"\n── 模型选中分布 ──")
    for model, count in model_counter.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {model:<35} {count:>4} 次 ({pct:5.1f}%)  {bar}")

    # ── 3. 分类方式 ──
    method_counter = Counter(r.get("method", "?") for r in records)
    print(f"\n── 分类方式 ──")
    for method, count in method_counter.most_common():
        print(f"  {method}: {count} 次 ({count/total*100:.1f}%)")

    # 分类延迟
    llm_latencies = [r["latency_ms"] for r in records if r.get("method") == "llm"]
    rule_latencies = [r["latency_ms"] for r in records if r.get("method") == "rule"]
    if llm_latencies:
        print(f"\n── 分类延迟 ──")
        print(f"  LLM 分类: avg={sum(llm_latencies)/len(llm_latencies):.0f}ms, "
              f"max={max(llm_latencies)}ms")
    if rule_latencies:
        print(f"  规则分类: avg={sum(rule_latencies)/len(rule_latencies):.0f}ms, "
              f"max={max(rule_latencies)}ms")

    # ── 4. 复杂度分布 ──
    complexity_counter = Counter(r.get("complexity", "?") for r in records)
    print(f"\n── 复杂度分布 ──")
    for c, count in complexity_counter.most_common():
        print(f"  {c:<10} {count:>4} 次 ({count/total*100:.1f}%)")

    # ── 5. 任务类型分布 ──
    task_counter = Counter(r.get("task_type", "?") for r in records)
    print(f"\n── 任务类型分布 ──")
    for t, count in task_counter.most_common():
        print(f"  {t:<15} {count:>4} 次 ({count/total*100:.1f}%)")

    # ── 6. 按模型看成功率 ──
    print(f"\n── 各模型路由成功率 ──")
    model_results = defaultdict(lambda: {"success": 0, "failure": 0})
    for r in records:
        m = r.get("model_routed_to")
        if m and r.get("routing_success") is not None:
            if r["routing_success"]:
                model_results[m]["success"] += 1
            else:
                model_results[m]["failure"] += 1
    for model, counts in sorted(model_results.items()):
        total_model = counts["success"] + counts["failure"]
        rate = counts["success"] / total_model * 100 if total_model > 0 else 0
        print(f"  {model:<35} {counts['success']}/{total_model} ({rate:.0f}%)")

    # ── 7. 按复杂度看路由模型分布 ──
    print(f"\n── 各复杂度路由模型分布 ──")
    for complexity in ("simple", "medium", "complex"):
        subset = [r for r in records if r.get("complexity") == complexity
                  and r.get("model_routed_to")]
        if not subset:
            continue
        counter = Counter(r["model_routed_to"] for r in subset)
        print(f"  [{complexity}]")
        for model, count in counter.most_common(5):
            print(f"    {model:<33} {count} 次")

    # ── 8. 按日期看趋势 ──
    print(f"\n── 每日路由统计 ──")
    daily = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
    for r in records:
        date = r["timestamp"][:10]
        daily[date]["total"] += 1
        if r.get("routing_success") is True:
            daily[date]["success"] += 1
        elif r.get("routing_success") is False:
            daily[date]["failure"] += 1
    for date in sorted(daily):
        d = daily[date]
        rate = d["success"] / (d["success"] + d["failure"]) * 100 \
               if (d["success"] + d["failure"]) > 0 else 0
        print(f"  {date}: {d['total']} 次, 成功 {d['success']}/{d['success']+d['failure']} "
              f"({rate:.0f}%), 未路由 {d['total']-d['success']-d['failure']}")

    print()


if __name__ == "__main__":
    main()
