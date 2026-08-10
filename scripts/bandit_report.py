#!/usr/bin/env python3
"""
bandit 状态查看工具

读 data/bandit_simple_models.json 和 data/bandit_complex_models.json，
输出: 模型排名、reward分差、任务类型偏好、诊断建议。

用法:
  python3 scripts/bandit_report.py           # 看所有池
  python3 scripts/bandit_report.py simple     # 只看 simple
  python3 scripts/bandit_report.py complex    # 只看 complex
"""

import json, sys, math
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DECAY = 0.95  # bandit.py 里的 DECAY 常量


def load(pool):
    path = DATA / f"bandit_{pool}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report(pool, data):
    total = data["total_rounds"]
    c = data["c"]
    alpha = data["alpha"]
    stats = data["stats"]

    print(f"\n{'='*60}")
    print(f"  {pool}")
    print(f"  总轮次: {total}    c={c}    alpha={alpha}")
    print(f"  衰减因子: {DECAY}    有效记忆: ~{int(1/(1-DECAY))} 轮")
    print(f"{'='*60}")

    if not stats:
        print("  (无数据)")
        return

    # ── 模型总览 ──
    print(f"\n  {'模型':<28} {'pulls':>6} {'avg':>7} {'avgTok':>8}    分布")
    print(f"  {'-'*65}")

    for name, s in sorted(stats.items()):
        o = s.get("overall", {})
        p = o.get("pulls", 0)
        r = o.get("total_reward", 0)
        t = o.get("total_tokens", 0)
        avg = r / p if p > 0 else 0
        avgt = t / p if p > 0 else 0
        bar = "\u2588" * max(1, int(p * 2))
        print(f"  {name:<28} {p:>6.1f} {avg:>7.3f} {avgt:>8.0f}    {bar}")

    # ── 排名 ──
    ranked = sorted(
        [(n, s["overall"]["total_reward"] / max(s["overall"]["pulls"], 0.01))
         for n, s in stats.items()],
        key=lambda x: -x[1]
    )
    spread = ranked[0][1] - ranked[-1][1]
    print(f"\n  排名: {' > '.join(f'{n}({v:.3f})' for n, v in ranked)}")
    print(f"  分差: {spread:.3f}  "
          f"{'← 够大，算法可区分' if spread > 0.1 else '← 偏小，模型间差异不够'}")

    # ── 收敛度 ──
    top_pulls = stats[ranked[0][0]]["overall"]["pulls"]
    concentration = top_pulls / sum(
        s["overall"]["pulls"] for s in stats.values()
    ) if total > 0 else 0
    print(f"  主力占比: {concentration:.0%}  "
          f"{'← 已收敛' if concentration > 0.4 else '← 均匀探索'}")

    # ── 按任务类型 ──
    task_types = set()
    for s in stats.values():
        for k in s:
            if k != "overall":
                task_types.add(k)

    if task_types:
        print(f"\n  按任务类型:")
        for tt in sorted(task_types):
            tt_models = []
            for name, s in stats.items():
                td = s.get(tt, {})
                tp = td.get("pulls", 0)
                tr = td.get("total_reward", 0)
                if tp > 0:
                    tt_models.append((name, tr / tp, tp))
            tt_models.sort(key=lambda x: -x[1])
            parts = [f"{n}({v:.3f},p={tp:.1f})" for n, v, tp in tt_models]
            print(f"    [{tt}]  {' > '.join(parts)}")

    # ── 诊断 ──
    print(f"\n  诊断:")
    if total < 10:
        print(f"    \u26a0 仅 {total} 轮，样本太少无统计意义")
    elif total < 30:
        print(f"    \u2139 {total} 轮，刚开始积累")

    if spread < 0.05:
        print(f"    \u26a0 分差极小 — 模型几乎无差异，bandit 无法区分优劣")
    if c > 0.5 and ranked[0][1] < 1.5:
        print(f"    \u26a0 c={c} 偏大，探索项压过利用项，建议 c=0.1~0.3")
    if concentration > 0.5:
        print(f"    \u2713 已收敛到主力模型")
    if len(task_types) >= 2:
        # 检查不同 task_type 的排名是否不同
        top_by_task = {}
        for tt in task_types:
            best = max(
                ((n, s.get(tt, {}).get("pulls", 0))
                 for n, s in stats.items()),
                key=lambda x: x[1]
            )
            if best[1] > 0:
                top_by_task[tt] = best[0]
        if len(set(top_by_task.values())) >= 2:
            print(f"    \u2713 不同任务选了不同最优模型 — bandit 学到了差异化偏好")


def main():
    pools = ["simple_models", "complex_models"]
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "simple":
            pools = ["simple_models"]
        elif arg == "complex":
            pools = ["complex_models"]

    for pool in pools:
        data = load(pool)
        if data is None:
            print(f"\n{pool}: 文件不存在（该池还没被调用过）")
            continue
        report(pool, data)

    print()


if __name__ == "__main__":
    main()
