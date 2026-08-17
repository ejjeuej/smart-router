#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
借鉴点 33 — 分位数阈值校准（RouteLLM calibrate_threshold.py 的落地模板）

输入：data/classifications.jsonl（每条含 confidence + complexity）
输出：
  1. confidence 分布摘要（分位数 + 直方图）
  2. 目标 complex 比例 → conf 阈值 对照表
  3. length 分布 + length>200 样本复核（验证"长粘贴不定级"是否生效）

用法：
  python calibrate_threshold.py                      # 默认读 data/classifications.jsonl
  python calibrate_threshold.py --data <path>        # 指定历史日志
  python calibrate_threshold.py --complex-ratio 0.2  # 只看目标 20% complex 的阈值

方向约定（conf 语义 = P(simple 够用)，见 classifier.py 文档）：
  conf 高 → simple；conf 低 → complex。
  所以"让 X% 请求走 complex" = threshold 取 conf 分布的 X 分位数
  （conf < threshold 的请求走 complex）。

纯标准库，离线运行，不依赖 Hermes 运行时。
"""

import argparse
import json
import os
from collections import Counter

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "classifications.jsonl")


def load_records(path):
    """读 jsonl，过滤无 confidence/complexity 的脏行。"""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "confidence" not in rec or "complexity" not in rec:
                continue
            rec["_len"] = len(rec.get("user_message", ""))
            records.append(rec)
    return records


def percentile(sorted_vals, p):
    """p 为 0~1，线性插值分位数（RouteLLM np.percentile 的等价物）。"""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def histogram(values, n_bins=10, width=40):
    """ASCII 直方图。"""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if hi == lo:
        return f"all equal ({lo:.2f})"
    bins = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)
        bins[idx] += 1
    max_c = max(bins) or 1
    lines = []
    for i, c in enumerate(bins):
        left = lo + (hi - lo) * i / n_bins
        right = lo + (hi - lo) * (i + 1) / n_bins
        bar = "#" * round(c / max_c * width)
        lines.append(f"  [{left:6.2f},{right:6.2f}) {'%4d' % c} {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="分位数阈值校准：由历史 conf 分布反推达到目标 complex 比例的阈值")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="历史分类日志路径（默认 data/classifications.jsonl）")
    ap.add_argument("--complex-ratio", type=float, default=None,
                    help="只打印目标 complex 比例的阈值（如 0.2 = 20%% 走 complex）")
    ap.add_argument("--bins", type=int, default=10,
                    help="直方图分桶数（默认 10）")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[错误] 找不到历史日志: {args.data}\n"
              f"先跑一段时间让 data/classifications.jsonl 积累数据,再校准。")
        return 1

    records = load_records(args.data)
    if not records:
        print(f"[错误] {args.data} 里没有可用的记录(缺 confidence/complexity 字段)。")
        return 1

    confs = sorted(r["confidence"] for r in records)
    lens = [r["_len"] for r in records]
    n = len(records)
    simple_n = sum(1 for r in records if r["complexity"] == "simple")
    complex_n = n - simple_n

    print(f"[数据] {n} 条历史记录 | simple={simple_n} ({simple_n / n:.1%}) "
          f"| complex={complex_n} ({complex_n / n:.1%})\n")

    # ── 1. confidence 分布摘要 ──
    print("[confidence 分布]")
    for p in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0):
        print(f"  p{int(p * 100):>3} = {percentile(confs, p):.3f}")
    print("  直方图:")
    print(histogram(confs, args.bins))
    print()

    # ── 2. 目标 complex 比例 → conf 阈值 对照表 ──
    print("[目标 complex 比例 → conf 阈值]")
    print("  (方向: conf < threshold 的请求走 complex)")
    print("  complex 比例    阈值    覆盖请求数")
    ratios = [args.complex_ratio] if args.complex_ratio else \
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for ratio in ratios:
        t = percentile(confs, ratio)
        cnt = sum(1 for c in confs if c < t)
        print(f"  {ratio:8.0%}     {t:.3f}    {cnt}/{n}")
    print()

    # ── 3. length 复核（验证借鉴点 25"长粘贴不定级"）──
    print("[length 分布]")
    for p in (0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0):
        print(f"  p{int(p * 100):>3} = {percentile(sorted(lens), p):.0f} 字符")
    long_recs = [r for r in records if r["_len"] > 200]
    if long_recs:
        long_simple = sum(1 for r in long_recs if r["complexity"] == "simple")
        print(f"\n  len>200 共 {len(long_recs)} 条:"
              f" simple={long_simple} / complex={len(long_recs) - long_simple}")
        print("  (期望: 纯长粘贴落 simple——长+无信号不再强制 complex,"
              "即借鉴点 25 生效)")
        for r in long_recs[:10]:
            snippet = r.get("user_message", "")[:60].replace("\n", " ")
            print(f"    [{r['complexity']:7s}] conf={r['confidence']:.2f} "
                  f"len={r['_len']:4d} {snippet!r}")
    else:
        print("\n  len>200 的记录 0 条(数据还少,继续积累)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
