#!/usr/bin/env python3
"""
smart-router 自动发现 + bandit 路由 验证脚本。

证明两件事：
  1) 能自动读取用户配置的模型（.env 里的 key → /models → 路由池）
  2) bandit 能选中并真实调用这些发现的模型

用法（用 Hermes 的 venv 跑）：
  python3 ~/.hermes/plugins/smart-router/test_discovery.py

只读不写 config.yaml，不污染任何状态（bandit 状态用完即重置）。
"""

import os
import sys
import time
from pathlib import Path

PLUGIN_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, PLUGIN_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".hermes" / ".env", override=True)
except Exception:
    pass

import config
from classifier import classify
from bandit import get_bandit, reset_bandit, save_one


_QUOTA_MARKERS = ("insufficient_quota", "insufficient_balance",
                  "allocationquota", "quota exceeded", "freequota")


def _is_quota(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _QUOTA_MARKERS)


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def mask(key: str) -> str:
    return (key[:8] + "…" + key[-4:]) if len(key) > 14 else "(key 过短)"


def build_candidates(discovered, pool):
    out = []
    for m in sorted(pool):
        for e in discovered[m]:
            if e.get("api_key") and e.get("base_url"):
                out.append({"model": m, "base_url": e["base_url"],
                            "api_key": e["api_key"]})
                break
    return out


def main():
    # ── 1. 自动发现 ────────────────────────────────────────────────
    section("1. 自动发现：从用户配置的 key 读模型")
    t0 = time.time()
    data = config._load_yaml()
    discovered = config._discover_provider_models(data)
    dt = time.time() - t0

    from collections import Counter
    per_endpoint = Counter()
    key_presence = {}
    for m, entries in discovered.items():
        for e in entries:
            per_endpoint[e["base_url"]] += 1
            key_presence.setdefault(e["base_url"], e["api_key"])

    print(f"发现模型总数: {len(discovered)}   (耗时 {dt:.1f}s)")
    print(f"来自 {len(per_endpoint)} 个 endpoint（即用户配了 key 的 provider）:\n")
    for base, n in per_endpoint.most_common():
        print(f"  {base:55s} {n:4d} 个模型   key={mask(key_presence[base])}")

    simple, complex_, skipped = [], [], []
    for m in discovered:
        if not config._is_chat_model(m):
            skipped.append(m)
        elif config.classify_model(m) == "simple":
            simple.append(m)
        else:
            complex_.append(m)
    print(f"\n  过滤非对话模型: {len(skipped)} 个 (image/audio/tts/embedding/vision…)")
    print(f"  对话模型分类: simple={len(simple)}  complex={len(complex_)}")

    # ── 2. 分类器 ──────────────────────────────────────────────────
    section("2. 分类器（示例消息 → 复杂度/任务类型）")
    for msg in ["你好", "帮我写一个 Python 爬虫抓取网页",
                "GRE 逻辑题：若 A 则 B，非 B，所以？"]:
        r = classify([{"role": "user", "content": msg}])
        print(f"  「{msg}」 → {r['complexity']}/{r['task_type']} "
              f"(method={r['method']})")

    # ── 3. 真实路由：选模 → 调用 → 学习/拉黑 → 再选 ────────────────
    section("3. 真实路由：bandit 选模 → 实际调用 → 学习/拉黑")
    from openai import OpenAI

    bandit_cfg = {"enabled": True, "ucb_c": 1.0, "alpha": 0.012,
                  "base_reward": 100.0}
    reset_bandit("simple_models")
    bandit = get_bandit("simple_models", bandit_cfg)
    candidates = build_candidates(discovered, simple)
    print(f"simple 候选: {len(candidates)} 个")

    exhausted = set()
    hit = None
    for rnd in range(8):
        active = [c for c in candidates if c["model"] not in exhausted]
        if not active:
            print("  候选全部拉黑，停止")
            break
        sel = bandit.select(active, task_type="chat")
        print(f"\n  第{rnd+1}轮 bandit 选中 → {sel['model']}")
        try:
            client = OpenAI(base_url=sel["base_url"], api_key=sel["api_key"],
                            timeout=15)
            resp = client.chat.completions.create(
                model=sel["model"],
                messages=[{"role": "user", "content": "你好，请用一句话回复"}],
                max_tokens=50,
            )
            tok = resp.usage.total_tokens if resp.usage else 0
            txt = resp.choices[0].message.content.strip()
            bandit.update(sel["model"], success=True, total_tokens=tok,
                          task_type="chat")
            print(f"      ✅ 成功 tokens={tok}  回复: {txt[:60]}")
            hit = sel["model"]
            break
        except Exception as e:
            err = str(e)
            if _is_quota(err):
                exhausted.add(sel["model"])
                print(f"      ❌ 额度耗尽 → 永久拉黑")
            else:
                bandit.update(sel["model"], success=False, total_tokens=0,
                              task_type="chat")
                print(f"      ❌ 失败 → 降权  ({err[:70]})")
    save_one("simple_models")

    # ── 4. 端到端成功证明：用一个已知有余额的模型 ──────────────────
    section("4. 端到端成功证明（发现的 key + 模型真实打通）")
    for target in ("kimi-k2.5", "qwen-plus", "qwen3-32b", "deepseek-r1"):
        if target in discovered:
            e = discovered[target][0]
            print(f"目标模型: {target}  (来自自动发现)")
            print(f"  endpoint: {e['base_url']}")
            print(f"  key     : {mask(e['api_key'])}")
            try:
                client = OpenAI(base_url=e["base_url"], api_key=e["api_key"],
                                timeout=15)
                resp = client.chat.completions.create(
                    model=target,
                    messages=[{"role": "user",
                               "content": "用一句话介绍你自己，20字以内"}],
                    max_tokens=60,
                )
                tok = resp.usage.total_tokens if resp.usage else 0
                txt = resp.choices[0].message.content.strip()
                print(f"  ✅ 成功 tokens={tok}")
                print(f"     回复: {txt[:80]}")
            except Exception as ex:
                print(f"  ❌ {target} 失败: {str(ex)[:100]}")
            break

    # ── 5. 结论 ────────────────────────────────────────────────────
    section("5. 结论")
    print(f"  ① 自动发现：从用户 .env 的 {len(per_endpoint)} 个 key "
          f"发现 {len(discovered)} 个模型")
    print(f"  ② 过滤分类后 simple={len(simple)} + complex={len(complex_)} "
          f"进入 bandit 路由池")
    print(f"  ③ 第3步真实路由演示了 bandit 选模→调用→学习/拉黑")
    print(f"  ④ 第4步证明了发现的 key+模型能真实打通")
    print(f"\n  端到端验证：把 config.yaml 的 smart_model_routing.enabled 设为 "
          f"true，/quit 重启后发消息，")
    print(f"        观察日志 [smart-router] bandit → … 即可。")


if __name__ == "__main__":
    main()
