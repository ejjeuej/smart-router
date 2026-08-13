#!/usr/bin/env python3
"""探测池子里每个模型是否可用（发最小请求，标 ✅/❌）。

用法：
  python3 probe_models.py            # 探测全部（238 个，约 1-2 分钟）
  python3 probe_models.py simple     # 只探测简单池
  python3 probe_models.py complex    # 只探测复杂池
  python3 probe_models.py -w 16      # 指定并发数（默认 16）

注意：探测请求带 tools + max_tokens=32768，尽量贴近 Hermes 真实调用，
所以「不支持 tool call / max_tokens 范围」这类 400 也会被如实标出。
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

_TOOL = [{
    "type": "function",
    "function": {
        "name": "echo",
        "description": "echo a string",
        "parameters": {"type": "object", "properties": {"s": {"type": "string"}}},
    },
}]


def classify_error(err_str: str) -> str:
    low = err_str.lower()
    if ("insufficient_quota" in low or "free quota" in low
            or "allocated quota" in low or "freetieronly" in low
            or "exhausted" in low or "insufficient_balance" in low):
        return "❌ 403 额度耗尽"
    if ("notfound" in low or "not found" in low or "not exist" in low
            or "modelnotopen" in low or "model not open" in low):
        return "❌ 404 未开通"
    if ("tool call" in low or "invalid_parameter" in low or "invalidparameter" in low
            or "max_tokens" in low or "enable_thinking" in low
            or "input length" in low):
        return "❌ 400 参数不兼容"
    if "access" in low and "denied" in low:
        return "❌ 403 无权限"
    if "timeout" in low or "timed out" in low:
        return "⏱ 超时"
    return "❌ 其他错误"


def probe_one(model, entry):
    base_url = entry.get("base_url", "")
    api_key = entry.get("api_key", "")
    if not base_url or not api_key:
        return model, "❌ 无 key", ""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key,
                        timeout=8.0, max_retries=0)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            tools=_TOOL,
            max_tokens=32768,
            temperature=0,
        )
        return model, "✅ 可用", ""
    except Exception as e:
        err = str(e)
        return model, classify_error(err), err[:90]


def short_label(base_url):
    host = (base_url or "").split("://")[-1].split("/")[0]
    if "maas.aliyuncs.com" in host:
        return "MaaS(百炼)"
    if "volces.com" in host or "ark" in host:
        return "Ark(火山方舟)"
    if "deepseek.com" in host:
        return "DeepSeek"
    if "moonshot" in host:
        return "Moonshot(Kimi)"
    return host


def main():
    workers = 16
    argv = sys.argv[1:]
    tier = "all"
    for i, a in enumerate(argv):
        if a in ("simple", "complex", "all"):
            tier = a
        elif a == "-w" and i + 1 < len(argv):
            workers = int(argv[i + 1])

    cfg = config.load_router_config(force=True)
    providers = cfg.get("providers", {})
    simple = cfg.get("simple_models", [])
    complex_ = cfg.get("complex_models", [])

    models = []
    if tier in ("simple", "all"):
        models += simple
    if tier in ("complex", "all"):
        models += complex_

    print(f"探测 {len(models)} 个模型（并发 {workers}，每请求超时 8s）...\n", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for m in models:
            es = providers.get(m, [])
            entry = es[0] if es else {}
            futs[ex.submit(probe_one, m, entry)] = m
        for fut in as_completed(futs):
            m, status, detail = fut.result()
            results[m] = (status, detail)
            line = f"  {status}  {m}"
            if detail:
                line += f"   [{detail}]"
            print(line, flush=True)

    # ── 汇总：按状态分组 ──
    print(f"\n{'=' * 64}")
    print("汇总（按状态分组）")
    print(f"{'=' * 64}")
    by_status = defaultdict(list)
    for m, (status, _) in results.items():
        by_status[status].append(m)
    for status in sorted(by_status):
        ms = by_status[status]
        print(f"\n[{status}] × {len(ms)}")

    # ── 可用模型按 provider 分组 ──
    ok = [m for m, (s, _) in results.items() if s.startswith("✅")]
    print(f"\n{'=' * 64}")
    print(f"✅ 可用模型（{len(ok)} 个）按 provider：")
    print(f"{'=' * 64}")
    by_prov = defaultdict(list)
    for m in ok:
        es = providers.get(m, [])
        label = short_label(es[0].get("base_url", "")) if es else "?"
        by_prov[label].append(m)
    for label in sorted(by_prov):
        print(f"\n  [{len(by_prov[label])}] {label}")
        for m in by_prov[label]:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
