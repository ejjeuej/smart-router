#!/usr/bin/env python3
"""调试用：打印 smart-router 的简单池 / 复杂池模型清单（按 provider 分组）。

用法：
  python dump_pools.py            # 打印两个池的全部模型
  python dump_pools.py simple     # 只看简单池
  python dump_pools.py complex    # 只看复杂池
  python dump_pools.py counts     # 只看各 provider 的模型数量
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def short_label(base_url: str) -> str:
    """把冗长 base_url 缩成可读的 provider 标签。"""
    host = (base_url or "").split("://")[-1].split("/")[0]
    if "maas.aliyuncs.com" in host:
        return "MaaS(百炼)"
    if "volces.com" in host or "ark" in host:
        return "Ark(火山方舟)"
    if "deepseek.com" in host:
        return "DeepSeek"
    if "moonshot" in host:
        return "Moonshot(Kimi)"
    if "dashscope" in host:
        return "DashScope"
    if "openai.com" in host:
        return "OpenAI"
    return host or "(无 base_url)"


def endpoint_label(model: str, providers: dict) -> str:
    es = providers.get(model, [])
    if not es:
        return "(无 provider)"
    return short_label(es[0].get("base_url", ""))


def print_pool(title: str, models: list, providers: dict) -> None:
    print(f"\n{'=' * 68}")
    print(f"{title}（{len(models)} 个）")
    print(f"{'=' * 68}")
    groups = defaultdict(list)
    for m in models:
        groups[endpoint_label(m, providers)].append(m)
    for label in sorted(groups):
        ms = groups[label]
        print(f"\n  [{len(ms)}] {label}")
        for m in ms:
            print(f"    - {m}")


def print_counts(models: list, providers: dict) -> None:
    groups = defaultdict(int)
    for m in models:
        groups[endpoint_label(m, providers)] += 1
    for label in sorted(groups, key=lambda x: -groups[x]):
        print(f"  {groups[label]:>4}  {label}")


def main() -> None:
    # force=True：即使 smart_model_routing.enabled=false 也照常发现（调试用）
    cfg = config.load_router_config(force=True)
    if not cfg.get("providers"):
        print("未发现任何模型（检查 .env key / custom_providers）", file=sys.stderr)
        return

    providers = cfg.get("providers", {})
    simple = cfg.get("simple_models", [])
    complex_ = cfg.get("complex_models", [])

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"总计: {len(simple)} simple + {len(complex_)} complex")

    if arg == "counts":
        print("\n简单池 simple_models 按 provider：")
        print_counts(simple, providers)
        print("\n复杂池 complex_models 按 provider：")
        print_counts(complex_, providers)
        return

    if arg in ("simple", "all"):
        print_pool("简单池 simple_models", simple, providers)
    if arg in ("complex", "all"):
        print_pool("复杂池 complex_models", complex_, providers)


if __name__ == "__main__":
    main()
