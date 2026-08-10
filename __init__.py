import atexit
import os
import sys
import time
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from config import load_router_config, load_default_model
from classifier import classify, get_state
from data_logger import log_classification

# 记录最近一次路由成功的模型配置，用于 tool 续调
_last_routed = None

# ── 家族映射表：用于 bandit 冷启动先验 ──
FAMILY_MAP = {
    "qwen":     ["qwen", "qwen2", "qwen2.5", "qwen3", "qwen3.5", "qwen3.6", "qwen3.7", "qwq"],
    "deepseek": ["deepseek"],
    "glm":      ["glm", "chatglm"],
}


def _get_family(model: str) -> str:
    """返回模型所属家族名，未匹配返回自身。"""
    lower = model.lower()
    for family, prefixes in FAMILY_MAP.items():
        for p in prefixes:
            if lower.startswith(p.lower()):
                return family
    return model


def register(ctx):
    ctx.register_middleware("llm_execution", on_llm_execution)
    atexit.register(_save_bandits_on_exit)

    # 注入家族映射到 bandit 模块
    try:
        from bandit import set_family_fn
        set_family_fn(_get_family)
    except Exception:
        pass


def _save_bandits_on_exit():
    try:
        from bandit import save_all
        save_all()
    except Exception:
        pass


def _safe_pass_through(request, next_call):
    """放行给 Hermes 默认 provider，同时修正模型名。

    request["model"] 可能已被 Hermes core 同步为 MaaS 模型名
    （如 deepseek-v4-flash-0731），但 Hermes 默认 provider 不认识。
    需从 config.yaml 读回正确默认模型。
    """
    global _last_routed
    _last_routed = None
    default_model = load_default_model()
    if default_model and default_model != request.get("model", ""):
        fallback_request = dict(request)
        fallback_request["model"] = default_model
        print(f"[smart-router] normalizing model: "
              f"{request.get('model')} → {default_model}", file=sys.stderr)
        return next_call(fallback_request)
    return next_call(request)


def on_llm_execution(request, next_call, **context):
    global _last_routed
    cfg = load_router_config()
    if not cfg or not cfg.get("enabled"):
        return _safe_pass_through(request, next_call)

    # ── 只路由每轮对话的第一次 LLM 调用：如果最后一条消息是 tool 结果，说明是续调 ──
    messages = request.get("messages", [])
    if messages and messages[-1].get("role") == "tool":
        # tool 续调：用上次路由成功的 MaaS 配置直接调，不经过 Hermes provider
        if _last_routed:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=_last_routed["base_url"],
                                api_key=_last_routed["api_key"])
                modified = dict(request)
                modified["model"] = _last_routed["model"]
                modified.pop("stream", None)
                modified.pop("stream_options", None)
                modified.pop("enable_thinking", None)
                return client.chat.completions.create(**modified)
            except Exception:
                pass
        return _safe_pass_through(request, next_call)

    # ── 构建分类器配置：取 simple_models 第一个作为分类模型 ──
    classifier_cfg = None
    simple_models = cfg.get("simple_models", [])
    providers = cfg.get("providers", {})
    if simple_models:
        classifier_model = simple_models[0]
        pcfg = providers.get(classifier_model, {})
        api_key = os.environ.get(pcfg.get("api_key_env", ""), "")
        if pcfg.get("base_url") and api_key:
            classifier_cfg = {
                "base_url": pcfg["base_url"],
                "api_key": api_key,
                "model": classifier_model,
            }

    # ── 分类 ──
    t0 = time.monotonic()
    result = classify(request.get("messages", []), classifier_cfg)
    latency_ms = int((time.monotonic() - t0) * 1000)

    complexity = result["complexity"]
    task_type = result["task_type"]

    # ── 实时分类日志 ──
    method = result.get("method", "?")
    reasoning = result.get("reasoning", "?")
    print(f"[smart-router] classify: method={method} "
          f"→ {complexity}/{task_type} "
          f"({reasoning}) {latency_ms}ms",
          file=sys.stderr)

    # ── 提取用户消息（供后续日志使用）──
    user_texts = [m.get("content", "") for m in request.get("messages", [])
                  if m.get("role") == "user"]
    user_message = user_texts[-1] if user_texts else ""

    # ── 选池：task_type + complexity 联合决策 ──
    FORCE_COMPLEX = {"coding", "reasoning"}
    FORCE_SIMPLE = {"chat", "translation", "other"}

    if task_type in FORCE_COMPLEX:
        pool_key = "complex_models"
    elif task_type in FORCE_SIMPLE:
        pool_key = "simple_models"
    else:  # writing, analysis
        pool_key = "simple_models" if complexity == "simple" else "complex_models"

    providers = cfg.get("providers", {})
    model_list = cfg.get(pool_key, [])

    candidates = []
    for model_name in model_list:
        pcfg = providers.get(model_name)
        if not pcfg:
            continue
        api_key = os.environ.get(pcfg.get("api_key_env", ""), "")
        if not api_key:
            continue
        candidates.append({
            "model": model_name,
            "base_url": pcfg["base_url"],
            "api_key": api_key,
        })

    if not candidates:
        return _safe_pass_through(request, next_call)

    # ── Phase 2: UCB 老虎机选模型（若启用）──
    bandit_cfg = cfg.get("bandit", {})
    use_bandit = bandit_cfg.get("enabled", False)

    exhausted_models = set()   # 402/403 额度耗尽，进程生命周期永久拉黑
    round_blacklist = set()    # 429/5xx 临时故障，仅本轮拉黑
    tried_models = set()

    if use_bandit:
        from bandit import get_bandit, save_one

        bandit = get_bandit(pool_key, bandit_cfg)

        active = [c for c in candidates
                  if c["model"] not in exhausted_models
                  and c["model"] not in round_blacklist]
        if active:
            selected = bandit.select(active, task_type=task_type)
            if selected:
                tried_models.add(selected["model"])
                try:
                    from openai import OpenAI
                    client = OpenAI(base_url=selected["base_url"],
                                    api_key=selected["api_key"])
                    modified = dict(request)
                    modified["model"] = selected["model"]
                    modified.pop("stream", None)
                    modified.pop("stream_options", None)
                    modified.pop("enable_thinking", None)
                    print(f"[smart-router] bandit → {pool_key}({task_type}) → "
                          f"{selected['model']}", file=sys.stderr)
                    response = client.chat.completions.create(**modified)

                    total_tokens = response.usage.total_tokens if response.usage else 0
                    bandit.update(selected["model"], success=True,
                                  total_tokens=total_tokens, task_type=task_type)
                    save_one(pool_key)

                    # ── 更新会话状态 — 用于下一轮的 R1 上下文继承 ──
                    routed_tier = "complex" if pool_key == "complex_models" else "simple"
                    get_state().update_after_turn(routed_tier, topic=task_type)

                    # ── 路由成功落盘 ──
                    log_classification(
                        user_message=user_message,
                        complexity=complexity,
                        task_type=task_type,
                        confidence=result["confidence"],
                        reasoning=result.get("reasoning", ""),
                        method=result["method"],
                        latency_ms=latency_ms,
                        model_routed_to=selected["model"],
                        routing_success=True,
                    )

                    _last_routed = {"model": selected["model"],
                                    "base_url": selected["base_url"],
                                    "api_key": selected["api_key"]}
                    return response

                except Exception as e:
                    err_str = str(e)
                    print(f"[smart-router] bandit {selected['model']} failed: {e}",
                          file=sys.stderr)

                    if (("403" in err_str and "insufficient_quota" in err_str)
                            or ("402" in err_str and "insufficient_balance"
                                in err_str.lower())):
                        # 额度耗尽：永久拉黑，不污染 bandit
                        exhausted_models.add(selected["model"])
                        print(f"[smart-router] quota/balance exhausted for "
                              f"{selected['model']}, skipping permanently",
                              file=sys.stderr)
                    elif any(s in err_str for s in
                             ("400", "429", "500", "502", "503", "504")):
                        # 服务端临时故障：仅本轮拉黑，不污染 bandit
                        round_blacklist.add(selected["model"])
                        print(f"[smart-router] server error for "
                              f"{selected['model']}, skipping this round",
                              file=sys.stderr)
                    else:
                        # 真正的模型问题：更新 bandit
                        bandit.update(selected["model"], success=False,
                                      total_tokens=0, task_type=task_type)
                        save_one(pool_key)

    # ── 顺序 fallback：逐个尝试未试过的候选 ──
    for c in candidates:
        if c["model"] in exhausted_models:
            print(f"[smart-router] skipping {c['model']} — quota exhausted",
                  file=sys.stderr)
            continue
        if c["model"] in round_blacklist:
            print(f"[smart-router] skipping {c['model']} — server error this round",
                  file=sys.stderr)
            continue
        if c["model"] in tried_models:
            continue

        try:
            from openai import OpenAI
            client = OpenAI(base_url=c["base_url"], api_key=c["api_key"])
            modified = dict(request)
            modified["model"] = c["model"]
            modified.pop("stream", None)
            modified.pop("stream_options", None)
            modified.pop("enable_thinking", None)
            print(f"[smart-router] fallback {pool_key}({task_type}) → {c['model']}",
                  file=sys.stderr)
            response = client.chat.completions.create(**modified)

            # fallback 成功也更新 bandit（如果启用）
            if use_bandit:
                total_tokens = response.usage.total_tokens if response.usage else 0
                bandit.update(c["model"], success=True,
                              total_tokens=total_tokens, task_type=task_type)
                save_one(pool_key)

            # ── 更新会话状态 — 用于下一轮的 R1 上下文继承 ──
            routed_tier = "complex" if pool_key == "complex_models" else "simple"
            get_state().update_after_turn(routed_tier, topic=task_type)

            # ── 路由成功落盘 ──
            log_classification(
                user_message=user_message,
                complexity=complexity,
                task_type=task_type,
                confidence=result["confidence"],
                reasoning=result.get("reasoning", ""),
                method=result["method"],
                latency_ms=latency_ms,
                model_routed_to=c["model"],
                routing_success=True,
            )

            _last_routed = {"model": c["model"], "base_url": c["base_url"],
                            "api_key": c["api_key"]}
            return response
        except Exception as e:
            err_str = str(e)
            print(f"[smart-router] {c['model']} failed: {e}", file=sys.stderr)

            if (("403" in err_str and "insufficient_quota" in err_str)
                    or ("402" in err_str and "insufficient_balance"
                        in err_str.lower())):
                # 额度耗尽：永久拉黑，不污染 bandit
                exhausted_models.add(c["model"])
                print(f"[smart-router] quota/balance exhausted for {c['model']}, "
                      f"skipping permanently", file=sys.stderr)
            elif any(s in err_str for s in
                     ("400", "429", "500", "502", "503", "504")):
                # 服务端临时故障：仅本轮拉黑，不污染 bandit
                round_blacklist.add(c["model"])
                print(f"[smart-router] server error for {c['model']}, "
                      f"skipping this round", file=sys.stderr)
            else:
                # 真正的模型问题：更新 bandit
                if use_bandit:
                    bandit.update(c["model"], success=False, total_tokens=0,
                                  task_type=task_type)
                    save_one(pool_key)
            continue

    # ── 全部候选失败落盘 ──
    log_classification(
        user_message=user_message,
        complexity=complexity,
        task_type=task_type,
        confidence=result["confidence"],
        reasoning=result.get("reasoning", ""),
        method=result["method"],
        latency_ms=latency_ms,
        model_routed_to=candidates[0]["model"] if candidates else None,
        routing_success=False,
    )

    return _safe_pass_through(request, next_call)
