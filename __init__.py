import atexit
import os
import sys
import time
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from config import load_router_config, load_default_model, get_api_key
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
    lower = model.lower()
    for family, prefixes in FAMILY_MAP.items():
        for p in prefixes:
            if lower.startswith(p.lower()):
                return family
    return model


def _normalize_provider_list(pcfg):
    """把 provider 条目统一为 list（单 dict → [dict]，list → 原样，空 → []）。"""
    if not pcfg:
        return []
    if isinstance(pcfg, dict):
        return [pcfg]
    if isinstance(pcfg, list):
        return [e for e in pcfg if isinstance(e, dict)]
    return []


def _resolve_entry_key(entry):
    """解析 entry 的 api_key：内联 api_key 优先，api_key_env 兜底。"""
    return entry.get("api_key") or get_api_key(entry.get("api_key_env", ""))


_QUOTA_MARKERS = (
    "insufficient_quota",
    "insufficient_balance",
    "allocationquota",     # 百炼 AllocationQuota.FreeTierOnly
    "quota exceeded",      # 百炼 Free allocated quota exceeded
    "freequota",
    "free tier",
)


def _is_quota_exhausted(err_str: str) -> bool:
    """判断错误是否属于「额度/余额耗尽」（临时，额度恢复可重试）。"""
    low = err_str.lower()
    return any(m in low for m in _QUOTA_MARKERS)


# 进程级永久拉黑：这些错误意味着模型在本次进程内永远不可用
# （404 未开通 / 400 不支持 tool call / 无访问权限），重启后重扫才可能恢复。
_PERMANENT_BLACKLIST = set()

_PERMANENT_BROKEN_MARKERS = (
    "access_denied",            # 403 无权限（非额度问题）
    "access denied",
    "modelnotopen",             # Ark 模型未开通
    "notfound",                 # InvalidEndpointOrModel.NotFound
    "not found",
    "not exist",                # Model not exist / does not exist
    "tool call is not supported",  # 蒸馏模型不支持 function calling
    "invalidparameter",         # 400 Algo.InvalidParameter（camelCase）
    "invalid_parameter",        # 400 code=invalid_parameter_error（enable_thinking / max_tokens 范围等）
)

REQUEST_TIMEOUT = 20.0  # 单次模型调用超时（秒），避免坏端点/长跑模型卡死


def _is_permanent_broken(err_str: str) -> bool:
    """判断错误是否属于「永久不可用」（404 未开通 / 400 不支持 / 无权限）。"""
    low = err_str.lower()
    return any(m in low for m in _PERMANENT_BROKEN_MARKERS)


def register(ctx):
    ctx.register_middleware("llm_execution", on_llm_execution)
    atexit.register(_save_bandits_on_exit)

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
    """放行给 Hermes 默认 provider，同时修正模型名。"""
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

    # ── 只路由每轮对话的第一次 LLM 调用：最后一条是 tool 结果 → 续调 ──
    messages = request.get("messages", [])
    if messages and messages[-1].get("role") == "tool":
        if _last_routed:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=_last_routed["base_url"],
                                api_key=_last_routed["api_key"],
                                timeout=REQUEST_TIMEOUT)
                modified = dict(request)
                modified["model"] = _last_routed["model"]
                modified.pop("stream", None)
                modified.pop("stream_options", None)
                modified.pop("enable_thinking", None)
                return client.chat.completions.create(**modified)
            except Exception:
                pass
        return _safe_pass_through(request, next_call)

    # ── 构建分类器配置：轻量模型优先，复杂模型兜底，组成 fallback 链 ──
    classifier_cfg = []
    simple_models = cfg.get("simple_models", [])
    complex_models = cfg.get("complex_models", [])
    providers = cfg.get("providers", {})
    complex_set = set(complex_models)

    def _classifier_priority(model_name):
        m = model_name.lower()
        if model_name in complex_set:
            return 3  # 复杂模型最后兜底（轻量全挂时才用）
        if "distill" in m:
            return 2  # 蒸馏推理模型（会输出推理链，JSON 解析易失败）
        for kw in ("flash", "lite", "turbo", "mini", "small", "air", "nano", "tiny"):
            if kw in m:
                return 0  # 轻量优先
        return 1  # 中档

    all_classifier_models = list(simple_models) + [m for m in complex_models
                                                   if m not in simple_models]
    for model_name in sorted(all_classifier_models, key=_classifier_priority):
        if model_name in _PERMANENT_BLACKLIST:
            continue
        for entry in _normalize_provider_list(providers.get(model_name)):
            api_key = _resolve_entry_key(entry)
            if entry.get("base_url") and api_key:
                classifier_cfg.append({
                    "base_url": entry["base_url"],
                    "api_key": api_key,
                    "model": model_name,
                })
                break

    # ── 分类 ──
    t0 = time.monotonic()
    result = classify(request.get("messages", []), classifier_cfg)
    latency_ms = int((time.monotonic() - t0) * 1000)

    complexity = result["complexity"]
    task_type = result["task_type"]

    method = result.get("method", "?")
    reasoning = result.get("reasoning", "?")
    print(f"[smart-router] classify: method={method} "
          f"→ {complexity}/{task_type} ({reasoning}) {latency_ms}ms",
          file=sys.stderr)

    user_texts = [m.get("content", "") for m in request.get("messages", [])
                  if m.get("role") == "user"]
    user_message = user_texts[-1] if user_texts else ""

    # ── 选池：task_type + complexity 联合决策 ──
    # coding/reasoning 强制 complex；chat/translation 强制 simple；
    # 其余（other/writing/analysis）交给 complexity 判定。
    FORCE_COMPLEX = {"coding", "reasoning"}
    FORCE_SIMPLE = {"chat", "translation"}

    if task_type in FORCE_COMPLEX:
        pool_key = "complex_models"
    elif task_type in FORCE_SIMPLE:
        pool_key = "simple_models"
    else:
        pool_key = "simple_models" if complexity == "simple" else "complex_models"

    providers = cfg.get("providers", {})
    model_list = cfg.get(pool_key, [])

    candidates = []
    for model_name in model_list:
        if model_name in _PERMANENT_BLACKLIST:
            continue
        for entry in _normalize_provider_list(providers.get(model_name)):
            api_key = _resolve_entry_key(entry)
            if not api_key or not entry.get("base_url"):
                continue
            candidates.append({
                "model": model_name,
                "base_url": entry["base_url"],
                "api_key": api_key,
                "api_key_env": entry.get("api_key_env", ""),
            })

    if not candidates:
        return _safe_pass_through(request, next_call)

    # ── Phase 2: UCB 老虎机选模型（若启用）──
    bandit_cfg = cfg.get("bandit", {})
    use_bandit = bandit_cfg.get("enabled", False)

    exhausted_models = set()   # 额度耗尽，进程生命周期永久拉黑
    round_blacklist = set()    # 临时故障，仅本轮拉黑
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
                                    api_key=selected["api_key"],
                                    timeout=REQUEST_TIMEOUT)
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

                    routed_tier = "complex" if pool_key == "complex_models" else "simple"
                    get_state().update_after_turn(routed_tier, topic=task_type)

                    log_classification(
                        user_message=user_message,
                        complexity=complexity,
                        task_type=task_type,
                        confidence=result["confidence"],
                        reasoning=result.get("reasoning", ""),
                        method=result["method"],
                        latency_ms=latency_ms,
                        model_routed_to=selected["model"],
                        model_actual=getattr(response, "model", selected["model"]),
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

                    if _is_quota_exhausted(err_str):
                        exhausted_models.add(selected["model"])
                        bandit.update(selected["model"], success=False,
                                      total_tokens=0, task_type=task_type)
                        print(f"[smart-router] quota exhausted for "
                              f"{selected['model']}, down-ranked (本轮跳过，"
                              f"后续轮次/下次额度恢复会重试)", file=sys.stderr)
                    elif _is_permanent_broken(err_str):
                        _PERMANENT_BLACKLIST.add(selected["model"])
                        print(f"[smart-router] {selected['model']} permanently "
                              f"broken (404/400/access denied), blacklisted",
                              file=sys.stderr)
                    elif any(s in err_str for s in
                             ("429", "500", "502", "503", "504")):
                        round_blacklist.add(selected["model"])
                        print(f"[smart-router] server error for "
                              f"{selected['model']}, skipping this round",
                              file=sys.stderr)
                    else:
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
            client = OpenAI(base_url=c["base_url"], api_key=c["api_key"],
                            timeout=REQUEST_TIMEOUT)
            modified = dict(request)
            modified["model"] = c["model"]
            modified.pop("stream", None)
            modified.pop("stream_options", None)
            modified.pop("enable_thinking", None)
            print(f"[smart-router] fallback {pool_key}({task_type}) → {c['model']}",
                  file=sys.stderr)
            response = client.chat.completions.create(**modified)

            if use_bandit:
                total_tokens = response.usage.total_tokens if response.usage else 0
                bandit.update(c["model"], success=True,
                              total_tokens=total_tokens, task_type=task_type)
                save_one(pool_key)

            routed_tier = "complex" if pool_key == "complex_models" else "simple"
            get_state().update_after_turn(routed_tier, topic=task_type)

            log_classification(
                user_message=user_message,
                complexity=complexity,
                task_type=task_type,
                confidence=result["confidence"],
                reasoning=result.get("reasoning", ""),
                method=result["method"],
                latency_ms=latency_ms,
                model_routed_to=c["model"],
                model_actual=getattr(response, "model", c["model"]),
                routing_success=True,
            )

            _last_routed = {"model": c["model"], "base_url": c["base_url"],
                            "api_key": c["api_key"]}
            return response
        except Exception as e:
            err_str = str(e)
            print(f"[smart-router] {c['model']} failed: {e}", file=sys.stderr)

            if _is_quota_exhausted(err_str):
                exhausted_models.add(c["model"])
                if use_bandit:
                    bandit.update(c["model"], success=False, total_tokens=0,
                                  task_type=task_type)
                print(f"[smart-router] quota exhausted for {c['model']}, "
                      f"down-ranked (本轮跳过，后续轮次/下次额度恢复会重试)",
                      file=sys.stderr)
            elif _is_permanent_broken(err_str):
                _PERMANENT_BLACKLIST.add(c["model"])
                print(f"[smart-router] {c['model']} permanently broken "
                      f"(404/400/access denied), blacklisted", file=sys.stderr)
            elif any(s in err_str for s in
                     ("429", "500", "502", "503", "504")):
                round_blacklist.add(c["model"])
                print(f"[smart-router] server error for {c['model']}, "
                      f"skipping this round", file=sys.stderr)
            else:
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
