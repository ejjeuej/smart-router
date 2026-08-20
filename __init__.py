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
from logger import debug, error, info

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
        info(f"normalizing model: "
              f"{request.get('model')} → {default_model}")
        return next_call(fallback_request)
    return next_call(request)


def _annotate_response(response, cfg, model_routed, complexity, task_type,
                       method, latency_ms, pool_key):
    """在回复末尾追加路由脚注，让用户在对话界面直接看到路由结果。

    开关：config.yaml smart_model_routing.announce（默认 true）。
    仅注入有文本内容的成功响应；纯 tool_calls（无文本）不注入。
    主路径与 tool 续调路径都会调用；任何异常静默跳过，绝不影响路由主流程。
    """
    try:
        if not cfg.get("announce", True):
            return response
        if not (response and getattr(response, "choices", None)):
            return response
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if not content:
            return response  # 纯 tool_calls 等无文本响应，不注入
        actual = getattr(response, "model", None) or model_routed
        tier = "complex" if pool_key == "complex_models" else "simple"
        note = (f"\n\n────\n[smart-router] 已路由至 {actual}"
                f" · {tier}/{task_type} · {method} · {latency_ms}ms")
        msg.content = content + note
    except Exception as e:
        info(f"annotate response failed: {e}")
    return response


def _inject_tier_hint(request: dict, tier: str) -> dict:
    """借鉴点 22 — LLM 档位感知提示注入。

    把 get_tier_hint()（三态：首次/同档/换档）拼进 messages：
    - 已有 system 消息 → 追加到其 content 末尾
    - 没有 → 在最前插入一条新 system 消息
    与 _annotate_response（给用户看的脚注）分离——这个给模型看。
    返回新 dict，不修改原 request。任何异常静默返回原 request（不阻断路由）。
    """
    try:
        from classifier import get_tier_hint
        hint = get_tier_hint(tier)
        if not hint:
            return request
        msgs = []
        injected = False
        for m in request.get("messages", []):
            if not injected and m.get("role") == "system":
                m2 = dict(m)
                m2["content"] = str(m.get("content", "")) + "\n\n" + hint
                msgs.append(m2)
                injected = True
            else:
                msgs.append(m)
        if not injected:
            msgs.insert(0, {"role": "system", "content": hint})
        modified = dict(request)
        modified["messages"] = msgs
        return modified
    except Exception as e:
        info(f"inject tier hint failed: {e}")
        return request


# ── 思考参数补注 ──────────────────────────────────────────────────────
# Hermes 的 thinking 注入发生在 transport 层 build_kwargs（中间件之后）：
# 主循环默认 reasoning_config = {enabled: true, effort: medium}，到 transport
# 才转成 enable_thinking / extra_body.thinking 等 wire 参数。插件 llm_execution
# 中间件自己直连上游，把这个注入绕过了 → 上游默认不开思考 → 响应没有
# reasoning_content → 前端不显示思考过程（即使"透传"也无效，因为请求里
# 根本没有这个字段）。
# 修复：转发时对"能思考的模型"主动补开思考的 wire 参数。判断分两层：
#   ① 模型家族 —— 所有已知默认支持思考的家族（qwen/glm/kimi/deepseek/
#      o 系/claude/gemini…），与 config.py 的 _FAMILY_TIER（复杂度分级）
#      是两个独立维度：simple 池的 glm-5.2-fast-preview 同样能思考。
#   ② 上游平台 —— 决定参数名：智谱/Kimi 官方接口用 thinking，
#      百炼/OpenAI 兼容统一用 enable_thinking。
# request 里已有 thinking 相关字段则透传不动（尊重 Hermes /thinkon
# /thinkoff 与 reasoning_effort 等显式设置）。注意 extra_body 不在此列：
# 它可能是其他用途的透传通道，注入时合并而非阻断（内部已有思考键才阻断）。
_THINKING_CTL_KEYS = (
    "enable_thinking", "thinking", "thinking_config",
    "reasoning_effort", "reasoning",
)
# extra_body 内部出现这些键 = 已有显式思考控制，同样透传不动
_THINKING_EB_KEYS = ("enable_thinking", "thinking", "thinking_config")

# 已知"默认支持思考"的模型家族（启发式，按模型名子串匹配，防厂商前缀
# 如 zhipu-glm4 / alibaba-qwen 与日期后缀如 -2026-02-23 的干扰）。
# 命中 → 请求里无 thinking 控制键时主动补开思考参数。
_THINKING_FAMILIES = (
    # 阿里 / 通义（百炼 enable_thinking）
    "qwen", "qwq", "qvq",
    # DeepSeek（reasoner 系默认思考；v3/v4 系可开）
    "deepseek",
    # 智谱 GLM（z1/5 系思考，官方接口参数 thinking）
    "glm", "zhipu",
    # Kimi / Moonshot（K2 系思考，官方接口参数 thinking）
    "kimi", "moonshot",
    # OpenAI o 系 / gpt-5（reasoning_effort，通常已由 Hermes 显式携带）
    "o1", "o3", "o4", "gpt-5",
    # Anthropic（extra_body.thinking）
    "claude",
    # Google（thinking_config）
    "gemini",
)


def _thinking_params_for(model: str, base_url: str) -> dict:
    """按模型家族 + 上游平台返回应补的思考开关参数（不适用则空 dict）。"""
    low = str(model or "").lower()
    if not any(f in low for f in _THINKING_FAMILIES) \
            and "thinking" not in low and "reasoner" not in low:
        return {}  # 非思考系家族，不补（避免不支持参数的上游 400）
    url = str(base_url or "").lower()
    # 平台专用参数名：智谱官方 / Kimi 官方用 thinking；
    # 百炼/OpenAI 兼容统一用 enable_thinking（未知平台按 OpenAI 兼容处理）。
    if "open.bigmodel.cn" in url or "moonshot.cn" in url:
        return {"thinking": True}
    return {"enable_thinking": True}


def _ensure_thinking(modified: dict, model: str, base_url: str = "") -> dict:
    """确保转发请求携带思考开关（对齐 Hermes transport 层的默认注入）。

    注入通道必须是 extra_body：enable_thinking / thinking 都是平台私有参数，
    OpenAI SDK 的 create() 对未知顶层 kwarg 直接抛 TypeError（端到端评测
    实测踩坑），extra_body 会被 SDK 原样合并进请求 JSON body —— 这也是
    DashScope 官方文档给的接法。
    """
    try:
        if any(k in modified for k in _THINKING_CTL_KEYS):
            return modified  # 已有显式控制，透传不动
        params = _thinking_params_for(model, base_url)
        if params:
            eb = dict(modified.get("extra_body") or {})
            if any(k in eb for k in _THINKING_EB_KEYS):
                return modified  # extra_body 内已有思考控制，透传不动
            m = dict(modified)
            eb.update(params)
            m["extra_body"] = eb
            return m
    except Exception:
        pass
    return modified


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
                # Hermes 的 thinking 注入在 transport 层被插件绕过，
                # 这里按模型家族+平台补注思考开关（qwen/glm/kimi/deepseek 等）
                modified = _ensure_thinking(
                    modified, _last_routed["model"], _last_routed["base_url"])
                _t_call = time.monotonic()
                response = client.chat.completions.create(**modified)
                _latency_call_ms = int((time.monotonic() - _t_call) * 1000)
                return _annotate_response(
                    response, cfg, _last_routed["model"],
                    _last_routed.get("complexity", "?"),
                    _last_routed.get("task_type", "?"),
                    _last_routed.get("method", "?"),
                    _latency_call_ms,
                    _last_routed.get("pool_key", "simple_models"))
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
    info(f"classify: method={method} "
          f"→ {complexity}/{task_type} ({reasoning}) {latency_ms}ms")

    user_texts = []
    for m in request.get("messages", []):
        if m.get("role") != "user" or not isinstance(m.get("content"), str):
            continue
        content = m["content"].strip()
        if content.startswith("[System:"):
            end = content.find("]")
            if end == -1:
                continue
            content = content[end + 1:].strip()
        if content:
            user_texts.append(content)
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
            # 借鉴点 32 接线：用户明说"省钱/随便/便宜点" →
            # classifier 返回 cost_mode="cheap" → bandit 本轮临时抬高 λ_c，
            # 在池内显著偏向便宜臂（只影响本轮打分，不落盘、不改学习状态）。
            selected = bandit.select(
                active, task_type=task_type,
                cheap=result.get("cost_mode") == "cheap")
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
                    # Hermes 的 thinking 注入在 transport 层被插件绕过，
                    # 这里按模型家族+平台补注思考开关（qwen/glm/kimi/deepseek 等）
                    modified = _ensure_thinking(
                        modified, selected["model"], selected["base_url"])
                    # 借鉴点 22: 档位感知提示（给 LLM 看，与 announce 分离）
                    modified = _inject_tier_hint(
                        modified, "complex" if pool_key == "complex_models" else "simple")
                    info(f"bandit → {pool_key}({task_type}) → "
                          f"{selected['model']}")
                    _t_call = time.monotonic()
                    response = client.chat.completions.create(**modified)
                    _latency_call_ms = int((time.monotonic() - _t_call) * 1000)

                    total_tokens = response.usage.total_tokens if response.usage else 0
                    bandit.update(selected["model"], success=True,
                                  total_tokens=total_tokens, task_type=task_type,
                                  latency_ms=_latency_call_ms)
                    save_one(pool_key)

                    routed_tier = "complex" if pool_key == "complex_models" else "simple"
                    get_state().update_after_turn(routed_tier, topic=task_type)

                    log_classification(
                        user_message=user_message,
                        complexity=complexity,
                        task_type=task_type,
                        reasoning=result.get("reasoning", ""),
                        method=result["method"],
                        latency_ms=latency_ms,
                        model_routed_to=selected["model"],
                        model_actual=getattr(response, "model", selected["model"]),
                        routing_success=True,
                    )

                    _last_routed = {"model": selected["model"],
                                    "base_url": selected["base_url"],
                                    "api_key": selected["api_key"],
                                    "complexity": complexity,
                                    "task_type": task_type,
                                    "method": method,
                                    "pool_key": pool_key}
                    return _annotate_response(
                        response, cfg, selected["model"], complexity,
                        task_type, method, latency_ms, pool_key)

                except Exception as e:
                    err_str = str(e)
                    info(f"bandit {selected['model']} failed: {e}")

                    if _is_quota_exhausted(err_str):
                        exhausted_models.add(selected["model"])
                        bandit.update(selected["model"], success=False,
                                      total_tokens=0, task_type=task_type,
                                      penalty=0.5)  # 借鉴点50: 软惩罚, 恢复后回血
                        info(f"quota exhausted for "
                              f"{selected['model']}, down-ranked (本轮跳过，"
                              f"后续轮次/下次额度恢复会重试)")
                    elif _is_permanent_broken(err_str):
                        _PERMANENT_BLACKLIST.add(selected["model"])
                        info(f"{selected['model']} permanently "
                              f"broken (404/400/access denied), blacklisted")
                    elif any(s in err_str for s in
                             ("429", "500", "502", "503", "504")):
                        round_blacklist.add(selected["model"])
                        bandit.update(selected["model"], success=False,
                                      total_tokens=0, task_type=task_type,
                                      penalty=0.3)  # 借鉴点50: 故障期降权, 可自动恢复
                        info(f"server error for "
                              f"{selected['model']}, skipping this round")
                    else:
                        bandit.update(selected["model"], success=False,
                                      total_tokens=0, task_type=task_type)
                        save_one(pool_key)

    # ── 顺序 fallback：逐个尝试未试过的候选 ──
    fallback_list = candidates
    if use_bandit and result.get("cost_mode") == "cheap":
        # cheap 意图：fallback 也按估计成本升序，先试便宜的（bandit 已
        # 在 use_bandit 分支内创建，此处安全引用；未启用 bandit 时无价格
        # 信号，保持原白名单顺序）。
        fallback_list = sorted(candidates,
                               key=lambda c: bandit._est_cost(c["model"]))
    for c in fallback_list:
        if c["model"] in exhausted_models:
            info(f"skipping {c['model']} — quota exhausted")
            continue
        if c["model"] in round_blacklist:
            info(f"skipping {c['model']} — server error this round")
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
            # Hermes 的 thinking 注入在 transport 层被插件绕过，
            # 这里按模型家族+平台补注思考开关（qwen/glm/kimi/deepseek 等）
            modified = _ensure_thinking(
                modified, c["model"], c["base_url"])
            # 借鉴点 22: 档位感知提示（给 LLM 看，与 announce 分离）
            modified = _inject_tier_hint(
                modified, "complex" if pool_key == "complex_models" else "simple")
            info(f"fallback {pool_key}({task_type}) → {c['model']}")
            _t_call = time.monotonic()
            response = client.chat.completions.create(**modified)
            _latency_call_ms = int((time.monotonic() - _t_call) * 1000)

            if use_bandit:
                total_tokens = response.usage.total_tokens if response.usage else 0
                bandit.update(c["model"], success=True,
                              total_tokens=total_tokens, task_type=task_type,
                              latency_ms=_latency_call_ms)
                save_one(pool_key)

            routed_tier = "complex" if pool_key == "complex_models" else "simple"
            get_state().update_after_turn(routed_tier, topic=task_type)

            log_classification(
                user_message=user_message,
                complexity=complexity,
                task_type=task_type,
                reasoning=result.get("reasoning", ""),
                method=result["method"],
                latency_ms=latency_ms,
                model_routed_to=c["model"],
                model_actual=getattr(response, "model", c["model"]),
                routing_success=True,
            )

            _last_routed = {"model": c["model"], "base_url": c["base_url"],
                            "api_key": c["api_key"],
                            "complexity": complexity,
                            "task_type": task_type,
                            "method": method,
                            "pool_key": pool_key}
            return _annotate_response(
                response, cfg, c["model"], complexity,
                task_type, method, latency_ms, pool_key)
        except Exception as e:
            err_str = str(e)
            info(f"{c['model']} failed: {e}")

            if _is_quota_exhausted(err_str):
                exhausted_models.add(c["model"])
                if use_bandit:
                    bandit.update(c["model"], success=False, total_tokens=0,
                                  task_type=task_type, penalty=0.5)  # 借鉴点50
                info(f"quota exhausted for {c['model']}, "
                      f"down-ranked (本轮跳过，后续轮次/下次额度恢复会重试)")
            elif _is_permanent_broken(err_str):
                _PERMANENT_BLACKLIST.add(c["model"])
                info(f"{c['model']} permanently broken "
                      f"(404/400/access denied), blacklisted")
            elif any(s in err_str for s in
                     ("429", "500", "502", "503", "504")):
                round_blacklist.add(c["model"])
                if use_bandit:
                    bandit.update(c["model"], success=False, total_tokens=0,
                                  task_type=task_type, penalty=0.3)  # 借鉴点50
                info(f"server error for {c['model']}, "
                      f"skipping this round")
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
        reasoning=result.get("reasoning", ""),
        method=result["method"],
        latency_ms=latency_ms,
        model_routed_to=candidates[0]["model"] if candidates else None,
        routing_success=False,
    )

    return _safe_pass_through(request, next_call)
