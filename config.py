"""smart-router 配置加载 + 模型自动发现。

打包进 app 后，用户在设置界面（密钥页）配置 API key —— 写入
~/.hermes/.env 或 custom_providers。本模块自动发现这些 key 对应的模型，
构建路由池，全程不依赖手改 config.yaml 里的模型列表。

发现来源（按顺序）：
  1. custom_providers（内联 api_key / api_key_env + 显式 models 列表，
     缺失时调 /models）
  2. Hermes provider 注册表（providers.list_providers()，覆盖 app 里
     配好的标准 provider）
  3. 硬编码已知端点兜底（.env 只配了 key、config 没配 provider 的场景）

安全约定：日志绝不打印 key 值。
"""

import os
import re
import sys
from pathlib import Path

# ── 模块级缓存 ─────────────────────────────────────────────────────────
_FETCHED_MODELS_CACHE = {}   # base_url → [model_id, ...]
_ENV_FILE_CACHE = None       # ~/.hermes/.env 解析结果
_ROUTER_CACHE = None
_ROUTER_CACHE_KEY = None

# ── 非对话模型关键词（不进任何路由池）──────────────────────────────────
_NON_CHAT_KEYWORDS = (
    "image", "audio", "asr", "tts", "embedding", "rerank", "omni",
    "ocr", "video", "speech", "voice", "whisper",
    "translate", "s2s", "-vl", "vision",
    "seedance", "seedream", "seed3d", "seededit", "seaweed",
    "wan", "i2v", "t2v", "i2i", "t2i", "flf2v", "hyper3d", "hitem3d",
    "deep-research", "deep-search", "deep_research", "deep_search",
    "-mt",
)
# ── 模型分级映射表 ─────────────────────────────────────
# 匹配优先级从高到低：
#   1) _EXACT_TIER    精确模型名（覆盖命名有歧义的模型）
#   2) _SPEC_SIMPLE   规格后缀 → simple（flash/lite/turbo 拉低）
#   3) _SPEC_COMPLEX  规格后缀 → complex（max/pro/plus/thinking 拉高）
#   4) _FAMILY_TIER   家族前缀（家族默认分级）
#   5) 参数量兜底（≥32b → complex，≤14b → simple）
#   6) 默认 simple（宁可强模型干简单活，不让弱模型干复杂推理）

_EXACT_TIER = {
    "deepseek-chat": "complex",      # v3 通用大模型别名
    "deepseek-reasoner": "complex",  # 推理
    "deepseek-v3": "complex",
    "deepseek-v3.1": "complex",
    "deepseek-v3.2": "complex",
    "gpt-4o-mini": "simple",
    "gpt-4o": "complex",
    "moonshot-v1-8k": "simple",
    "moonshot-v1-32k": "complex",
    "moonshot-v1-128k": "complex",
}

_SPEC_SIMPLE = (
    "distill", "flash", "lite", "air", "nano", "turbo", "tiny",
    "small", "-mini", "haiku",
)
_SPEC_COMPLEX = (
    "thinking", "reasoner", "max", "pro", "plus", "ultra", "premium",
    "mega", "large", "opus", "sonnet",
)

_FAMILY_TIER = (
    ("deepseek-r1", "complex"),
    ("deepseek-v3", "complex"),
    ("deepseek-v4", "complex"),
    ("qwq", "complex"),
    ("qvq", "complex"),
    ("kimi", "complex"),
    ("moonshot", "complex"),
    ("minimax", "complex"),
    ("glm-4", "complex"),
    ("glm-5", "complex"),
    ("doubao-seed", "complex"),
    ("qwen-coder", "complex"),
    ("qwen3-coder", "complex"),
    ("qwen-math", "complex"),
    ("qwen-long", "complex"),
    ("gpt-4", "complex"),
    ("gpt-5", "complex"),
    ("claude", "complex"),
    ("zhipu", "complex"),
    ("xiaomi", "complex"),
    ("vanchin", "complex"),
    ("siliconflow", "complex"),
)


def classify_model(name: str) -> str:
    """把模型分入 'simple' 或 'complex' 池。"""
    m = name.lower()

    # 1) 精确映射
    if m in _EXACT_TIER:
        return _EXACT_TIER[m]

    # 2) 规格后缀 → simple（flash/lite/turbo 等拉低，如 deepseek-v4-flash）
    for kw in _SPEC_SIMPLE:
        if kw in m:
            return "simple"

    # 3) 规格后缀 → complex
    for kw in _SPEC_COMPLEX:
        if kw in m:
            return "complex"

    # 4) 家族前缀
    for prefix, tier in _FAMILY_TIER:
        if m.startswith(prefix):
            return tier

    # 5) 参数量兜底
    mm = re.search(r"[-_](\d+(?:\.\d+)?)b(?=[-_.]|$)", m)
    if mm:
        val = float(mm.group(1))
        if val >= 32:
            return "complex"
        if val <= 14:
            return "simple"

    # 6) 默认 simple
    return "simple"


# ── 路径 ────────────────────────────────────────────────────────────────
def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _config_path() -> Path:
    return _hermes_home() / "config.yaml"


def _mtime(p: Path) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


# ── 配置读取 ────────────────────────────────────────────────────────────
def _load_yaml() -> dict:
    try:
        import yaml
        with open(_config_path(), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_env_file() -> dict:
    """解析 ~/.hermes/.env → {KEY: value}（仅内存，绝不打印值）。"""
    global _ENV_FILE_CACHE
    if _ENV_FILE_CACHE is None:
        _ENV_FILE_CACHE = {}
        try:
            with open(_hermes_home() / ".env", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and v:
                        _ENV_FILE_CACHE[k] = v
        except Exception:
            pass
    return _ENV_FILE_CACHE


def get_api_key(key: str) -> str:
    """返回 api_key 实际值。os.environ 优先，.env 文件兜底。"""
    if not key:
        return ""
    v = os.environ.get(key)
    if v:
        return v
    return _load_env_file().get(key, "")


def load_default_model() -> str:
    data = _load_yaml()
    return (data.get("model", {}) or {}).get("default", "")


# ── 模型分类 ────────────────────────────────────────────────────────────
def _is_chat_model(name: str) -> bool:
    m = name.lower()
    return not any(kw in m for kw in _NON_CHAT_KEYWORDS)


# ── 模型列表获取 ────────────────────────────────────────────────────────
def _fetch_models(base_url: str, api_key: str) -> list:
    """调 /models 拉模型列表（成功才缓存，失败不缓存以便下次重试）。"""
    if not base_url or not api_key:
        return []
    if base_url in _FETCHED_MODELS_CACHE:
        return _FETCHED_MODELS_CACHE[base_url]
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=6.0,
                        max_retries=0)
        ids = [m.id for m in client.models.list().data]
        _FETCHED_MODELS_CACHE[base_url] = ids
        return ids
    except Exception:
        return []


# ── provider 发现 ───────────────────────────────────────────────────────
def _resolve_key(cp: dict) -> str:
    """从 provider 条目解析 api_key：内联 api_key 优先，其次 api_key_env。"""
    raw = cp.get("api_key") or cp.get("key")
    if raw:
        return raw
    env = cp.get("api_key_env") or cp.get("key_env") or ""
    if env:
        return get_api_key(env)
    # 从 name 推导 env 名
    name = str(cp.get("name") or cp.get("provider_key") or "")
    name = name.upper().replace("-", "_").replace(".", "_")
    return get_api_key(f"{name}_API_KEY") if name else ""


def _iter_custom_providers(data: dict):
    """统一读取 custom_providers（list 或 keyed dict 两种格式）。"""
    cps = data.get("custom_providers", [])
    if isinstance(cps, dict):
        cps = list(cps.values())
    for cp in (cps or []):
        if isinstance(cp, dict):
            yield cp


def _scan_provider_plugins(add_model) -> None:
    """扫描 Hermes provider 注册表。打包环境 import 失败时静默跳过。

    对每个有 base_url + 已配 key 的 OpenAI 兼容 provider：
    优先用 fallback_models / fetch_models；都没有则主动调 /models 拉取。
    """
    # 非 OpenAI 兼容的 api_mode —— 调 /models 会失败或格式不对，直接跳过
    _SKIP_API_MODES = {"anthropic_messages", "bedrock_converse", "codex_responses", "copilot_acp"}
    try:
        from providers import list_providers
        profiles = list_providers()
    except Exception:
        return
    for prof in profiles:
        base = getattr(prof, "base_url", "") or ""
        if not base or not base.startswith("http"):
            continue
        api_mode = getattr(prof, "api_mode", "") or ""
        if api_mode in _SKIP_API_MODES:
            continue
        for env in (getattr(prof, "env_vars", ()) or ()):
            key = get_api_key(env)
            if not key:
                continue
            models = list(getattr(prof, "fallback_models", ()) or ())
            fn = getattr(prof, "fetch_models", None)
            if callable(fn):
                try:
                    fetched = fn(api_key=key, base_url=base, timeout=8.0) or []
                except Exception:
                    fetched = []
                for m in fetched:
                    if m not in models:
                        models.append(m)
            # ★ 兜底：fallback_models 和 fetch_models 都没出模型时，主动调 /models
            if not models:
                models = _fetch_models(base, key)
            for m in models:
                add_model(m, base, key)
            break  # 一个 provider 取一个已配 key 即可


def _discover_provider_models(data: dict) -> dict:
    """扫描所有 provider 来源，返回 {model_name: [{base_url, api_key}, ...]}。

    同一模型可挂多个 endpoint（按顺序 fallback）。全程不打印 key 值。
    """
    discovered = {}

    def add_model(model_name, base_url, api_key):
        if not model_name or not base_url or not api_key:
            return
        entry = {"base_url": base_url, "api_key": api_key}
        seen = discovered.setdefault(model_name, [])
        if entry not in seen:
            seen.append(entry)

    # 来源一：custom_providers（内联 api_key / api_key_env，优先显式 models）
    for cp in _iter_custom_providers(data):
        base_url = cp.get("base_url", "") or ""
        api_key = _resolve_key(cp)
        if not base_url or not api_key:
            continue
        models = cp.get("models") or cp.get("model") or []
        if isinstance(models, str):
            models = [models]
        models = [m for m in models if isinstance(m, str)]
        if not models:
            models = _fetch_models(base_url, api_key)
        for m in models:
            add_model(m, base_url, api_key)

    # 来源二：Hermes provider 注册表（app 里配的标准 provider）
    _scan_provider_plugins(add_model)

    return discovered


# ── 主入口 ──────────────────────────────────────────────────────────────
def load_router_config(force: bool = False) -> dict:
    """加载 smart_model_routing 配置，自动发现模型并构建路由池。

    带 mtime 缓存（config.yaml / .env 变了就重扫），改配置热生效，无需重启。
    返回 dict：enabled/bandit/simple_models/complex_models/providers。
    providers 为 {model: [{base_url, api_key}, ...]}（多 endpoint list）。

    force=True：调试用，即使 enabled=false 也照常发现并返回池子，
    且不读写缓存（避免污染正常路径）。
    """
    global _ROUTER_CACHE, _ROUTER_CACHE_KEY

    cache_key = (_mtime(_config_path()), _mtime(_hermes_home() / ".env"))
    if not force and _ROUTER_CACHE is not None and _ROUTER_CACHE_KEY == cache_key:
        return _ROUTER_CACHE

    data = _load_yaml()
    cfg = data.get("smart_model_routing", {}) or {}

    if not force and (not cfg or not cfg.get("enabled")):
        _ROUTER_CACHE, _ROUTER_CACHE_KEY = {}, cache_key
        return {}

    # 发现 + 过滤 + 分类
    discovered = _discover_provider_models(data)
    simple = set()
    complex_ = set()
    for m in discovered:
        if not _is_chat_model(m):
            continue
        (simple if classify_model(m) == "simple" else complex_).add(m)

    # ── 白名单收窄：config 显式指定 simple_models/complex_models 时只保留它们 ──
    # 任一列表非空即视为“收窄模式”：路由池限定为这两个列表的并集，
    # 且按列表归属重新分池（覆盖自动分类结果）。
    # 未发现可用 endpoint 的模型会被忽略并告警。
    explicit_simple = [m for m in (cfg.get("simple_models") or []) if isinstance(m, str)]
    explicit_complex = [m for m in (cfg.get("complex_models") or []) if isinstance(m, str)]
    if explicit_simple or explicit_complex:
        simple = {m for m in explicit_simple if m in discovered}
        complex_ = {m for m in explicit_complex if m in discovered}
        missing = (set(explicit_simple) | set(explicit_complex)) - simple - complex_
        if missing:
            print(f"[smart-router] 白名单收窄：以下模型未发现可用 endpoint，已忽略: "
                  f"{sorted(missing)}", file=sys.stderr)

    providers = {}
    for m in simple | complex_:
        providers[m] = discovered[m]

    non_chat = len(discovered) - len(simple) - len(complex_)
    n_endpoints = len({e["base_url"] for es in discovered.values() for e in es})
    print(f"[smart-router] 自动发现完成: {len(simple)} simple + {len(complex_)} "
          f"complex 模型（原始 {len(discovered)} 个，过滤非对话 {non_chat} 个，"
          f"来自 {n_endpoints} 个 provider）", file=sys.stderr)

    result = dict(cfg)
    result["simple_models"] = sorted(simple)
    result["complex_models"] = sorted(complex_)
    result["providers"] = providers

    if not force:
        _ROUTER_CACHE, _ROUTER_CACHE_KEY = result, cache_key
    return result
