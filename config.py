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
_ENV_FILE_CACHE = {}         # <home>/.env 解析结果 {KEY: value}
_ENV_FILE_CACHE_KEY = None   # <home>/.env mtime 缓存键（None=从未读过）
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
    # ── DeepSeek（命名规则的通用大模型，精确避免歧义）──
    "deepseek-chat": "complex",       # v3 通用大模型别名
    "deepseek-reasoner": "complex",   # 推理
    "deepseek-v3": "complex",
    "deepseek-v3.1": "complex",
    "deepseek-v3.2": "complex",
    # ── OpenAI（旗舰与轻量命名易冲突，精确锁定）──
    "gpt-4o-mini": "simple",
    "gpt-4o": "complex",
    "gpt-4-turbo": "complex",         # 旗舰，避免被 turbo 关键词误降为 simple
    "gpt-oss-20b": "simple",          # 小杯，无规格关键词，需精确
    "gpt-oss-120b": "complex",
    # ── Moonshot（按上下文窗口分级）──
    "moonshot-v1-8k": "simple",
    "moonshot-v1-32k": "complex",
    "moonshot-v1-128k": "complex",
    # ── 与 turbo 关键词冲突的旗舰（精确覆盖，避免误降）──
    "hunyuan-turbo": "complex",
    # ── 无参数量后缀、无家族前缀可兜底的旗舰 ──
    "llama-4-maverick": "complex",
    "llama-4-behemoth": "complex",
    "mistral-medium": "complex",
    "internlm3": "complex",
}

_SPEC_SIMPLE = (
    "distill", "flash", "lite", "air", "nano", "turbo", "tiny",
    "small", "-mini", "haiku",
    "lightning", "speed", "standard", "scout",
)
_SPEC_COMPLEX = (
    "thinking", "reasoner", "max", "pro", "plus", "ultra", "premium",
    "mega", "large", "opus", "sonnet",
)

_FAMILY_TIER = (
    # ── DeepSeek ──
    ("deepseek-r1", "complex"),
    ("deepseek-v3", "complex"),
    ("deepseek-v4", "complex"),
    ("deepseek", "complex"),          # 统一兜底（chat 等已精确）
    # ── OpenAI ──
    ("gpt-3.5", "simple"),            # 旧模型，轻量
    ("gpt-4", "complex"),
    ("gpt-5", "complex"),
    ("gpt-oss", "complex"),           # 兜底，20b 已精确 simple
    ("o1", "complex"),
    ("o3", "complex"),
    ("o4", "complex"),
    ("chatgpt", "complex"),           # chatgpt-4o-latest
    # ── Anthropic ──
    ("claude", "complex"),            # haiku 靠关键词降 simple
    # ── Google ──
    ("gemini", "complex"),            # flash 靠关键词降 simple
    ("gemma", "simple"),              # 开源轻量
    # ── xAI ──
    ("grok", "complex"),              # mini 靠关键词降 simple
    # ── Mistral 系（分品牌，不加整族以保留参数量兜底）──
    ("codestral", "complex"),
    ("pixtral", "complex"),
    ("ministral", "simple"),
    # ── 通义千问（仅精确族；版本号靠参数量兜底，避免 qwen-7b 被误升）──
    ("qwen-coder", "complex"),
    ("qwen3-coder", "complex"),
    ("qwen-math", "complex"),
    ("qwen-long", "complex"),
    ("qwq", "complex"),
    ("qvq", "complex"),
    # ── Kimi / Moonshot ──
    ("kimi", "complex"),
    ("moonshot", "complex"),
    # ── 智谱 GLM ──
    ("glm-4", "complex"),
    ("glm-5", "complex"),
    ("glm", "complex"),               # 兜底 glm-z1 / glm-3
    ("zhipu", "complex"),
    # ── 豆包 / 字节 ──
    ("doubao-seed", "complex"),
    ("doubao", "complex"),            # pro/lite/flash/standard 靠关键词
    # ── MiniMax ──
    ("minimax", "complex"),
    ("abab", "complex"),
    # ── 百度文心 ──
    ("ernie", "complex"),             # speed/lite/turbo 靠关键词降 simple
    ("wenxin", "complex"),
    # ── 腾讯混元 ──
    ("hunyuan", "complex"),           # lite/standard 靠关键词，turbo 已精确
    # ── 讯飞星火 ──
    ("spark", "complex"),             # lite 靠关键词
    # ── 阶跃星辰 ──
    ("step-1", "complex"),
    ("step-2", "complex"),
    ("step-3", "complex"),
    # ── 零一万物（只兜 medium；large 靠关键词，lightning 靠关键词，开源走参数量）──
    ("yi-medium", "complex"),
    # ── 百川 ──
    ("baichuan", "complex"),
    # ── Cohere ──
    ("command-r", "complex"),
    ("command-a", "complex"),
    # ── 书生 / 面壁 / 天工 / 盘古 / 商汤 / 小米 / Nous ──
    ("minicpm", "simple"),            # 面壁开源小模型
    ("skywork", "complex"),
    ("tiangong", "complex"),
    ("pangu", "complex"),
    ("sensechat", "complex"),
    ("sense-nova", "complex"),
    ("mimo", "simple"),               # 小米 MiMo 开源轻量
    ("hermes", "complex"),            # NousResearch hermes
    # ── 平台/其他（原表保留）──
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
    """解析 Hermes home 目录（多级探测，兼容桌面打包版）。

    解析顺序：
      1. HERMES_HOME 环境变量（Hermes 主程序显式指定，profile/自定义部署）
      2. Windows 桌面打包版（按品牌优先级探测）：
         - Sinoregal Agent（新版）：%LOCALAPPDATA%\\sinoregal
         - 旧版 hermes-desktop：    %LOCALAPPDATA%\\hermes
         Sinoregal Agent 把数据目录改名 sinoregal 以避免和旧 hermes 安装
         冲突，且**不设置** HERMES_HOME 环境变量（主程序内部默认路径已改，
         但插件是独立代码看不到），只能靠探测：目录存在即视为 home。
         这也是 .env / config.yaml / plugins 实际所在的位置。
         两个目录都探测，优先 sinoregal（新版），找不到才退回 hermes（旧版兼容）。
      3. POSIX（macOS / Linux）：~/.sinoregal（新版 Sinoregal Agent，与
         Windows 的 %LOCALAPPDATA%\\sinoregal 对称）→ ~/.hermes（旧版
         NousResearch hermes，兜底），按新版优先探测。
    """
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local:
            # Sinoregal Agent 打包版（新）→ 旧 hermes-desktop 打包版（兜底）
            for name in ("sinoregal", "hermes"):
                candidate = Path(local) / name
                try:
                    if candidate.is_dir():
                        return candidate
                except OSError:
                    pass
    else:
        # POSIX：新版 Sinoregal Agent 数据目录 ~/.sinoregal（新版优先），
        # 旧版 hermes 兜底 ~/.hermes。
        for name in (".sinoregal", ".hermes"):
            candidate = Path.home() / name
            try:
                if candidate.is_dir():
                    return candidate
            except OSError:
                pass
    return Path.home() / ".hermes"


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
    """解析 <home>/.env → {KEY: value}（仅内存，绝不打印值）。

    带 mtime 缓存：.env 文件内容变化后自动重读，改 key 无需重启进程。
    与 load_router_config 的 cache_key 联动——.env 的 mtime 变了会触发
    重扫，本函数必须同步返回新 key，否则会拿到旧值（旧 bug）。
    文件不存在视为「空配置」缓存（文件出现时 mtime 变化自动失效）；
    读取失败不更新缓存键，下次调用会重试（与 _load_yaml 语义一致）。
    """
    global _ENV_FILE_CACHE, _ENV_FILE_CACHE_KEY
    env_path = _hermes_home() / ".env"
    key = _mtime(env_path)
    if _ENV_FILE_CACHE_KEY == key:
        return _ENV_FILE_CACHE
    cache = {}
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v:
                    cache[k] = v
        _ENV_FILE_CACHE, _ENV_FILE_CACHE_KEY = cache, key
    except FileNotFoundError:
        # .env 不存在：缓存空结果，避免每次调用都重试 open
        _ENV_FILE_CACHE, _ENV_FILE_CACHE_KEY = {}, key
    except Exception:
        pass  # 其他读取失败：保持旧缓存键，下次重试
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
# 非 OpenAI 兼容的 api_mode —— 调 /models 会失败或格式不对，直接跳过
_SKIP_API_MODES = {"anthropic_messages", "bedrock_converse", "codex_responses", "copilot_acp"}


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
    """统一读取 custom_providers（list 或 keyed dict 两种格式）。

    兼容两种落点：
      1. 旧版/插件自举:顶层 custom_providers 段（list 或 keyed dict）
      2. App 界面配置:providers.<name> 段（dict，含 base_url/key_env/models）——
         主程序「自定义模型端点」写在这里，custom_providers 段可能为空或缺失。
    """
    cps = data.get("custom_providers", [])
    if isinstance(cps, dict):
        cps = list(cps.values())
    for cp in (cps or []):
        if isinstance(cp, dict):
            yield cp
    # App 界面「自定义模型端点」的落点: providers.<name>（name/base_url/key_env/models）
    for name, pv in (data.get("providers") or {}).items():
        if not isinstance(pv, dict):
            continue
        base_url = pv.get("base_url", "") or ""
        if not base_url.startswith("http"):
            continue
        api_mode = pv.get("api_mode", "") or ""
        if api_mode in _SKIP_API_MODES:
            continue
        cp = dict(pv)
        cp.setdefault("name", name)
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
        if isinstance(models, dict):
            # App 界面格式: models: {模型名: {...配置...}} → 取 keys
            models = list(models.keys())
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


# ── 自举默认配置（config.yaml 缺段时写入，装上即用）────────────────────
_BOOTSTRAP_TEXT = (
    "\nsmart_model_routing:\n"
    "  enabled: true\n"
    "  announce: true\n"
    "  bandit:\n"
    "    enabled: true\n"
    "    ucb_c: 1.0\n"
    "    alpha: 0.012\n"
    "    base_reward: 100.0\n"
    "    budget: 0.002        # $/请求 预算上限（Budget Pacer, 0=关闭）\n"
    "    lambda_c: 0.3        # 静态成本偏好\n"
    "    prices: {}           # 模型 → $/1k tokens，空则默认 0.001\n"
    "    quality_w: 1.0\n"
    "    cost_w: 0.4\n"
    "    latency_w: 0.3\n"
    "    burn_in_pulls: 20\n"
    "    tie_eps: 0.02\n"
)


def _bootstrap_router_config() -> None:
    """config.yaml 顶层缺少 smart_model_routing 段时，追加默认配置。

    只在段完全不存在时写入；段存在（含 enabled:false）一律不动，
    保证用户显式关闭的选择不被覆盖。文本追加而非 yaml 重写，
    保留原文件注释与格式。失败仅告警，不抛错、不影响放行路径。
    """
    try:
        import yaml
        path = _config_path()
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            top = yaml.safe_load(text or "{}") or {}
        except Exception:
            top = {}
        if "smart_model_routing" in top:
            return  # 已存在（重入/并发保护）
        if text and not text.endswith("\n"):
            text += "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + _BOOTSTRAP_TEXT)
        print("[smart-router] config.yaml 缺少 smart_model_routing 段，"
              "已写入默认配置（enabled: true）", file=sys.stderr)
    except Exception as e:
        print(f"[smart-router] 自举写入默认配置失败: {e}", file=sys.stderr)


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
        # ── 自举：config.yaml 完全没有 smart_model_routing 段时写入默认配置 ──
        # 段存在但 enabled=false 视为用户主动关闭，绝不改写。
        # 写入后重读（mtime 已变，缓存自然失效），仍不可用才放行。
        if "smart_model_routing" not in data:
            _bootstrap_router_config()
            data = _load_yaml()
            cfg = data.get("smart_model_routing", {}) or {}
        if not cfg or not cfg.get("enabled"):
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
