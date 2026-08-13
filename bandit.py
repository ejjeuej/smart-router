"""
Phase 2: UCB 多臂老虎机（D-UCB 变体）——动态选择性价比最高的模型。

针对非平稳环境（模型池运行时可变）做了五项改进：
1. 全局衰减（所有模型同步 decay，而非只衰减被选中的）
2. task_type 线性混合（专精不够时用 overall 兜底）
3. 贝叶斯先验冷启动（新模型注入同家族虚拟 stats）
4. 滑动窗口 N（防止探索项随时间漂移）
5. tokens 与 pulls/reward 同步衰减（保持比值一致）

算法本质：D-UCB + Sliding-Window UCB (Garivier & Moulines 2008) + Bayesian Prior

持久化: <plugin_dir>/data/bandit_<pool_key>.json
"""

import json
import math
import threading
from pathlib import Path

_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DECAY = 0.99          # 全局衰减率，每轮旧数据 ×0.95
PRIOR_PULLS = 3        # 先验虚拟 pulls
PRIOR_BLEND_MIN = 5    # 专精线性混合的 min_pulls
EPS = 0.01             # 除零保护

# 家族映射函数，由 init.py 在加载时注入。
# 默认：每个模型自己当自己的家族（无先验共享）。
_get_family_fn = lambda model: model


def set_family_fn(fn):
    """注入家族映射函数，由 init.py 调用。"""
    global _get_family_fn
    _get_family_fn = fn


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_family(model: str) -> str:
    """返回模型所属家族名，由 init.py 注入的映射函数决定。"""
    return _get_family_fn(model)


def _compute_family_avg(stats: dict, family: str, exclude_model: str) -> float | None:
    """计算同家族已有模型的 overall 平均 reward，排除自身。

    stats: {model_name: {overall: {pulls, total_reward, ...}, ...}}
    返回 avg_reward，无同类时返回 None。
    """
    total_r = 0.0
    count = 0
    for model, s in stats.items():
        if model == exclude_model:
            continue
        if _get_family(model) != family:
            continue
        overall = s.get("overall", {})
        pulls = overall.get("pulls", 0)
        if pulls > 0:
            total_r += overall["total_reward"] / pulls
            count += 1
    if count == 0:
        return None
    return total_r / count


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class UCBBandit:
    def __init__(self, c=1.0, alpha=0.001, base_reward=100.0):
        """c: 探索系数  alpha: token 成本权重  base_reward: 基础奖励"""
        self.c = float(c)
        self.alpha = float(alpha)
        self.base_reward = float(base_reward)
        # stats: {model: {overall: {pulls, total_reward, total_tokens},
        #                 coding: {...}, writing: {...}, ...}}
        self.stats = {}
        self.total_rounds = 0

    # ── 冷启动先验 ────────────────────────────────────────────────

    def _inject_prior(self, model: str, all_models: list[str]) -> None:
        """为新模型注入虚拟拉臂次数，基于同家族历史平均 reward。

        若无同类，用 base_reward / 2 作为中性先验。
        """
        family = _get_family(model)
        family_avg = _compute_family_avg(self.stats, family, model)

        if family_avg is not None:
            prior_reward = family_avg
        else:
            prior_reward = 0.5  # 中性先验（归一化后 base_reward/2 → 0.5）

        # 估算先验 token 用量：用同家族平均 tokens/pull，否则 5000
        estimated_tokens = 0
        for sib_name, sib_stats in self.stats.items():
            if _get_family(sib_name) == family:
                overall = sib_stats.get("overall", {})
                if overall.get("pulls", 0) > 0:
                    estimated_tokens = overall["total_tokens"] / overall["pulls"]
                    break
        if estimated_tokens == 0:
            estimated_tokens = 5000

        self.stats[model] = {
            "overall": {
                "pulls": float(PRIOR_PULLS),
                "total_reward": PRIOR_PULLS * prior_reward,
                "total_tokens": PRIOR_PULLS * estimated_tokens,
            }
        }

    # ── 混合 reward ─────────────────────────────────────────────────

    def _blended_stats(self, model: str, task_type: str | None):
        """返回 (avg_reward, pulls) 的线性混合结果。

        task_type 专精 pulls < PRIOR_BLEND_MIN 时，与 overall 按比例混合。
        专精不存在时纯 overall。
        """
        overall = self.stats[model].get("overall", {})
        o_pulls = max(overall.get("pulls", 0), EPS)
        o_avg = overall.get("total_reward", 0) / o_pulls

        if not task_type or task_type not in self.stats[model]:
            return o_avg, o_pulls

        t = self.stats[model][task_type]
        t_pulls = max(t.get("pulls", 0), EPS)
        if t_pulls < EPS:
            return o_avg, o_pulls

        t_avg = t["total_reward"] / t_pulls

        # 线性混合：专精 pulls 越多，专精分权重越大
        w = min(t_pulls / PRIOR_BLEND_MIN, 1.0)
        blended_avg = w * t_avg + (1.0 - w) * o_avg
        blended_pulls = w * t_pulls + (1.0 - w) * o_pulls

        return blended_avg, blended_pulls

    # ── 全局衰减 ────────────────────────────────────────────────────

    def _decay_all(self) -> None:
        """对所有模型的所有键做全局衰减。"""
        for model in self.stats:
            for key in list(self.stats[model].keys()):
                s = self.stats[model][key]
                s["pulls"] = max(s.get("pulls", 0) * DECAY, EPS)
                s["total_reward"] *= DECAY
                s["total_tokens"] *= DECAY

    # ── 选择 ────────────────────────────────────────────────────────

    def select(self, candidates, task_type=None):
        """从候选列表中选出 UCB score 最高的模型。

        candidates: [{"model": "...", "base_url": "...", "api_key": "..."}, ...]
        task_type:  "coding" / "writing" / "analysis" / ... 用于专精混合

        新模型自动注入先验后参与正常 UCB 评分（不再无条件优先冷启动）。
        """
        if not candidates:
            return None

        # 1. 全局衰减（不管成败，每次选模都腐化旧数据）
        self._decay_all()

        # 2. 注入先验：池中已有的模型名全量
        all_models = [c["model"] for c in candidates]
        for c in candidates:
            if c["model"] not in self.stats:
                self._inject_prior(c["model"], all_models)

        # 3. UCB 选最优：effective_N = 当前候选模型 overall pulls 之和
        effective_N = sum(
            self.stats[c["model"]]["overall"]["pulls"]
            for c in candidates
            if c["model"] in self.stats
        )
        best = None
        best_score = float("-inf")

        for c in candidates:
            avg, pulls = self._blended_stats(c["model"], task_type)
            pulls_safe = max(pulls, EPS)

            exploration = self.c * math.sqrt(
                math.log(effective_N + 1) / pulls_safe
            )
            score = avg + exploration

            if score > best_score:
                best_score = score
                best = c

        return best

    # ── 更新 ────────────────────────────────────────────────────────

    def update(self, model, success, total_tokens, task_type=None):
        """更新模型统计。

        model:        模型名
        success:      API 调用是否成功
        total_tokens: 本次调用消耗的 token 数
        task_type:    任务类型，同时更新 overall 和对应专精
        """
        # 1. 计算 reward（归一化到 [0, 1]，匹配 UCB 理论假设）
        if success:
            raw = max(self.base_reward - self.alpha * total_tokens, 0.1)
            reward = raw / self.base_reward     # → [0.001, 1.0]
        else:
            reward = 0.0

        # 2. 确保 stats 条目存在
        if model not in self.stats:
            self.stats[model] = {}
        for key in ("overall", task_type):
            if key is None:
                continue
            if key not in self.stats[model]:
                self.stats[model][key] = {
                    "pulls": 0.0, "total_reward": 0.0, "total_tokens": 0.0
                }

        # 3. 更新 overall（始终更新）
        o = self.stats[model]["overall"]
        o["pulls"] += 1.0
        o["total_reward"] += reward
        o["total_tokens"] += total_tokens

        # 4. 更新 task_type 专精（如果有）
        if task_type:
            t = self.stats[model][task_type]
            t["pulls"] += 1.0
            t["total_reward"] += reward
            t["total_tokens"] += total_tokens

        # 5. 全局计数器
        self.total_rounds += 1

    # ── 序列化 ──────────────────────────────────────────────────────

    def to_dict(self):
        return {
            "stats": self.stats,
            "total_rounds": self.total_rounds,
            "c": self.c,
            "alpha": self.alpha,
            "base_reward": self.base_reward,
        }

    @classmethod
    def from_dict(cls, d):
        """从磁盘恢复，兼容旧格式（flat stats 自动迁移到嵌套结构）。"""
        b = cls(
            c=d.get("c", 2.0),
            alpha=d.get("alpha", 0.00001),
            base_reward=d.get("base_reward", 100.0),
        )
        raw = d.get("stats", {})
        b.total_rounds = d.get("total_rounds", 0)

        # 检测格式：如果每个 value 里有 "pulls" 键，是旧 flat 格式
        migrated = {}
        for model, val in raw.items():
            if "pulls" in val:
                # 旧格式：{model: {pulls, total_reward, total_tokens}}
                migrated[model] = {"overall": val}
            elif isinstance(val, dict) and "overall" in val:
                # 已经是新格式
                migrated[model] = val
            else:
                # 未知格式，跳过
                pass
        b.stats = migrated
        return b


# ---------------------------------------------------------------------------
# 按池管理（每个 pool_key 一个 bandit 实例）
# ---------------------------------------------------------------------------

_bandits = {}

_DATA_DIR = None


def _get_data_dir():
    global _DATA_DIR
    if _DATA_DIR is None:
        _DATA_DIR = Path(__file__).resolve().parent / "data"
    return _DATA_DIR


def get_bandit(pool_key, bandit_config=None):
    """获取或创建某个池的 bandit 实例，自动从磁盘加载历史数据。

    pool_key:       "simple_models" / "complex_models"
    bandit_config:  {"c": 2.0, "alpha": 0.00001, "base_reward": 100.0} 或 None
    """
    if pool_key not in _bandits:
        cfg = bandit_config or {}
        _bandits[pool_key] = UCBBandit(
            c=cfg.get("ucb_c", 2.0),
            alpha=cfg.get("alpha", 0.00001),
            base_reward=cfg.get("base_reward", 100.0),
        )
        _load(pool_key)
        # 关键修复：_load 会用 JSON 里持久化的 c/alpha/base_reward 覆盖 config，
        # 导致改 config.yaml 的 ucb_c 不生效。这里强制用 config 的值覆盖回来。
        b = _bandits[pool_key]
        b.c = float(cfg.get("ucb_c", 2.0))
        b.alpha = float(cfg.get("alpha", 0.00001))
        b.base_reward = float(cfg.get("base_reward", 100.0))
    return _bandits[pool_key]


def reset_bandit(pool_key):
    """强制重置某个池的 bandit（清空内存 + 磁盘）。
    
    pool_key: "simple_models" / "complex_models"
    """
    data_dir = _get_data_dir()
    path = data_dir / f"bandit_{pool_key}.json"
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
    _bandits.pop(pool_key, None)


def _load(pool_key):
    """从磁盘恢复 bandit 状态。"""
    path = _get_data_dir() / f"bandit_{pool_key}.json"
    try:
        with open(path) as f:
            data = json.load(f)
            _bandits[pool_key] = UCBBandit.from_dict(data)
    except Exception:
        pass


def save_all():
    """持久化所有 bandit 状态到磁盘。"""
    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        for key, bandit in _bandits.items():
            path = data_dir / f"bandit_{key}.json"
            try:
                tmp = str(path) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(bandit.to_dict(), f, indent=2)
                Path(tmp).replace(path)
            except Exception:
                pass


def save_one(pool_key):
    """持久化单个 bandit（API 调用后即时保存）。"""
    if pool_key not in _bandits:
        return
    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"bandit_{pool_key}.json"
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_bandits[pool_key].to_dict(), f, indent=2)
        Path(tmp).replace(path)
    except Exception:
        pass
