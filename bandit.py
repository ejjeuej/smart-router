"""
Phase 2: UCB 多臂老虎机(D-UCB 变体)——动态选择性价比最高的模型。

针对非平稳环境(模型池运行时可变)做了多项改进:
1. 全局衰减(所有模型同步 decay,而非只衰减被选中的)
2. task_type 线性混合(专精不够时用 overall 兜底)
3. 贝叶斯先验冷启动(新模型注入同家族虚拟 stats;可被离线预热覆盖)
4. 滑动窗口 N(防止探索项随时间漂移)
5. tokens 与 pulls/reward 同步衰减(保持比值一致)

2026-08-17 bandit 线落地(借鉴点 49/40/41/42/43/45/46/50/51/54):
6. 奖励四分量(49): r = wq·q − wc·c̃ − wl·ℓ̃ − p,成本/延迟/质量拆独立通道
7. Budget Pacer(40/46): λ 对偶上升 + EMA 成本信号,打分减 (λc+λt)·c̃;
   超预算时硬上限剪枝最贵臂(电路断路器)
8. 成本 log 归一化(45): c̃=(log c−log c_floor)/(log c_ceil−log c_floor)
9. 陈旧度方差膨胀(41): 闲置臂探索 bonus 按闲置轮数膨胀,封顶防失控
10. 强制探索 burn-in(43): 新臂前 N 次无条件轮转,之后 UCB 接管
11. margin tie-breaker(54): top-2 打分差 < ε 时选历史 Q 均值高者,防抖动
12. 离线全信息预热(51): inject_offline_prior() 用预热平均奖励初始化 Q
    (neff = 预热样本数,均值保持,不缩向零)
13. 软惩罚(50): update() 的 penalty 参数,失败/被拒按严重度扣分,可恢复

算法本质:D-UCB + Sliding-Window UCB (Garivier & Moulines 2008) + Bayesian Prior
          + Budget-Paced λ 对偶上升 (ParetoBandit) + 四分量奖励 (OrcaRouter)

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

DECAY = 0.99          # 全局衰减率，每轮旧数据 ×0.99
PRIOR_PULLS = 3        # 先验虚拟 pulls（在线家族先验强度，借鉴点 42 的 neff 旋钮）
PRIOR_BLEND_MIN = 5    # 专精线性混合的 min_pulls
EPS = 0.01             # 除零保护

# ── 奖励四分量默认权重(借鉴点 49, OrcaRouter 默认 1.0/0.4/0.3/0.5/0.3)──
W_Q = 1.0              # 质量分量权重
W_C = 0.4              # 成本分量权重
W_L = 0.3              # 延迟分量权重
PEN_RATE_LIMIT = 0.3   # 429/5xx 软惩罚(借鉴点 50)
PEN_QUOTA = 0.5        # 403/余额耗尽软惩罚
LATENCY_MAX = 60000.0  # 延迟归一化封顶(ms,≈REQUEST_TIMEOUT, 20s→60s 同步 2026-08-20)

# ── 成本 log 归一化(借鉴点 45)──
C_FLOOR = 1e-4         # $/1k tokens 下限
C_CEIL = 1.0           # $/1k tokens 上限(旗舰级)
DEFAULT_PRICE = 0.001  # 未配置价格表的默认 $/1k tokens

# ── Budget Pacer(借鉴点 40)──
ALPHA_EMA = 0.05       # EMA 平滑系数(半衰期 ≈14 请求)
ETA = 0.05             # λ 对偶上升步长
LAMBDA_MAX = 5.0       # λ 投影封顶
LAMBDA_C = 0.3         # 静态成本偏好(第一请求就起作用)
HARD_CAP_MULT = 10.0   # 硬上限 = budget × 该倍数(借鉴点 46 电路断路器)

# ── cheap 意图接线(借鉴点 32 落地)──
CHEAP_LAMBDA_MULT = 3.0  # 用户说"省钱/随便"时 λ_c 临时乘数(请求级,不落盘)

# ── 陈旧度方差膨胀(借鉴点 41)──
STALE_GROWTH = 1.15    # 每闲置一轮探索 bonus 膨胀系数
STALE_MAX = 14.0       # 封顶倍数(≈√Vmax, 防无限膨胀吞掉成本惩罚)

# ── burn-in(借鉴点 43)──
BURN_IN_PULLS = 10     # 新臂强制探索次数(离线预热注入后视为已完成; 2026-08-21 由 20 调低, 加速 UCB 打分启用)

# ── margin tie-breaker(借鉴点 54)──
TIE_EPS = 0.02         # top-2 打分差阈值


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
        """c: 探索系数  alpha: (保留兼容)  base_reward: 基础奖励(仅文档用)"""
        self.c = float(c)
        self.alpha = float(alpha)
        self.base_reward = float(base_reward)
        # stats: {model: {overall: {pulls, total_reward, total_tokens},
        #                 coding: {...}, writing: {...}, ...}}
        self.stats = {}
        self.total_rounds = 0
        # ── bandit 线新增运行时状态 ──
        self.plays = {}            # model → 被选次数(不衰减, burn-in 用)
        self.last_activity = {}    # model → 最近活动轮次(select/update, 陈旧度用)
        self.cost_ema = 0.0        # EMA 每请求成本(美元), Budget Pacer 信号
        self.lambda_t = 0.0        # 对偶上升拉格朗日乘子(预算约束)
        # ── 可配置项(apply_config 注入)──
        self.budget = 0.0          # $/请求 预算上限; 0 = 不启用 Budget Pacer
        self.lambda_c = LAMBDA_C   # 静态成本偏好
        self.prices = {}           # model → $/1k tokens
        self.w_q = W_Q
        self.w_c = W_C
        self.w_l = W_L
        self.burn_in_pulls = BURN_IN_PULLS
        self.tie_eps = TIE_EPS

    # ── 配置注入 ────────────────────────────────────────────────────

    def apply_config(self, cfg: dict) -> None:
        """把 config 的 bandit 段应用到实例（默认值兜底）。

        cfg 可为 None 或 {}。lambda_t / cost_ema 是学习状态，不受影响。
        """
        if not cfg:
            return
        self.budget = float(cfg.get("budget") or 0.0)
        self.lambda_c = float(cfg.get("lambda_c", LAMBDA_C))
        self.prices = dict(cfg.get("prices") or {})
        self.w_q = float(cfg.get("quality_w", W_Q))
        self.w_c = float(cfg.get("cost_w", W_C))
        self.w_l = float(cfg.get("latency_w", W_L))
        self.burn_in_pulls = int(cfg.get("burn_in_pulls", BURN_IN_PULLS))
        self.tie_eps = float(cfg.get("tie_eps", TIE_EPS))

    # ── 成本工具(借鉴点 45: log 归一化)──────────────────────────────

    def _price_of(self, model: str) -> float:
        return self.prices.get(model, DEFAULT_PRICE)

    def _cost_of_tokens(self, model: str, total_tokens: float) -> float:
        """本次调用的美元成本。"""
        return total_tokens * self._price_of(model) / 1000.0

    def _est_cost(self, model: str) -> float:
        """估计该模型每请求成本(美元)：平均 tokens × 单价/1000。"""
        overall = self.stats.get(model, {}).get("overall", {})
        pulls = overall.get("pulls", 0)
        avg_tokens = (overall.get("total_tokens", 0) / pulls) if pulls > 0 else 5000.0
        return avg_tokens * self._price_of(model) / 1000.0

    @staticmethod
    def _log_norm_cost(cost: float) -> float:
        """成本 log 归一化到 [0,1]，压缩量级差异（530× 跨度）。"""
        if cost <= C_FLOOR:
            return 0.0
        if cost >= C_CEIL:
            return 1.0
        return (math.log(cost) - math.log(C_FLOOR)) / (math.log(C_CEIL) - math.log(C_FLOOR))

    # ── 冷启动先验 ────────────────────────────────────────────────

    def _inject_prior(self, model: str, all_models: list[str]) -> None:
        """为新模型注入虚拟拉臂次数，基于同家族历史平均 reward。

        若无同类，用 0.5 作为中性先验。
        先验强度 = PRIOR_PULLS（借鉴点 42 的 neff 在线版旋钮）。
        """
        family = _get_family(model)
        family_avg = _compute_family_avg(self.stats, family, model)

        if family_avg is not None:
            prior_reward = family_avg
        else:
            prior_reward = 0.5  # 中性先验

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
        self.plays.setdefault(model, 0)
        self.last_activity.setdefault(model, self.total_rounds)

    def inject_offline_prior(self, model: str, avg_reward: float,
                             n: int, avg_tokens: float = 0.0) -> None:
        """离线全信息预热注入(借鉴点 51, 42 的 neff 落地)。

        用预热集实测的平均奖励初始化该模型 Q 值：
          pulls = n（等效伪观测数 neff），total_reward = n × avg_reward。
        均值保持（直接写均值，无缩向零问题）。
        同时把 plays 置为 n —— 预热样本即探索，跳过 burn-in。

        avg_reward: 该模型在预热集上的平均四分量奖励
        n:          预热样本数（neff）
        avg_tokens: 预热实测平均 tokens/请求（默认 0 → 5000 兜底）
        """
        est_tokens = avg_tokens if avg_tokens > 0 else 5000.0
        self.stats[model] = {
            "overall": {
                "pulls": float(max(n, 1)),
                "total_reward": max(n, 1) * float(avg_reward),
                "total_tokens": max(n, 1) * est_tokens,
            }
        }
        # 预热样本 = 已完成探索：plays 置 n（≥ burn-in 阈值时不再强制轮转）
        self.plays[model] = max(n, 1)
        self.last_activity[model] = self.total_rounds

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
        """对所有模型的所有键做全局衰减。plays / last_activity 不衰减。"""
        for model in self.stats:
            for key in list(self.stats[model].keys()):
                s = self.stats[model][key]
                s["pulls"] = max(s.get("pulls", 0) * DECAY, EPS)
                s["total_reward"] *= DECAY
                s["total_tokens"] *= DECAY

    # ── 选择 ────────────────────────────────────────────────────────

    def _mark_play(self, model: str) -> None:
        """记录一次拉臂：plays +1，last_activity 更新到当前轮。"""
        self.plays[model] = self.plays.get(model, 0) + 1
        self.last_activity[model] = self.total_rounds

    def select(self, candidates, task_type=None, cheap=False):
        """从候选列表中选出 UCB score 最高的模型。

        candidates: [{"model": "...", "base_url": "...", "api_key": "..."}, ...]
        task_type:  "coding" / "writing" / "analysis" / ... 用于专精混合
        cheap:      True 时本轮临时抬高成本惩罚(借鉴点 32 接线,
                    用户说"省钱/随便" → cost_mode="cheap")。只影响本轮
                    打分,不落盘、不改变 λ_t / cost_ema 学习状态。

        选择顺序：
          1. 全局衰减
          2. 新臂注入先验
          3. burn-in(43)：池中存在 plays < burn_in_pulls 的臂 → 最少探索优先轮转
          4. 硬上限剪枝(46)：λ>0 时排除成本 > budget×HARD_CAP_MULT/(1+λ) 的臂
          5. UCB 打分：avg + c·√(logN/n)·stale − (λc·cheap_mult+λt)·c̃
          6. tie-breaker(54)：top-2 差 < ε → 选历史 Q 均值高者
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
            self.plays.setdefault(c["model"], 0)
            self.last_activity.setdefault(c["model"], self.total_rounds)

        # 3. burn-in(43)：强制探索未完成的新臂（最少探索优先，近似轮转）
        fresh = [c for c in candidates
                 if self.plays[c["model"]] < self.burn_in_pulls]
        if fresh:
            fresh.sort(key=lambda c: self.plays[c["model"]])
            best = fresh[0]
            self._mark_play(best["model"])
            return best

        # 4. 硬上限剪枝(46)：超预算时排除最贵臂（电路断路器）
        active = list(candidates)
        if self.budget > 0 and self.lambda_t > 0:
            hard_cap = max(self.budget * HARD_CAP_MULT, 1e-6) / (1.0 + self.lambda_t)
            pruned = [c for c in active if self._est_cost(c["model"]) <= hard_cap]
            if pruned:
                active = pruned

        # 5. UCB 选最优：effective_N = 当前候选模型 overall pulls 之和
        effective_N = sum(
            self.stats[c["model"]]["overall"]["pulls"]
            for c in active
            if c["model"] in self.stats
        )
        scored = []
        for c in active:
            avg, pulls = self._blended_stats(c["model"], task_type)
            pulls_safe = max(pulls, EPS)

            exploration = self.c * math.sqrt(
                math.log(effective_N + 1) / pulls_safe
            )
            # 陈旧度方差膨胀(41)：闲置越久探索 bonus 越大，封顶防失控
            dt = self.total_rounds - self.last_activity.get(c["model"],
                                                            self.total_rounds)
            stale = min(STALE_GROWTH ** max(dt, 0), STALE_MAX)
            exploration *= stale

            # Budget Pacer 成本惩罚(40/45)：−(λc·mult + λt)·c̃
            cost_pen = 0.0
            if self.budget > 0 or self.lambda_c > 0:
                # cheap 意图(借鉴点 32 落地)：临时抬高 λ_c，让"省钱"请求
                # 在池内显著偏向便宜臂。只影响本轮打分，不落盘、不影响 λ_t。
                # 注意：若 lambda_c=0（成本通道未启用），cheap 也无效。
                lambda_c_eff = self.lambda_c * (
                    CHEAP_LAMBDA_MULT if cheap else 1.0)
                cost_pen = (lambda_c_eff + self.lambda_t) * self._log_norm_cost(
                    self._est_cost(c["model"]))

            score = avg + exploration - cost_pen
            scored.append((score, avg, c))

        if not scored:
            return None

        # 6. tie-breaker(54)：top-2 打分差 < ε → 选历史 Q 均值高者，防抖动
        scored.sort(key=lambda s: s[0], reverse=True)
        if len(scored) >= 2 and (scored[0][0] - scored[1][0]) < self.tie_eps:
            pick = scored[0] if scored[0][1] >= scored[1][1] else scored[1]
        else:
            pick = scored[0]

        self._mark_play(pick[2]["model"])
        return pick[2]

    # ── 更新 ────────────────────────────────────────────────────────

    def update(self, model, success, total_tokens, task_type=None,
               quality=None, latency_ms=None, penalty=0.0):
        """更新模型统计（奖励四分量，借鉴点 49）。

        model:        模型名
        success:      API 调用是否成功
        total_tokens: 本次调用消耗的 token 数
        task_type:    任务类型，同时更新 overall 和对应专精
        quality:      质量分 [0,1]，None → 0.5 中性（等待 B1 负面反馈信号接入）
        latency_ms:   延迟（毫秒），None → 0 不惩罚
        penalty:      操作惩罚（403 扣 0.5 / 429 扣 0.3 / 被拒扣分，借鉴点 50）
                     失败时 reward = −|penalty|（区分严重度，且可恢复）；
                     成功时直接从奖励里扣 penalty。
        """
        # 1. 计算 reward（四分量加权，归一化到 [-1, 1] 附近匹配 UCB 理论假设）
        if success:
            q = 0.5 if quality is None else max(0.0, min(1.0, float(quality)))
            c_tilde = self._log_norm_cost(self._cost_of_tokens(model, total_tokens))
            l_tilde = min(1.0, (latency_ms or 0) / LATENCY_MAX)
            reward = self.w_q * q - self.w_c * c_tilde - self.w_l * l_tilde - penalty
        else:
            reward = 0.0 if penalty <= 0 else -abs(float(penalty))

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

        # 5. 全局计数器 + 活跃度
        self.total_rounds += 1
        self.last_activity[model] = self.total_rounds

        # 6. Budget Pacer λ 对偶上升(40)：EMA 成本 vs 预算 → λ 自动升降
        if self.budget and self.budget > 0:
            cost_t = self._cost_of_tokens(model, total_tokens)
            # 冷启动：前 3 轮 cost_ema 从预算起步，避免 λ 被初始 0 拉低
            if self.total_rounds <= 3 and self.cost_ema <= 0:
                self.cost_ema = self.budget
            self.cost_ema = (1.0 - ALPHA_EMA) * self.cost_ema + ALPHA_EMA * cost_t
            ratio = self.cost_ema / self.budget if self.budget else 1.0
            self.lambda_t = min(max(
                self.lambda_t + ETA * (ratio - 1.0), 0.0), LAMBDA_MAX)

        return reward

    def merge_tokens(self, model, extra_tokens, task_type=None):
        """工具续调补记：把同一轮对话里后续续调的 token 并入统计。

        首轮已通过 update() 记过一笔（pulls / reward / total_rounds 只算
        一次）；续调（携带工具结果的再次调用，token 量往往更大）只补
        total_tokens，不重复计数——否则一次对话会被统计成多次调用，
        均 token 口径反而失真。
        """
        if not extra_tokens:
            return
        o = self.stats.get(model, {}).get("overall")
        if o is None:
            return  # 没有首轮记录（异常路径），无从合并
        o["total_tokens"] += extra_tokens
        t = self.stats[model].get(task_type) if task_type else None
        if t:
            t["total_tokens"] += extra_tokens
        # 预算保护同样要看到这笔成本（与 update 相同的 EMA / λ 更新）
        if self.budget and self.budget > 0:
            cost_extra = self._cost_of_tokens(model, extra_tokens)
            self.cost_ema = (1.0 - ALPHA_EMA) * self.cost_ema + ALPHA_EMA * cost_extra
            ratio = self.cost_ema / self.budget if self.budget else 1.0
            self.lambda_t = min(max(
                self.lambda_t + ETA * (ratio - 1.0), 0.0), LAMBDA_MAX)

    # ── 序列化 ──────────────────────────────────────────────────────

    def to_dict(self):
        return {
            "stats": self.stats,
            "total_rounds": self.total_rounds,
            "c": self.c,
            "alpha": self.alpha,
            "base_reward": self.base_reward,
            "plays": self.plays,
            "last_activity": self.last_activity,
            "cost_ema": self.cost_ema,
            "lambda_t": self.lambda_t,
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
        b.plays = dict(d.get("plays") or {})
        b.last_activity = dict(d.get("last_activity") or {})
        b.cost_ema = float(d.get("cost_ema", 0.0))
        b.lambda_t = float(d.get("lambda_t", 0.0))

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
    bandit_config:  {"ucb_c": 2.0, "alpha": 0.00001, "base_reward": 100.0,
                     "budget": 0.002, "prices": {...}, ...} 或 None
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
        b.apply_config(cfg)
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
