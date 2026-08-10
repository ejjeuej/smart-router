#!/usr/bin/env python3
"""
smart-router 老虎机全链路测试 v2

直接导入你的 bandit.py，模拟真实 MaaS 错误模式 + 黑名单逻辑。
测试 4 个场景，输出决策建议。

用法:  cd ~/.hermes/plugins/smart-router && python3 scripts/test_bandit.py
"""

import sys, os, math, random, copy
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class C:
    G="\033[92m";R="\033[91m";Y="\033[93m";CY="\033[96m";B="\033[1m";D="\033[2m";E="\033[0m"

def hdr(s):  print(f"\n{C.B}{C.CY}{'='*60}{C.E}\n{C.B}{C.CY}  {s}{C.E}\n{C.B}{C.CY}{'='*60}{C.E}")

# ============================================================
# MaaS 模拟
# ============================================================
class MaaSModel:
    def __init__(self, name, quality, tokens=3000, max_tokens=65536,
                 quota_exhausted=False, rate_limited_prob=0.0,
                 server_error_prob=0.0, enable_thinking_ok=True):
        self.name = name; self.quality = quality; self.tokens = tokens
        self.max_tokens = max_tokens; self.quota_exhausted = quota_exhausted
        self.rate_limited_prob = rate_limited_prob
        self.server_error_prob = server_error_prob
        self.enable_thinking_ok = enable_thinking_ok

    def get_quality(self, task_type="general"):
        if isinstance(self.quality, dict):
            return self.quality.get(task_type, 0.5)
        return self.quality

    def call(self, request_max_tokens=65536, task_type="general",
             has_enable_thinking=False):
        if self.quota_exhausted:
            return False, 0, 0, "403"
        if request_max_tokens > self.max_tokens:
            return False, 0, 0, "400"
        if has_enable_thinking and not self.enable_thinking_ok:
            return False, 0, 0, "400"
        if random.random() < self.rate_limited_prob:
            return False, 0, 0, "429"
        if random.random() < self.server_error_prob:
            return False, 0, 0, "5xx"
        q = self.get_quality(task_type)
        reward = q + random.gauss(0, 0.03)
        reward = max(0.01, min(1.0, reward))
        tokens = self.tokens + random.randint(-200, 200)
        return True, reward, tokens, None


# ============================================================
# 仿真（含黑名单逻辑，匹配 __init__.py）
# ============================================================
def simulate(bandit_cls, models, rounds, task_types):
    bandit = bandit_cls(c=2.0, alpha=1e-5, base_reward=100.0)
    exhausted = set()    # 永久拉黑 (403/402)
    round_bl = set()     # 本轮拉黑 (429/5xx)
    tried = set()

    selections = []
    rewards = []
    errors = Counter()

    for t in range(rounds):
        tt = task_types[t]
        round_bl.clear()
        tried.clear()

        # 可用候选（排除永久拉黑）
        active_models = [m for m in models if m.name not in exhausted]
        if not active_models:
            continue

        candidates = [{"model": m.name, "base_url": "", "api_key": ""}
                      for m in active_models]
        selected = bandit.select(candidates, task_type=tt)
        tried.add(selected["model"])
        model = next(m for m in active_models if m.name == selected["model"])

        success, reward, tokens, err = model.call(
            task_type=tt, has_enable_thinking=(not model.enable_thinking_ok))

        if err:
            errors[err] += 1

        if err in ("403", "402"):
            exhausted.add(model.name)
        elif err in ("429", "500", "502", "503", "504"):
            round_bl.add(model.name)  # 本轮不重试
        elif err == "400":
            bandit.update(model.name, False, 0, tt)
        else:
            bandit.update(model.name, True, int(tokens), tt)

        selections.append(selected["model"])
        rewards.append(reward if success else 0.0)

    return _compute_metrics(selections, rewards, models, task_types, errors, rounds)


def _compute_metrics(selections, rewards, models, task_types, errors, rounds):
    n = len(selections)
    if n == 0:
        return {"success_rate": 0, "optimal_rate": 0, "cum_regret": 0,
                "convergence": None, "selections": [], "errors": errors}

    # 最优选择率
    optimal_count = 0
    for i in range(n):
        tt = task_types[i]
        best_name = max(models, key=lambda m: m.get_quality(tt)).name
        if selections[i] == best_name:
            optimal_count += 1

    # Regret
    regrets = []
    for i in range(n):
        tt = task_types[i]
        best_q = max(m.get_quality(tt) for m in models)
        regrets.append(best_q - rewards[i])

    # 收敛
    convergence = None
    for start in range(0, n - 20, 5):
        cnt = 0
        for j in range(start, start + 20):
            tt = task_types[j]
            best_name = max(models, key=lambda m: m.get_quality(tt)).name
            if selections[j] == best_name:
                cnt += 1
        if cnt / 20 > 0.7:
            convergence = start
            break

    return {
        "success_rate": sum(1 for s in selections
                           if any(m.name == s for m in models)) / n * 100,
        "optimal_rate": optimal_count / n * 100,
        "cum_regret": sum(regrets),
        "convergence": convergence,
        "selections": selections,
        "errors": errors,
    }


# ============================================================
# 场景
# ============================================================
def s1_your_config():
    """你的实际配置"""
    simple = [
        MaaSModel("qwen3.6-27b",     0.88, 2000),
        MaaSModel("qwen-max",         0.92, 3500),
        MaaSModel("qwen3.6-35b-a3b",  0.90, 2500),
        MaaSModel("qwen3.5-122b-a10b",0.91, 3000),
        MaaSModel("qwen3.5-397b-a17b",0.93, 2800),
    ]
    complex_ = [
        MaaSModel("qwen3.6-plus",    0.94, 4000),
        MaaSModel("qwen3.7-max",     0.96, 5000),
        MaaSModel("glm-5",           0.93, 4500),
        MaaSModel("deepseek-v3",     0.95, 4200),
        MaaSModel("qwen3-coder-plus",0.94, 3800),
    ]
    stt = ["chat","translation","other"]
    ctt = ["coding","reasoning","analysis","writing"]
    tt = [random.choice(stt) for _ in range(300)] + \
         [random.choice(ctt) for _ in range(200)]
    return simple, complex_, tt


def s2_multitask():
    models = [
        MaaSModel("coder-pro", {"coding":0.95,"chat":0.35}, 4000),
        MaaSModel("chat-pro",  {"coding":0.35,"chat":0.95}, 2000),
        MaaSModel("balanced",  {"coding":0.65,"chat":0.65}, 3000),
    ]
    tt = ["coding" if i%2==0 else "chat" for i in range(500)]
    return models, tt


def s3_error_handling():
    models = [
        MaaSModel("good-A",       0.92, 3000),
        MaaSModel("good-B",       0.89, 2000),
        MaaSModel("dead-quota",   0.99, 1000, quota_exhausted=True),
        MaaSModel("small-cap",    0.95, 1500, max_tokens=4096),
        MaaSModel("flaky-30",     0.87, 2500, server_error_prob=0.3),
    ]
    tt = ["coding" if i%3==0 else "chat" for i in range(500)]
    return models, tt


def s4_stationary():
    models = [
        MaaSModel("A", 0.95, 3000),
        MaaSModel("B", 0.85, 2000),
        MaaSModel("C", 0.75, 2500),
        MaaSModel("D", 0.65, 1500),
        MaaSModel("E", 0.55, 1000),
    ]
    tt = ["general"]*500
    return models, tt


# ============================================================
# 报告
# ============================================================
def report(name, result, models, rounds):
    print(f"\n  {C.Y}--- {name} ---{C.E}")

    ok = lambda v,t: C.G if v>t else C.R
    print(f"  成功率:     {ok(result['success_rate'],90)}{result['success_rate']:.0f}%{C.E}")
    print(f"  最优选择率: {ok(result['optimal_rate'],65)}{result['optimal_rate']:.0f}%{C.E}")
    print(f"  累计Regret: {result['cum_regret']:.1f}")

    conv = result["convergence"]
    if conv is not None:
        print(f"  收敛轮次:   {conv}  {'✓' if conv<100 else '✗ 偏慢'}")
    else:
        print(f"  收敛轮次:   {C.R}未收敛{C.E}")

    dist = Counter(result["selections"])
    print(f"\n  选择分布:")
    for m in models:
        c = dist.get(m.name, 0)
        bar = "█" * int(c / max(rounds, 1) * 30)
        print(f"    {m.name:<22} {c:>4}  {bar}")

    if result["errors"]:
        print(f"\n  错误分布: {dict(result['errors'])}")


# ============================================================
# 主函数
# ============================================================
def main():
    random.seed(42)
    from bandit import UCBBandit, reset_bandit
    for key in list(sys.modules["bandit"]._bandits.keys()):
        reset_bandit(key)

    hdr("smart-router 老虎机测试")
    print(f"\n  导入: bandit.py 的 UCBBandit")
    print(f"  参数: c=2.0, alpha=1e-5, base_reward=100.0")
    print(f"  Hermes max_tokens=65536")

    # ── 1. 平稳环境（收敛能力） ──
    hdr("场景1: 平稳环境（5模型质量不变）")
    models, tt = s4_stationary()
    r = simulate(UCBBandit, models, 500, tt)
    report("平稳环境", r, models, 500)

    # ── 2. 你的实际配置 ──
    hdr("场景2: 你的实际双池配置")
    print(f"  Simple:  qwen3.6-27b/qwen-max/qwen3.6-35b-a3b/qwen3.5-122b-a10b/qwen3.5-397b-a17b")
    print(f"  Complex: qwen3.6-plus/qwen3.7-max/glm-5/deepseek-v3/qwen3-coder-plus")
    simple, complex_, tt = s1_your_config()
    stt = [t for t in tt[:300]]
    ctt = [t for t in tt[300:]]
    r_s = simulate(UCBBandit, simple, 300, stt)
    r_c = simulate(UCBBandit, complex_, 200, ctt)
    report("Simple池", r_s, simple, 300)
    report("Complex池", r_c, complex_, 200)

    # ── 3. 错误处理 ──
    hdr("场景3: 错误处理")
    print(f"  5模型: 2个好 + 1个403额度耗尽 + 1个400(max_tokens) + 1个30%故障")
    models, tt = s3_error_handling()
    r = simulate(UCBBandit, models, 500, tt)
    report("错误处理", r, models, 500)

    # ── 4. 多任务混合奖励 ──
    hdr("场景4: 多任务混合奖励")
    print(f"  coder-pro擅长coding(0.95)弱于chat(0.35)")
    print(f"  chat-pro擅长chat(0.95)弱于coding(0.35)")
    print(f"  期望: coding任务选coder-pro, chat任务选chat-pro")
    models, tt = s2_multitask()
    r = simulate(UCBBandit, models, 500, tt)
    report("多任务", r, models, 500)

    # 多任务细分
    dist_coding = Counter()
    dist_chat = Counter()
    for i, s in enumerate(r["selections"]):
        if tt[i] == "coding": dist_coding[s] += 1
        else: dist_chat[s] += 1
    print(f"\n  coding任务选模型: {dict(dist_coding)}")
    print(f"  chat任务选模型:   {dict(dist_chat)}")

    # ── 总结 ──
    hdr("总结")
    print(f"""
  怎么读:

  场景1 平稳环境: UCB应该快速收敛到最优模型A(0.95)
    成功率>90% + 收敛<100轮 → 算法收敛能力OK

  场景2 你的配置: 模拟两个池独立运行
    Simple池选qwen3.5-397b-a17b(0.93最优)
    Complex池选qwen3.7-max(0.96最优)
    两者都>85%选择率 → 参数合理

  场景3 错误处理: dead-quota(403)应只被选1次然后拉黑
    small-cap(400)应被选但标记失败
    最终只有2个好模型被选中

  场景4 多任务: coding→coder-pro, chat→chat-pro
    分开统计两个方向的分布
""")

    print("  终端运行: cd ~/.hermes/plugins/smart-router && python3 scripts/test_bandit.py")


if __name__ == "__main__":
    main()
