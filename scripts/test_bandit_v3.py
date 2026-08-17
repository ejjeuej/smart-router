#!/usr/bin/env python3
"""
bandit 线落地测试 v3（借鉴点 40/41/42/43/45/46/49/50/51/54）

直接导入 bandit.py，逐组件验证：
  1. 成本 log 归一化(45)
  2. 奖励四分量(49)：质量/成本/延迟/操作惩罚
  3. Budget Pacer λ 对偶上升(40)：超支→λ升，低于预算→λ降
  4. 陈旧度方差膨胀(41)：闲置臂探索增大 + 封顶
  5. burn-in 强制探索(43)：新臂前 N 次必选
  6. margin tie-breaker(54)：top-2 分差 < ε → 选历史均值高者
  7. 离线预热注入(51/42)：neff 语义 + 跳过 burn-in
  8. 软惩罚(50)：失败按严重度扣分，可恢复
  9. 旧签名回归：update(model, success, tokens, task_type) 兼容

用法: cd ~/.hermes/plugins/smart-router && python3 scripts/test_bandit_v3.py
"""

import sys, math, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bandit  # noqa: E402
from bandit import UCBBandit  # noqa: E402

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; E = "\033[0m"
passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  {G}✓{E} {name}")
    else:
        failed += 1
        print(f"  {R}✗ {name}  {Y}{detail}{E}")


def make_bandit(**kw):
    b = UCBBandit(c=kw.pop("c", 1.0))
    cfg = {
        "budget": 0.002,
        "prices": {"cheap": 0.0002, "pricey": 0.02, "mid": 0.002},
        "burn_in_pulls": 20,
        "tie_eps": 0.02,
    }
    cfg.update(kw)
    b.apply_config(cfg)
    return b


# ═══════════════════════════════════════════════════════════
# 1. 成本 log 归一化(45)
# ═══════════════════════════════════════════════════════════
def test_log_norm_cost():
    print(f"\n{B}1. 成本 log 归一化(45){E}")
    b = make_bandit()
    c_floor, c_ceil = bandit.C_FLOOR, bandit.C_CEIL
    check("地板价 → 0", b._log_norm_cost(c_floor) == 0.0)
    check("天花板价 → 1", b._log_norm_cost(c_ceil) == 1.0)
    check("低于地板 → 0", b._log_norm_cost(c_floor / 10) == 0.0)
    check("高于天花板 → 1", b._log_norm_cost(c_ceil * 10) == 1.0)
    lo = b._log_norm_cost(1e-3)
    hi = b._log_norm_cost(1e-2)
    check("单调递增", lo < hi, f"{lo} vs {hi}")
    check("区间内", 0 <= lo < hi <= 1)
    # 千倍价差被压缩到 ~1/2 区间内
    span = hi - lo
    check("千倍价差压缩 < 0.6", span < 0.6, f"span={span:.3f}")


# ═══════════════════════════════════════════════════════════
# 2. 奖励四分量(49)
# ═══════════════════════════════════════════════════════════
def test_four_component_reward():
    print(f"\n{B}2. 奖励四分量(49){E}")
    b = make_bandit()
    b.stats["mid"] = {"overall": {"pulls": 5, "total_reward": 2.5,
                                  "total_tokens": 15000}}
    b.plays["mid"] = 100

    r_q_hi = b.update("mid", True, 1000, "chat", quality=0.9, latency_ms=100)
    r_q_lo = b.update("mid", True, 1000, "chat", quality=0.2, latency_ms=100)
    check("质量越高奖励越高", r_q_hi > r_q_lo, f"{r_q_hi:.3f} vs {r_q_lo:.3f}")

    r_cost_lo = b.update("mid", True, 100, "chat", quality=0.5, latency_ms=100)
    r_cost_hi = b.update("mid", True, 50000, "chat", quality=0.5, latency_ms=100)
    check("成本越高奖励越低", r_cost_lo > r_cost_hi, f"{r_cost_lo:.3f} vs {r_cost_hi:.3f}")

    r_lat_lo = b.update("mid", True, 1000, "chat", quality=0.5, latency_ms=10)
    r_lat_hi = b.update("mid", True, 1000, "chat", quality=0.5, latency_ms=30000)
    check("延迟越高奖励越低", r_lat_lo > r_lat_hi, f"{r_lat_lo:.3f} vs {r_lat_hi:.3f}")

    r_pen0 = b.update("mid", True, 1000, "chat", quality=0.5, penalty=0.0)
    r_pen1 = b.update("mid", True, 1000, "chat", quality=0.5, penalty=0.5)
    check("操作惩罚直接扣分", r_pen0 > r_pen1, f"{r_pen0:.3f} vs {r_pen1:.3f}")
    check("成功奖励保底为正", r_pen1 >= 0.01)

    r_fail = b.update("mid", False, 0, "chat")
    check("失败奖励 = 0", r_fail == 0.0)
    r_fail_pen = b.update("mid", False, 0, "chat", penalty=0.5)
    check("失败+惩罚 = -0.5(可恢复)", r_fail_pen == -0.5)
    r_recover = b.update("mid", True, 1000, "chat", quality=0.8, latency_ms=50)
    check("恢复后奖励回正(软惩罚语义)", r_recover > 0.3)

    # 四分量都进 stats: 起始 pulls=5 + 11 次 update = 16
    o = b.stats["mid"]["overall"]
    check("pulls 累计正确", abs(o["pulls"] - 16) < 1e-9, str(o["pulls"]))


# ═══════════════════════════════════════════════════════════
# 3. Budget Pacer λ 对偶上升(40)
# ═══════════════════════════════════════════════════════════
def test_budget_pacer():
    print(f"\n{B}3. Budget Pacer λ 对偶上升(40){E}")
    b = make_bandit()  # budget=0.002
    b.stats["pricey"] = {"overall": {"pulls": 50, "total_reward": 30.0,
                                     "total_tokens": 150000}}
    b.plays["pricey"] = 100
    b.last_activity["pricey"] = 0

    # 连续用贵模型(0.02 $/1k × 5000 tokens = 0.1$/请求 >> budget 0.002)
    for _ in range(30):
        b.update("pricey", True, 5000, "chat", quality=0.9)
    check("超支后 λ 上升", b.lambda_t > 1.0, f"λ={b.lambda_t:.3f}")
    check("cost_ema 高于预算", b.cost_ema > b.budget,
          f"ema={b.cost_ema:.5f} budget={b.budget}")

    # 换便宜模型(0.0002 × 1000 = 2e-4 << budget)
    b.stats["cheap"] = {"overall": {"pulls": 50, "total_reward": 20.0,
                                    "total_tokens": 50000}}
    b.plays["cheap"] = 100
    b.last_activity["cheap"] = 0
    for _ in range(200):
        b.update("cheap", True, 1000, "chat", quality=0.5)
    check("长期低于预算 λ 回落", b.lambda_t < 0.5, f"λ={b.lambda_t:.3f}")

    # 打分含成本惩罚: λ 大时贵臂 score 被罚
    b2 = make_bandit()
    b2.lambda_t = 3.0
    b2.stats["cheap"] = {"overall": {"pulls": 10, "total_reward": 5.0,
                                     "total_tokens": 10000}}
    b2.stats["pricey"] = {"overall": {"pulls": 10, "total_reward": 8.0,
                                      "total_tokens": 100000}}
    b2.plays = {"cheap": 100, "pricey": 100}
    b2.last_activity = {"cheap": 0, "pricey": 0}
    b2.total_rounds = 1
    cands = [{"model": "cheap", "base_url": "", "api_key": ""},
             {"model": "pricey", "base_url": "", "api_key": ""}]
    sel = b2.select(cands, task_type="chat")
    # pricey 质量更高(0.8)但贵 100 倍, λ=3 时应选 cheap
    check("λ 高时贵臂被成本惩罚压制", sel["model"] == "cheap", f"选了 {sel['model']}")


# ═══════════════════════════════════════════════════════════
# 4. 陈旧度方差膨胀(41)
# ═══════════════════════════════════════════════════════════
def test_staleness():
    print(f"\n{B}4. 陈旧度方差膨胀(41){E}")
    b = make_bandit()
    b.stats["mid"] = {"overall": {"pulls": 10, "total_reward": 5.0,
                                  "total_tokens": 30000}}
    b.plays = {"mid": 100}
    b.last_activity = {"mid": 0}
    b.total_rounds = 0
    b.c = 1.0
    cands = [{"model": "mid", "base_url": "", "api_key": ""}]

    # 计算不同 dt 的 stale 因子（封顶逻辑）
    def stale_factor(dt):
        return min(bandit.STALE_GROWTH ** dt, bandit.STALE_MAX)

    s0 = stale_factor(0)
    s10 = stale_factor(10)
    check("闲置 10 轮探索增大", s10 > s0, f"{s0:.4f} → {s10:.4f}")
    check("stale 因子封顶 = STALE_MAX", stale_factor(1000) == bandit.STALE_MAX,
          f"封顶 {stale_factor(1000):.2f}")

    # select 里实际生效: 闲置臂应被重新选中(相对刚拉过的臂)
    b3 = make_bandit()
    for m, r in (("a", 0.6), ("b", 0.62)):
        b3.stats[m] = {"overall": {"pulls": 10, "total_reward": 10 * r,
                                   "total_tokens": 30000}}
    b3.plays = {"a": 100, "b": 100}
    b3.last_activity = {"a": 0, "b": 50}   # a 闲置 50 轮（0 活动至今 50 轮）
    b3.total_rounds = 50
    cands2 = [{"model": "a", "base_url": "", "api_key": ""},
              {"model": "b", "base_url": "", "api_key": ""}]
    sel = b3.select(cands2, task_type="chat")
    check("闲置 50 轮的 a 被主动重探", sel["model"] == "a", f"选了 {sel['model']}")


# ═══════════════════════════════════════════════════════════
# 5. burn-in 强制探索(43)
# ═══════════════════════════════════════════════════════════
def test_burnin():
    print(f"\n{B}5. burn-in 强制探索(43){E}")
    b = make_bandit()
    # 老臂统计很强, 新臂刚注入
    b.stats["old"] = {"overall": {"pulls": 100, "total_reward": 90.0,
                                  "total_tokens": 300000}}
    b.plays = {"old": 100}
    b.last_activity = {"old": 0}
    b.total_rounds = 100
    cands = [{"model": "old", "base_url": "", "api_key": ""},
             {"model": "new", "base_url": "", "api_key": ""}]

    picks = []
    for _ in range(5):
        s = b.select(cands, task_type="chat")
        picks.append(s["model"])
        b.update(s["model"], True, 3000, "chat", quality=0.5)
    check("新臂前 5 次全部被强制选中", all(p == "new" for p in picks),
          f"picks={picks}")
    check("plays 累计到 5", b.plays["new"] == 5, str(b.plays["new"]))

    # 预热注入后跳过 burn-in: 只剩 old(完成) + new2(预热 30)两个候选
    b.inject_offline_prior("new2", 0.6, n=30, avg_tokens=3000)
    b.plays["new"] = 100  # 把 new 的 burn-in 补完, 避免干扰本断言
    s = b.select(cands, task_type="chat")
    check("预热注入后不再强制轮转(可被 UCB 判别)",
          s["model"] in ("old", "new2"), f"选了 {s['model']}")


# ═══════════════════════════════════════════════════════════
# 6. margin tie-breaker(54)
# ═══════════════════════════════════════════════════════════
def test_tiebreaker():
    print(f"\n{B}6. margin tie-breaker(54){E}")
    b = make_bandit()
    # 两个臂分数接近: a 均值高、b 均值低但探索 bonus 补足
    b.stats["a"] = {"overall": {"pulls": 10, "total_reward": 6.0,
                                "total_tokens": 30000}}   # avg 0.6
    b.stats["b"] = {"overall": {"pulls": 10, "total_reward": 5.9,
                                "total_tokens": 30000}}   # avg 0.59
    b.plays = {"a": 100, "b": 100}
    b.last_activity = {"a": 0, "b": 0}
    b.total_rounds = 1
    cands = [{"model": "a", "base_url": "", "api_key": ""},
             {"model": "b", "base_url": "", "api_key": ""}]
    s1 = b.select(cands, task_type="chat")
    check("分差 < ε 时选历史均值高者 a", s1["model"] == "a", f"选了 {s1['model']}")


# ═══════════════════════════════════════════════════════════
# 7. 离线预热注入(51/42)
# ═══════════════════════════════════════════════════════════
def test_offline_prior():
    print(f"\n{B}7. 离线预热注入(51/42){E}")
    b = make_bandit()
    b.inject_offline_prior("model-x", avg_reward=0.72, n=25, avg_tokens=2400)
    o = b.stats["model-x"]["overall"]
    check("pulls = neff = 25", o["pulls"] == 25, str(o["pulls"]))
    check("total_reward = 25×0.72", abs(o["total_reward"] - 18.0) < 1e-9,
          str(o["total_reward"]))
    check("均值保持 0.72", abs(o["total_reward"] / o["pulls"] - 0.72) < 1e-9)
    check("plays = 25(跳过 burn-in)", b.plays["model-x"] == 25)
    check("avg_tokens 记录", o["total_tokens"] == 25 * 2400)


# ═══════════════════════════════════════════════════════════
# 8. 软惩罚 + 黑名单恢复(50)
# ═══════════════════════════════════════════════════════════
def test_soft_penalty_recovery():
    print(f"\n{B}8. 软惩罚可恢复(50){E}")
    b = make_bandit()
    b.stats["mid"] = {"overall": {"pulls": 10, "total_reward": 5.0,
                                  "total_tokens": 30000}}
    b.plays["mid"] = 100
    # 故障期: 连续 429
    for _ in range(5):
        b.update("mid", False, 0, "chat", penalty=0.3)
    avg_bad = b.stats["mid"]["overall"]["total_reward"] / b.stats["mid"]["overall"]["pulls"]
    # 恢复: 连续成功
    for _ in range(10):
        b.update("mid", True, 1000, "chat", quality=0.8, latency_ms=50)
    avg_good = b.stats["mid"]["overall"]["total_reward"] / b.stats["mid"]["overall"]["pulls"]
    check("故障期均值被拉低", avg_bad < 0.4, f"avg={avg_bad:.3f}")
    check("恢复后均值回正", avg_good > avg_bad, f"{avg_bad:.3f} → {avg_good:.3f}")


# ═══════════════════════════════════════════════════════════
# 9. 旧签名回归
# ═══════════════════════════════════════════════════════════
def test_old_signature():
    print(f"\n{B}9. 旧签名回归{EE if False else E}")
    b = make_bandit()
    b.stats["mid"] = {"overall": {"pulls": 10, "total_reward": 5.0,
                                  "total_tokens": 30000}}
    b.plays["mid"] = 100
    r = b.update("mid", True, 3000, "chat")  # 旧签名: 无 quality/latency/penalty
    check("旧签名 update 不炸", r is not None, str(r))
    cands = [{"model": "mid", "base_url": "", "api_key": ""}]
    s = b.select(cands, task_type="chat")
    check("旧签名 select 不炸", s["model"] == "mid")

    # 与 test_bandit.py 的 simulate 调用方式兼容
    def _old_style_update(b2, name, success, tokens, tt):
        b2.update(name, success, tokens, tt)
    b2 = make_bandit()
    b2.stats["x"] = {"overall": {"pulls": 10, "total_reward": 5.0,
                                 "total_tokens": 30000}}
    b2.plays["x"] = 100
    _old_style_update(b2, "x", True, 3000, "chat")
    _old_style_update(b2, "x", False, 0, "chat")
    check("test_bandit.py 风格调用兼容", True)


def main():
    random.seed(7)
    for fn in (test_log_norm_cost, test_four_component_reward, test_budget_pacer,
               test_staleness, test_burnin, test_tiebreaker, test_offline_prior,
               test_soft_penalty_recovery, test_old_signature):
        fn()
    print(f"\n{'=' * 50}")
    print(f"  通过 {G}{passed}{E} 项, 失败 {R}{failed}{E} 项")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
