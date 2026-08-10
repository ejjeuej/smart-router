#!/usr/bin/env python3
"""
批量发送提示词给 Hermes（每条独立会话）。

用法:
  python3 scripts/batch_test.py
"""

import subprocess
import time
import sys
from pathlib import Path

# 强制重置 bandit 数据，防止 gateway 进程残留数据干扰
# 注意：只删 bandit_*.json，不碰 classifications.jsonl（ML 训练数据）
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
from bandit import reset_bandit
reset_bandit("simple_models")
reset_bandit("complex_models")

HERMES = "/home/linzizhou/Hermes_code/hermes-agent/.venv/bin/hermes"

PROMPTS = [
    "你好",
    "解释一下什么是蒙特卡洛方法",
    "把人工智能翻译成英文",
    "帮我写一段关于新产品的推广文案",
    "写一个Python函数检查字符串是否是回文",
    "什么是深度学习",
    "Rust和Go有什么区别",
    "写一首关于冬天的诗",
    "帮我分析一下这段代码的时间复杂度：for i in range(n): for j in range(i, n): print(i, j)",
    "什么是区块链",
    "用递归实现斐波那契数列",
    "解释一下什么是过拟合",
    "翻译这句话到日语：今天天气很好",
    "帮我写一封求职邮件",
    "如何用Docker部署一个Flask应用",
    "什么是正态分布",
    "React和Vue的优缺点",
    "写一个Shell脚本备份文件夹",
    "解释一下TCP三次握手",
    "Git merge和rebase的区别",
    "今天天气怎么样",
    "1加1等于几",
    "解释一下量子力学的基本原理",
    "把你好世界翻译成法语",
    "帮我写一个请假条",
    "什么是RESTful API",
    "Python和JavaScript的区别",
    "写一首关于春天的五言绝句",
    "排序算法的时间复杂度对比",
    "什么是微服务架构",
    "用Python实现二分查找",
    "解释一下什么是API",
    "翻译成韩语：谢谢你的帮助",
    "帮我写一份辞职信",
    "什么是数据库索引",
    "HTTP和HTTPS的区别",
    "写一个正则表达式匹配邮箱地址",
    "什么是敏捷开发",
    "Linux常用命令有哪些",
    "解释一下MVC模式",
    "什么是SQL注入",
    "用Shell脚本统计文件行数",
    "写一首关于秋天的诗",
    "什么是机器学习",
    "C和C++的区别",
    "帮我翻译成英文：深度学习是人工智能的一个分支",
    "什么是设计模式",
    "用Python写一个简单的爬虫",
    "解释一下什么是云计算",
    "什么是NoSQL数据库",
    "解释一下CAP定理",
    "用Python实现快速排序",
    "什么是负载均衡",
    "Redis和Memcached的区别",
    "写一个Python函数计算阶乘",
    "什么是操作系统",
    "TCP和UDP的区别",
    "帮我写一个技术方案文档",
    "什么是虚拟DOM",
    "解释一下什么是闭包",
    "用SQL查询第二高的工资",
    "什么是CDN",
    "Docker和虚拟机的区别",
    "写一个Python脚本监控CPU使用率",
    "解释一下什么是多线程",
    "什么是OAuth2.0",
    "用JavaScript写一个防抖函数",
    "什么是Kubernetes",
    "写一首关于夏天的诗",
    "帮我翻译成日语：机器学习是人工智能的一个分支",
    "什么是进程和线程的区别",
    "用Python实现冒泡排序",
    "什么是CI/CD",
    "正则表达式怎么写",
    "解释一下什么是哈希表",
    "什么是DNS",
    "写一个Python函数判断素数",
    "Redux和MobX有什么区别",
    "什么是WebSocket",
    "用Shell脚本批量重命名文件",
    "什么是死锁",
    "解释一下什么是协程",
    "写一封英文邮件给客户",
    "帮我分析冒泡排序和快速排序的性能差异",
    "什么是内存泄漏",
    "Python装饰器是什么",
    "用Python实现LRU缓存",
    "解释一下WebRTC",
    "写一首关于月亮的诗",
    "什么是事件驱动架构",
    "Git stash的用法",
    "帮我写一个项目README模板",
    "什么是CORS",
    "解释一下零拷贝",
    "什么是JWT",
    "用Python写一个简单的Web服务器",
    "解释一下A星寻路算法",
    "写一段产品功能介绍文案",
    "什么是分布式锁",
]

# 记录运行前 classifications 行数，避免历史数据干扰计数
_classifications_path = _plugin_dir / "data" / "classifications.jsonl"
_classifications_before = 0
if _classifications_path.exists():
    with open(_classifications_path) as f:
        _classifications_before = sum(1 for _ in f)

total = len(PROMPTS)
success_count = 0
fail_count = 0
timeout_count = 0

for i, prompt in enumerate(PROMPTS, 1):
    print(f"[{i}/{total}] {prompt[:50]}... ", end="", flush=True)
    try:
        result = subprocess.run(
            [HERMES, "-z", prompt, "--yolo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            text=True,
        )
        if result.returncode == 0:
            print("✓ 完成")
            success_count += 1
        else:
            print(f"✗ 失败 (exit {result.returncode})")
            # 提取最后一行 stderr 作为错误摘要
            err_lines = result.stderr.strip().split("\n")
            last_err = err_lines[-1][:120] if err_lines else "unknown error"
            print(f"  └─ {last_err}")
            fail_count += 1
    except subprocess.TimeoutExpired:
        print("⏰ 超时 (>300s)")
        timeout_count += 1
    except Exception as e:
        print(f"✗ 异常: {e}")
        fail_count += 1

print()
print(f"结果: {success_count} 成功 / {fail_count} 失败 / {timeout_count} 超时")

# 增量分类计数（不碰 classifications.jsonl）
_classifications_after = 0
if _classifications_path.exists():
    with open(_classifications_path) as f:
        _classifications_after = sum(1 for _ in f)
_classifications_delta = _classifications_after - _classifications_before
print(f"分类记录: {_classifications_delta} 条新增 (文件共 {_classifications_after} 条)")
print("跑 python3 scripts/bandit_report.py 看结果。")
