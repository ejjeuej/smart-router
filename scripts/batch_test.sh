#!/bin/bash
# 批量发送 50 条测试对话给 Hermes
# 用法: bash scripts/batch_test.sh

PROMPTS=(
"你好"
"解释一下什么是蒙特卡洛方法"
"把人工智能翻译成英文"
"帮我写一段关于新产品的推广文案"
"写一个Python函数检查字符串是否是回文"
"什么是深度学习"
"Rust和Go有什么区别"
"写一首关于冬天的诗"
"帮我分析一下这段代码的时间复杂度：for i in range(n): for j in range(i, n): print(i, j)"
"什么是区块链"
"用递归实现斐波那契数列"
"解释一下什么是过拟合"
"翻译这句话到日语：今天天气很好"
"帮我写一封求职邮件"
"如何用Docker部署一个Flask应用"
"什么是正态分布"
"React和Vue的优缺点"
"写一个Shell脚本备份文件夹"
"解释一下TCP三次握手"
"Git merge和rebase的区别"
"今天天气怎么样"
"1加1等于几"
"解释一下量子力学的基本原理"
"把你好世界翻译成法语"
"帮我写一个请假条"
"什么是RESTful API"
"Python和JavaScript的区别"
"写一首关于春天的五言绝句"
"排序算法的时间复杂度对比"
"什么是微服务架构"
"用Python实现二分查找"
"解释一下什么是API"
"翻译成韩语：谢谢你的帮助"
"帮我写一份辞职信"
"什么是数据库索引"
"HTTP和HTTPS的区别"
"写一个正则表达式匹配邮箱地址"
"什么是敏捷开发"
"Linux常用命令有哪些"
"解释一下MVC模式"
"什么是SQL注入"
"用Shell脚本统计文件行数"
"写一首关于秋天的诗"
"什么是机器学习"
"C和C++的区别"
"帮我翻译成英文：深度学习是人工智能的一个分支"
"什么是设计模式"
"用Python写一个简单的爬虫"
"解释一下什么是云计算"
"什么是NoSQL数据库"
)

COUNT=0
for prompt in "${PROMPTS[@]}"; do
  COUNT=$((COUNT + 1))
  echo "[$COUNT/${#PROMPTS[@]}] $prompt"
  echo "$prompt" | hermes 2>/dev/null
  echo "---"
done

echo ""
echo "完成。跑 python3 scripts/bandit_report.py 看结果。"
