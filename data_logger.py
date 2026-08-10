"""
分类数据落盘。每次分类结果追加写入 JSONL，为 bandit 和离线分析积累数据。

存储路径: <plugin_dir>/data/classifications.jsonl
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
_DATA_DIR = None


def _get_data_dir():
    global _DATA_DIR
    if _DATA_DIR is None:
        _DATA_DIR = Path(__file__).resolve().parent / "data"
    return _DATA_DIR


def log_classification(
    user_message,
    complexity,
    task_type,
    confidence,
    reasoning="",
    method="rule",
    latency_ms=0,
    model_routed_to=None,
    routing_success=None,
):
    """
    追加一条分类记录。

    user_message:  用户输入（截断到前 500 字符）
    complexity:    simple / medium / complex
    task_type:     chat / coding / reasoning / writing / analysis / translation / other
    confidence:    0.0 ~ 1.0
    reasoning:     分类理由简述
    method:        llm / rule
    latency_ms:    分类耗时（毫秒）
    model_routed_to:  最终路由到的模型名，可为 None
    routing_success:  路由调用是否成功，可为 None
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message[:500],
        "complexity": complexity,
        "task_type": task_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "method": method,
        "latency_ms": latency_ms,
        "model_routed_to": model_routed_to,
        "routing_success": routing_success,
    }

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "classifications.jsonl"

    with _LOCK:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
