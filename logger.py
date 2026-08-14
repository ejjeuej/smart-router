"""
smart-router 运行日志模块。

解决 Windows 打包 App 中插件的 stderr/stdout 被吞掉、用户看不到调试信息的问题。
日志写入 Hermes 数据目录下的 logs/smart-router.log（与 agent.log 同目录），
用户可以直接用文本编辑器打开查看。
"""

import threading
from datetime import datetime, timezone
from pathlib import Path

from config import _hermes_home

_LOCK = threading.Lock()
_LOG_PATH = None


def _get_log_path() -> Path:
    """定位日志文件: <hermes_home>/logs/smart-router.log"""
    global _LOG_PATH
    if _LOG_PATH is None:
        log_dir = _hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / "smart-router.log"
    return _LOG_PATH


def _write(level: str, msg: str):
    """线程安全地追加一行日志;失败时静默丢弃,绝不影响主流程。"""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}\n"
        with _LOCK:
            with open(_get_log_path(), "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def debug(msg: str):
    _write("DEBUG", msg)


def info(msg: str):
    _write("INFO", msg)


def warning(msg: str):
    _write("WARN", msg)


def error(msg: str):
    _write("ERROR", msg)
