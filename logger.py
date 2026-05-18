"""
5 文件日志体系 — 每个 Agent 独立日志 + 一份完整汇总日志

输出目录:
    logs/agent/
        all.log           完整日志（所有 Agent 消息汇总）
        dispatcher.log    分配Agent
        query.log         查询Agent
        inspector.log     巡检Agent
        diagnosis.log     诊断Agent

控制台同步输出所有日志。

DEBUG 切换:
    from logger import set_agent_debug
    set_agent_debug("query")       # 只看查询Agent 的 debug
    set_agent_debug()              # 所有 Agent 开启 debug
    set_agent_info("query")        # 恢复为 INFO
"""

import logging
import sys
from pathlib import Path

from config import LOG_DIR, AGENT_NAMES, AGENT_LOG_LEVELS

_log_dir = None
_agent_files = {}
_combined_handler = None
_console_handler = None
_formatter = None


def _ensure_setup():
    global _log_dir, _agent_files, _combined_handler, _console_handler, _formatter
    if _log_dir is not None:
        return

    _log_dir = Path(LOG_DIR)
    _log_dir.mkdir(parents=True, exist_ok=True)

    _formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(_formatter)

    _combined_handler = logging.FileHandler(
        str(_log_dir / "all.log"), encoding="utf-8"
    )
    _combined_handler.setFormatter(_formatter)

    for name in AGENT_NAMES:
        _agent_files[name] = logging.FileHandler(
            str(_log_dir / f"{name}.log"), encoding="utf-8"
        )
        _agent_files[name].setFormatter(_formatter)


def get_agent_logger(name: str) -> logging.Logger:
    """
    获取指定 Agent 的 logger。

    每个 logger 同时写入:
        1. logs/agent/{name}.log  (Agent 独立日志)
        2. logs/agent/all.log     (完整汇总日志)
        3. 控制台 stdout
    """
    _ensure_setup()

    if name not in AGENT_NAMES:
        raise ValueError(f"未知 Agent: {name}，可选: {AGENT_NAMES}")

    full_name = f"agent.{name}"
    logger = logging.getLogger(full_name)

    if not logger.handlers:
        level_name = AGENT_LOG_LEVELS.get(name, "INFO")
        logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))

        logger.addHandler(_agent_files[name])
        logger.addHandler(_combined_handler)
        logger.addHandler(_console_handler)
        logger.propagate = False

    return logger


def get_rag_logger() -> logging.Logger:
    """
    RAG 模块专用 logger。
    写入 all.log + 控制台，不生成独立文件。
    """
    _ensure_setup()

    logger = logging.getLogger("rag")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(_combined_handler)
        logger.addHandler(_console_handler)
        logger.propagate = False

    return logger


def set_agent_debug(name: str | None = None):
    """
    设置指定 Agent 为 DEBUG 级别。

    set_agent_debug("query")   → 查询Agent 输出详细 debug 日志
    set_agent_debug()          → 所有 Agent 输出 debug
    """
    _ensure_setup()
    targets = AGENT_NAMES if name is None else [name]
    for n in targets:
        logger = logging.getLogger(f"agent.{n}")
        logger.setLevel(logging.DEBUG)
        AGENT_LOG_LEVELS[n] = "DEBUG"


def set_agent_info(name: str | None = None):
    """
    恢复指定 Agent 为 INFO 级别。

    set_agent_info("query")    → 查询Agent 恢复 INFO
    set_agent_info()           → 所有 Agent 恢复 INFO
    """
    _ensure_setup()
    targets = AGENT_NAMES if name is None else [name]
    for n in targets:
        logger = logging.getLogger(f"agent.{n}")
        logger.setLevel(logging.INFO)
        AGENT_LOG_LEVELS[n] = "INFO"
