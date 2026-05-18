"""
项目级配置常量 — 集中管理所有跨模块共享的常量

路径常量:
    LOG_DIR              日志输出目录（5 文件日志体系）
    MCP_SERVER_URL       内嵌 MCP Server 地址
    MCP_LOG_DIR          MCP 调用日志目录
    BASE_DIR             项目根目录
    CONFIG_DIR           配置目录
    CONVERSATIONS_DIR    会话历史存储目录
    INSPECTOR_REPORTS_DIR 巡检报告文件目录
    WORK_RECORDS_DIR     工作记录文件目录
    VECTOR_DB_DIR        ChromaDB 向量数据目录

运行时常量:
    AGENT_COUNT          当前系统 Agent 总数
    AGENT_NAMES          所有 Agent 名称列表（供 logger 自动注册）
    AGENT_LOG_LEVELS     每个 Agent 的默认日志级别
    DIAGNOSIS_TIMEOUT    诊断Agent 每轮并行采集超时（秒）
    DIAGNOSIS_MAX_ROUNDS 诊断Agent 最大诊断轮数
"""

from pathlib import Path

# ============================================================
# 路径常量
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

LOG_DIR = "logs/agent"
MCP_LOG_DIR = BASE_DIR / "logs" / "mcp"

# MCP 服务地址 — 内嵌 FastMCP Server，供 Agent 内部调用
MCP_SERVER_URL = "http://127.0.0.1:1315"

# ChromaDB 持久化目录 — 向量数据存在磁盘上，重启不丢失
VECTOR_DB_DIR = CONFIG_DIR / "ChromaDB"

# 巡检报告目录 — 巡检Agent 生成 .md 报告 + RAG 启动注入源
INSPECTOR_REPORTS_DIR = CONFIG_DIR / "inspector_reports"

# 工作记录目录 — 调度Agent 生成诊断工作记录 + RAG 启动注入源
WORK_RECORDS_DIR = CONFIG_DIR / "work_records"

# 会话历史目录 — ConversationManager 持久化对话
CONVERSATIONS_DIR = CONFIG_DIR / "conversations"

# ============================================================
# Agent 标签
# ============================================================

AGENT_COUNT = 4

AGENT_NAMES = ["dispatcher", "query", "inspector", "diagnosis"]

AGENT_LOG_LEVELS = {
    "dispatcher": "INFO",
    "query": "INFO",
    "inspector": "INFO",
    "diagnosis": "INFO",
}

# ============================================================
# 诊断Agent 配置
# ============================================================

DIAGNOSIS_TIMEOUT = 60
DIAGNOSIS_MAX_ROUNDS = 3
