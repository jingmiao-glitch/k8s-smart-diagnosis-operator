"""
巡检Agent — 定时拉取集群指标和状态，生成结构化巡检报告，写入 RAG 知识库和本地文件

核心逻辑:
    1. asyncio 定时器每 30 分钟自动触发
    2. 通过 MCP client 调用内嵌 MCP Server（127.0.0.1:1315）的 7 个工具收集数据
    3. LLM 整合数据生成巡检报告
    4. 报告向量化后写入 RAG inspection_reports 知识库
    5. 报告同时保存为 Markdown 文件至 config/inspector_reports/

使用方式:
    from agents.inspector_agent import InspectorAgent

    agent = InspectorAgent()
    agent.start_schedule(interval_seconds=1800)          # 启动定时巡检
    report = await agent.generate_report()               # 手动生成即时报告
    agent.stop_schedule()                                # 停止定时巡检
"""

import asyncio
import json
from datetime import datetime

from logger import get_agent_logger
from utils.llm import get_model
from utils.rag import RAGKnowledgeBase
from utils.llm_utils import get_llm_text
from config import INSPECTOR_REPORTS_DIR
from tools.mcp_utils import call_mcp_tools
from utils.write_guard import critical_write

_log = get_agent_logger("inspector")


def _build_inspector_tools() -> list:
    """
    根据 INSPECTION_TOOL_NAMES + INSPECTION_TOOL_ARGS 构建工具调用列表

    新增巡检维度只需在 INSPECTION_TOOL_NAMES 中添加工具名，
    在 INSPECTION_TOOL_ARGS 中补充参数（如需要非默认空参数），重启服务即可生效。
    """
    return [
        {"tool": name, "args": INSPECTION_TOOL_ARGS.get(name, {})}
        for name in INSPECTION_TOOL_NAMES
    ]


INSPECTION_TOOL_NAMES = [
    "get_nodes",
    "get_pods",
    "get_events",
    "top_node",
    "top_pod",
    "node_cpu_usage",
    "node_memory_usage",
]

INSPECTION_TOOL_ARGS = {
    "node_cpu_usage": {"node_name": "ai-agent", "minutes": 360},
    "node_memory_usage": {"node_name": "ai-agent"},
}


REPORT_SYSTEM_PROMPT = """你是 Kubernetes 集群运维专家。请根据提供的集群数据生成巡检报告。

## 报告格式要求

严格按以下模板输出，不要添加或删除任何章节：

## 巡检报告 | {日期时间}

### 节点概览
（表格：各节点状态、CPU、内存）

### 异常节点
（列出 CPU 或内存超过 80% 的节点，每行一个事实陈述）
示例：node3 CPU使用率92%，超过80%阈值，状态异常

### 异常Pod
（列出非 Running 状态的 Pod，每行一个事实陈述，附带所在节点和异常原因）
示例：nginx-deploy-7d4f8b-x9k2m Pod状态：CrashLoopBackOff，重启3次，所在节点node3

### 集群事件
（列出 Warning 类型事件，每行一个事实陈述）

### 资源趋势
（从 Prometheus 时序数据中总结趋势。如果缺少 Prometheus 数据，写"缺乏 Prometheus 趋势数据，略过"）
示例：node3 CPU使用率过去6小时从60%持续上升至92%

### 评估结论
（整体评估 1-2 句话 + 是否需要人工介入）

## 格式约束
- 所有异常描述必须写成完整的句子，不要只列数字
- 异常节点和异常Pod章节每行一条事实，便于检索"""


async def generate_report(
    rag: RAGKnowledgeBase | None = None,
    context: str = "",
) -> str:
    """
    巡检Agent 核心函数: 收集集群数据 -> LLM生成报告 -> 写入RAG + 本地文件

    被调用方:
        - 定时任务触发 (写入RAG + 本地文件，不返回)
        - 调度Agent/诊断Agent 按需触发 (写入RAG + 本地文件 + 返回报告)

    流程:
        1. 通过 MCP client 调用 7 个工具获取集群数据
        2. 将所有数据汇总发给 LLM (inspector模型)
        3. LLM 按固定模板生成巡检报告
        4. 写入 RAG inspection_reports (如果 rag 不为空)
        5. 同时保存到 config/inspector_reports/{时间}.md
        6. 返回报告文本
    """
    now = datetime.now()

    _log.info("开始收集集群数据（7 个 MCP 工具）...")
    tool_calls = _build_inspector_tools()
    tool_results = await call_mcp_tools(tool_calls, agent_name="inspector")

    data_text = "\n\n".join(
        f"### {name}\n```\n{result}\n```"
        for name, result in tool_results.items()
    )

    context_section = f"\n\n诊断上下文（请重点关注以下方面）:\n{context}" if context else ""
    user_prompt = (
        f"当前时间: {now.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"以下是集群数据汇总：\n\n{data_text}"
        f"{context_section}\n\n"
        "请根据上述数据生成巡检报告。"
    )

    _log.info("共收集 7 项数据，准备调用 LLM 生成报告...")

    model = get_model("inspector")
    response = await model([
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    report = get_llm_text(response)

    if rag is not None:
        await rag.ingest_inspection(
            content=report,
            metadata={
                "date": now.strftime("%Y-%m-%d %H:%M"),
                "time": now.strftime("%H:%M"),
            },
        )
        _log.info("巡检报告已写入 RAG")

    INSPECTOR_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.md"
    filepath = INSPECTOR_REPORTS_DIR / filename
    with critical_write():
        filepath.write_text(report, encoding="utf-8")
    _log.info(f"巡检报告已保存: {filepath}")

    return report


class InspectorAgent:
    """
    巡检Agent 实例封装

    使用方式:
        agent = InspectorAgent()

        # 按需生成报告 (调度Agent/诊断Agent 调用)
        report = await agent.generate_report()

        # 启动定时巡检 (main.py 中调用)
        agent.start_schedule(interval_seconds=1800)
    """

    def __init__(self, rag: RAGKnowledgeBase | None = None):
        self._rag = rag or RAGKnowledgeBase()
        self._task: asyncio.Task | None = None

    async def generate_report(self, context: str = "") -> str:
        """生成巡检报告并写入 RAG"""
        return await generate_report(rag=self._rag, context=context)

    def start_schedule(self, interval_seconds: int = 1800):
        """启动定时巡检任务，默认30分钟一次"""

        async def _loop():
            _log.info(f"定时巡检已启动，间隔 {interval_seconds}s")
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await generate_report(rag=self._rag)
                except Exception as e:
                    _log.error(f"定时巡检出错: {e}")

        self._task = asyncio.ensure_future(_loop())

    def stop_schedule(self):
        """停止定时巡检"""
        if self._task:
            self._task.cancel()
