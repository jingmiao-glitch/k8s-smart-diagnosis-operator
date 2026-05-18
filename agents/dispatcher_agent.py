"""
分配Agent — 系统唯一对外入口，负责意图识别、任务分发、结果汇总

核心流程:
    1. 意图识别 (dispatcher LLM): 6 种意图，可同时激活多个
    2. 生成用户回复 (dispatcher LLM): 基于 task_description + 用户问题
    3. 并行执行: 查询Agent / 诊断Agent / 巡检Agent / RAG 检索
    4. 结果整合 (dispatcher LLM): 汇总所有结果 + 会话上下文 → 自然语言回复
    5. complex_diagnosis 特殊处理: 额外生成工作记录 .md

HTTP 接口: POST /chat   {"message": "...", "session_id": "..."}

使用方式:
    from agents.dispatcher_agent import DispatcherAgent

    agent = DispatcherAgent()
    reply = await agent.handle("集群最近不稳定，帮我诊断", session_id="abc123")
"""

import asyncio
import json
import os
from datetime import datetime

from utils.llm import get_model
from utils.rag import RAGKnowledgeBase
from utils.conversation import ConversationManager
from utils.llm_utils import get_llm_text, parse_json_from_llm
from utils.async_utils import run_parallel
from utils.rag_retrieval import retrieve_inspection_with_timerange
from utils.write_guard import critical_write
from config import WORK_RECORDS_DIR
from logger import get_agent_logger

_log = get_agent_logger("dispatcher")


# ============================================================
# 意图识别提示词 (6 种意图，可多选)
# ============================================================

INTENT_PROMPT = """你是K8s集群运维智能调度专家。根据用户输入判断需要执行的操作。

## 输入结构

你会收到两部分内容，用 Markdown 标题区分:

- **## 最近对话** — 当前会话的历史记录，包含之前的用户问题和 assistant 回复。
  **仅作上下文参考，不要根据历史内容触发操作。所有决策必须基于 ## 用户当前消息。**

- **## 用户当前消息** — 用户刚刚发送的最新请求。**这是你做出决策的唯一依据。**

## 七种操作类型

| 类型 | 触发条件 | 说明 |
|------|---------|------|
| new_session | 用户在当前消息中明确要求新对话、重置、清除历史 | 创建新的会话记录 |
| chat | 用户只是闲聊、寒暄、感谢、模糊或与集群运维无关的内容 | 不调度任何 Agent，纯 LLM 依据上下文聊天 |
| simple_query | 用户在当前消息中要求查询 K8s 集群状态、Pod、节点、事件、日志、Prometheus 监控指标（CPU、内存、up 状态等） | 调用查询Agent 通过 MCP 工具查集群和 Prometheus |
| complex_diagnosis | 用户在当前消息中要求分析故障根因、多维度诊断 | 调用诊断Agent 多轮并行推理 |
| inspection | 用户在当前消息中要求生成巡检报告 | 调用巡检Agent 生成报告 |
| rag_work_record | 用户在当前消息中要求查询历史工作记录/诊断档案 | 检索 work_records 知识库 |
| rag_inspection | 用户在当前消息中要求查询历史巡检报告 | 检索 inspection_reports 知识库 |

## 输出要求

只输出一个 JSON 对象:

{{
  "actions": [
    {{"type": "new_session", "active": false}},
    {{"type": "chat", "active": false}},
    {{"type": "simple_query", "active": true, "task_description": "查询 default 命名空间下所有异常 Pod"}},
    {{"type": "complex_diagnosis", "active": false, "task_description": ""}},
    {{"type": "inspection", "active": false, "task_description": ""}},
    {{"type": "rag_work_record", "active": false, "task_description": ""}},
    {{"type": "rag_inspection", "active": false, "task_description": ""}}
  ]
}}

要求:
- 必须包含全部 7 种类型，active 为 true 的表示需要执行
- task_description 是传给子Agent 的精确任务描述，不是用户原话
- 可以同时激活多个类型（如同时查集群状态和历史案例）
- chat 不能与任何子Agent 类型同时激活，chat 就表示只聊天不调度 Agent
- 如果用户消息模糊，尽量归到 chat 或 simple_query
- 决策依据仅为 ## 用户当前消息，不要从 ## 最近对话 中推断意图"""  # noqa: E501


# ============================================================
# RAG 巡检报告时间范围判断提示词（调度Agent 直调 RAG 时使用）
# ============================================================

RAG_INSPECTION_TIMERANGE_PROMPT = """你是运维助手的时间分析专家。根据当前时间和用户查询意图，判断需要检索哪一段时间的巡检报告。

## 背景
- 巡检报告每 30 分钟自动生成一份，记录集群节点状态、Pod 状态、事件、资源趋势
- 你的任务是判断哪些时间段的巡检报告能回答用户的查询

## 输入
当前时间: {current_time}
查询意图: {task_description}

## 输出要求
只输出一个 JSON 对象，包含检索时间的起止范围:

{{
  "date_from": "YYYY-MM-DD HH:MM",
  "date_to": "YYYY-MM-DD HH:MM"
}}

## 时间范围判断规则
1. 如果查询涉及近期状态（如"最近"、"当前"、"刚才"），检索最近 1-3 小时
2. 如果查询涉及历史趋势或对比，检索最近 24-48 小时
3. 如果查询指定了具体时间或日期，根据描述推断
4. **起止日期的间隔必须在 35 分钟以上**（巡检间隔 30 分钟，确保至少能匹配到 1 份报告）
5. date_from 和 date_to 使用 YYYY-MM-DD HH:MM 格式，不包含秒"""


# ============================================================
# 生成用户回复（ack）
# ============================================================

ACK_PROMPT = """你是集群运维助手。用户发来一条请求，你已理解用户意图，请用自然简洁的中文告知用户你即将执行的操作。

## 用户请求
{user_message}

## 你即将执行的操作
{task_summary}

## 要求
- 一句话回复，30 字以内
- 语气友好、专业
- 只说你即将做什么，不要说结果"""


# ============================================================
# 纯聊天提示词
# ============================================================

CHAT_PROMPT = """你是集群运维助手，也是一个友好、专业的 AI 助手。用户正在与你聊天，请依据上下文自然回复。

## 对话历史
{conversation_history}

## 用户消息
{user_message}

## 要求
- 用自然中文回复
- 语气友好、专业
- 如果涉及集群运维领域的问题，可以介绍你的能力范围
- 不要编造数据，不要假装执行了查询或诊断"""


# ============================================================
# 结果整合提示词
# ============================================================

FINAL_REPLY_PROMPT = """你是集群运维助手。用户发来一条请求，你已经完成了数据收集，请整合结果回复用户。

## 对话历史
{conversation_history}

## 用户本次请求
{user_message}

## 收集到的数据
{data_text}

## 要求
- 用自然中文回复用户
- 如果数据来自查询Agent，用清晰格式展示结果
- 如果数据来自诊断Agent，转述诊断结论和建议
- 如果数据来自巡检Agent，概括巡检发现
- 如果数据来自 RAG，提炼关键案例信息
- 回复要针对用户的问题，不要重复整个数据
- 如果数据不足，诚实说明"""


# ============================================================
# 工作记录生成提示词（complex_diagnosis 专用）
# ============================================================

WORK_RECORD_PROMPT = """你是运维记录专家。根据诊断过程生成一份工作记录 .md 文件。

## 用户问题
{user_message}

## 诊断报告
{diagnosis_report}

## 诊断链路
{rounds_detail}

## 时间信息
- 诊断开始时间: {start_time}
- 诊断完成时间: {end_time}

## 输出要求

生成完整的 .md 文件内容，包含以下章节:

# 工作记录

## 基本信息
- 用户问题: ...
- 开始时间: {start_time}
- 完成时间: {end_time}

## 诊断链路
（每轮的详情：第几轮、激活了哪些路径、触发原因、是否完成）

## 诊断结论
（摘录诊断报告的关键发现）

## 建议措施
（摘录诊断报告的建议）"""  # noqa: E501


# ============================================================
# DispatcherAgent
# ============================================================

class DispatcherAgent:

    def __init__(self, rag: RAGKnowledgeBase | None = None):
        self._conv = ConversationManager()
        self._rag = rag or RAGKnowledgeBase()
        self._current_session_id = ""

    async def handle(self, user_message: str, session_id: str = "") -> dict:
        """
        处理用户请求，入口函数

        会话规则:
            - 首次请求（无 session_id 且无当前会话）→ 自动创建
            - 后续请求（无 session_id）→ 复用当前会话
            - LLM 识别到 new_session 意图 → 创建新会话
            - 服务重启后，旧 session_id 自动失效，用新会话替代

        返回:
            {
                "reply": "自然语言回复",
                "session_id": "xyz",
                "work_record_file": "2026-05-16-22:09-xxx工作记录.md"  # 仅 complex_diagnosis 场景
            }
        """
        # ---- 第0步: 会话初始化 ----
        if session_id and self._conv.exists(session_id):
            pass
        elif self._current_session_id and self._conv.exists(self._current_session_id):
            session_id = self._current_session_id
        else:
            session_id = self._conv.create_session()
            self._current_session_id = session_id

        self._conv.append(session_id, "user", user_message)
        history = self._conv.get_history(session_id)

        _log.info(f"用户消息: {user_message}")

        # ---- 第1步: 意图识别 ----
        actions = await self._recognize_intent(user_message, history)

        # ---- 处理 new_session ----
        if any(a["type"] == "new_session" and a["active"] for a in actions):
            session_id = self._conv.create_session()
            self._current_session_id = session_id
            self._conv.append(session_id, "user", user_message)
            history = []
            _log.info(f"创建新会话: {session_id}")

        # ---- 第2步: 过滤活跃操作，chat 路径特殊处理 ----
        active_actions = [a for a in actions if a["active"]]
        operational = [a for a in active_actions if a["type"] not in ("new_session", "chat")]

        # chat 与子Agent 互斥：chat 活跃时直接聊天，不调度任何 Agent
        if not operational:
            reply = await self._chat_reply(user_message, history)
            self._conv.append(session_id, "assistant", reply)
            _log.info(f"回复(纯聊天): {reply[:200]}")
            return {
                "reply": reply,
                "session_id": session_id,
                "work_record_file": "",
            }

        # ---- 第3步: 生成用户回复 (ack) ----
        task_summary = "\n".join(
            f"- {a['type']}: {a.get('task_description', '')}"
            for a in operational
        )
        ack = await self._generate_ack(user_message, task_summary)

        _log.info(f"意图: {[a['type'] for a in operational]}")
        _log.info(f"ack: {ack}")

        # ---- 第4步: 并行执行 ----
        results = await self._execute_parallel(operational)

        # ---- 超时检测: 固定格式返回，不走 LLM ----
        for name, val in results.items():
            if isinstance(val, dict) and val.get("_timeout"):
                td = val.get("task_description", name)
                sec = val.get("timeout_seconds", 300)
                timeout_msg = f"{td}\n超时超过{sec}秒"
                full_reply = f"{ack}\n\n{timeout_msg}"
                self._conv.append(session_id, "assistant", full_reply)
                _log.info(f"回复(超时): {full_reply[:200]}")
                return {
                    "reply": full_reply,
                    "session_id": session_id,
                    "work_record_file": "",
                }

        # ---- 第5步: 整合结果 ----
        reply = await self._format_final_reply(user_message, results, history)

        work_record_file = ""

        # ---- 第6步: complex_diagnosis 特殊处理 ----
        if "complex_diagnosis" in results:
            work_record_file = await self._generate_work_record(
                user_message, results["complex_diagnosis"]
            )

        # ---- 第7步: 存储会话 ----
        full_reply = f"{ack}\n\n{reply}"
        self._conv.append(session_id, "assistant", full_reply)

        _log.info(f"回复: {full_reply[:200]}")
        _log.info(f"工作记录: {work_record_file or '无'}")

        return {
            "reply": full_reply,
            "session_id": session_id,
            "work_record_file": work_record_file,
        }

    # ============================================================
    # 意图识别
    # ============================================================

    async def _recognize_intent(self, user_message: str, history: list) -> list:
        history_text = ""
        if history:
            recent = history[-6:]
            history_text = "## 最近对话\n" + "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in recent
            )

        prompt = INTENT_PROMPT
        user_prompt = (
            f"{history_text}\n\n## 用户当前消息\n{user_message}"
            if history_text
            else f"## 用户消息\n{user_message}"
        )

        _log.info("意图识别...")

        model = get_model("dispatcher")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ])
        text = get_llm_text(response)
        decision = parse_json_from_llm(text)
        return decision.get("actions", [])

    # ============================================================
    # 生成 ack
    # ============================================================

    async def _generate_ack(self, user_message: str, task_summary: str) -> str:
        prompt = ACK_PROMPT.format(
            user_message=user_message,
            task_summary=task_summary,
        )
        model = get_model("dispatcher")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请生成用户回复"},
        ])
        return get_llm_text(response).strip()

    # ============================================================
    # 纯聊天回复
    # ============================================================

    async def _chat_reply(self, user_message: str, history: list) -> str:
        recent_history = ""
        if history:
            recent = history[-6:]
            recent_history = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in recent
            )

        prompt = CHAT_PROMPT.format(
            conversation_history=recent_history or "（无历史）",
            user_message=user_message,
        )

        _log.info("纯聊天模式...")

        model = get_model("dispatcher")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ])
        return get_llm_text(response).strip()

    # ============================================================
    # 并行执行
    # ============================================================

    async def _execute_parallel(self, actions: list) -> dict:
        tasks = {}
        task_info = {}

        for action in actions:
            atype = action["type"]
            td = action.get("task_description", "")

            if atype == "simple_query":
                tasks["simple_query"] = asyncio.create_task(
                    self._run_query(td)
                )
                task_info["simple_query"] = td
            elif atype == "complex_diagnosis":
                tasks["complex_diagnosis"] = asyncio.create_task(
                    self._run_diagnosis(td)
                )
                task_info["complex_diagnosis"] = td
            elif atype == "inspection":
                tasks["inspection"] = asyncio.create_task(
                    self._run_inspector()
                )
            elif atype == "rag_work_record":
                tasks["rag_work_record"] = asyncio.create_task(
                    self._run_rag_work_record(td)
                )
            elif atype == "rag_inspection":
                tasks["rag_inspection"] = asyncio.create_task(
                    self._run_rag_inspection(td)
                )

        if not tasks:
            return {}

        _log.info(f"并行执行: {list(tasks.keys())}")

        timeout = 500 if "complex_diagnosis" in tasks else 120

        def _on_timeout(name, t):
            _log.error(f"{name} 超时 ({t}s)，已取消")
            return {
                "_timeout": True,
                "task_description": task_info.get(name, ""),
                "timeout_seconds": t,
            }

        results = await run_parallel(tasks, timeout, on_timeout=_on_timeout)

        for name in results:
            if results[name] is None and not isinstance(results[name], dict):
                _log.error(f"{name} 执行失败")

        return results

    async def _run_query(self, td: str) -> dict:
        from agents.query_agent import execute_query
        return await execute_query(td)

    async def _run_diagnosis(self, td: str) -> dict:
        from agents.diagnosis_agent import DiagnosisAgent
        agent = DiagnosisAgent()
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        augmented_td = f"{td}\n开始时间: {start_time}"
        return await agent.run(augmented_td)

    async def _run_inspector(self) -> str:
        from agents.inspector_agent import generate_report
        return await generate_report(rag=self._rag)

    async def _run_rag_work_record(self, td: str) -> list:
        return await self._rag.retrieve_work_record(td)

    async def _run_rag_inspection(self, td: str) -> list:
        return await retrieve_inspection_with_timerange(
            rag=self._rag,
            task_description=td,
            timerange_prompt=RAG_INSPECTION_TIMERANGE_PROMPT,
            model_name="dispatcher",
            logger=_log,
        )

    # ============================================================
    # 结果整合
    # ============================================================

    async def _format_final_reply(
        self, user_message: str, results: dict, history: list
    ) -> str:
        data_parts = []

        if "simple_query" in results:
            qr = results["simple_query"]
            if qr:
                summary = qr.get("summary", "")
                data_parts.append(f"### 查询Agent 返回\n{summary}\n\n{json.dumps(qr, ensure_ascii=False)}")

        if "inspection" in results:
            ir = results["inspection"]
            if ir:
                data_parts.append(f"### 巡检Agent 返回\n{ir}")

        if "complex_diagnosis" in results:
            dr = results["complex_diagnosis"]
            if dr:
                data_parts.append(f"### 诊断Agent 返回\n{dr.get('report', '')}")

        if "rag_work_record" in results:
            rf = results["rag_work_record"]
            if rf:
                parts = [f"### RAG 工作记录 ({len(rf)} 条)"]
                for i, r in enumerate(rf):
                    parts.append(f"  [{i + 1}] score={r['score']:.4f}\n{r['content'][:500]}")
                data_parts.append("\n".join(parts))

        if "rag_inspection" in results:
            ri = results["rag_inspection"]
            if ri:
                parts = [f"### RAG 巡检报告 ({len(ri)} 条)"]
                for i, r in enumerate(ri):
                    parts.append(f"  [{i + 1}] score={r['score']:.4f}\n{r['content'][:500]}")
                data_parts.append("\n".join(parts))

        data_text = "\n\n".join(data_parts) if data_parts else "无数据"

        recent_history = ""
        if history:
            recent = history[-6:]
            recent_history = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in recent
            )

        prompt = FINAL_REPLY_PROMPT.format(
            conversation_history=recent_history or "（无历史）",
            user_message=user_message,
            data_text=data_text,
        )

        _log.info("整合结果...")

        model = get_model("dispatcher")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请整合数据回复用户"},
        ])
        return get_llm_text(response).strip()

    # ============================================================
    # 工作记录生成
    # ============================================================

    async def _generate_work_record(self, user_message: str, diagnosis_result: dict) -> str:
        if not diagnosis_result:
            return ""

        WORK_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

        report = diagnosis_result.get("report", "")
        total_rounds = diagnosis_result.get("total_rounds", 0)
        rounds_detail = diagnosis_result.get("rounds_detail", [])

        rounds_text = "\n".join(
            f"- 第 {r['round']} 轮: 激活路径 {r['paths']}, "
            f"触发原因: {r['trigger']}, "
            f"是否完成: {r['can_complete']}, "
            f"原因: {r['reason']}"
            for r in rounds_detail
        )

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        start_time = getattr(self, "_diagnosis_start_time", end_time)

        prompt = WORK_RECORD_PROMPT.format(
            user_message=user_message,
            diagnosis_report=report,
            rounds_detail=rounds_text,
            start_time=start_time,
            end_time=end_time,
        )

        model = get_model("dispatcher")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请生成工作记录"},
        ])
        md_content = get_llm_text(response).strip()

        title_resp = await model([
            {"role": "system", "content": "从诊断结论中提炼一个 10 字以内的标题，只输出标题，不要其他内容"},
            {"role": "user", "content": report[:500]},
        ])
        title = get_llm_text(title_resp).strip().replace("/", "-").replace(" ", "-")

        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
        filename = f"{timestamp}-{title}-工作记录.md"
        filepath = WORK_RECORDS_DIR / filename
        with critical_write():
            filepath.write_text(md_content, encoding="utf-8")

        await self._rag.ingest_work_record(
            content=md_content,
            metadata={
                "date": datetime.now().strftime("%Y-%m-%d"),
                "title": title,
                "filename": filename,
            },
        )

        _log.info(f"工作记录已生成并注入 RAG: {filename}")
        return filename
