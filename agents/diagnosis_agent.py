"""
诊断Agent — 接收调度Agent分派的任务，通过 3 轮循环完成数据收集与推理，生成最终报告

核心逻辑 (每轮):
    1. 策略选择 (diagnosis_selector 模型): 根据"不能完成任务的原因 + 还需要的数据"
       决定本轮激活哪些并行路径 (query / inspector / RAG)
    2. 并行采集: asyncio.wait 同时执行选中的路径，超时 40 秒
    3. 推理判断 (diagnosis_reasoning 模型): 整合所有轮次数据，判断能否完成任务
       ├── can_complete=true  → 生成最终报告 → return
       └── can_complete=false → 输出"不能完成的原因 + 还需要的数据" → 下一轮

最多 3 轮循环，超限直接生成报告。

返回值:
    {
        "report": "最终报告字符串",
        "total_rounds": 2,
        "rounds_detail": [
            {
                "round": 1,
                "paths": ["query", "inspector", "rag"],
                "trigger": "首次诊断",
                "can_complete": false,
                "reason": "缺少实时资源数据",
                "need_more": "需要 node3 Prometheus 时序数据"
            },
            ...
        ]
    }

使用方式:
    from agents.diagnosis_agent import DiagnosisAgent

    agent = DiagnosisAgent()
    result = await agent.run(task_description="集群最近不稳定，帮我诊断原因")
"""

import asyncio
import json
from datetime import datetime

from logger import get_agent_logger
from utils.llm import get_model
from utils.rag import RAGKnowledgeBase
from utils.task_context import TaskContext
from utils.llm_utils import get_llm_text, parse_json_from_llm
from utils.async_utils import run_parallel
from utils.rag_retrieval import retrieve_inspection_with_timerange
from config import DIAGNOSIS_TIMEOUT, DIAGNOSIS_MAX_ROUNDS

_log = get_agent_logger("diagnosis")

AVAILABLE_PATHS = [
    ("query", "调用查询Agent — 查询K8s集群实时状态"),
    ("inspector", "调用巡检Agent — 立即生成一份巡检报告"),
    ("rag", "检索RAG知识库 — 查找历史巡检报告和历史工作记录"),
]


# ============================================================
# 策略选择提示词
# ============================================================

SELECTOR_PROMPT = """你是运维诊断的策略协调专家。根据以下信息，决定本轮需要激活哪些数据收集路径。

## 可用路径
{path_list}

## 当前状态
调度任务: {task_description}

开始时间: {start_time}
当前时间: {current_time}

{context}

## 输出要求
只输出一个 JSON 对象，包含 paths（要激活的路径列表）和 reason（本轮选择这些路径的理由）。

{{
  "paths": ["query", "inspector", "rag"],
  "reason": "需要采集集群状态、巡检报告和历史数据"
}}

要求:
- paths 数组中的值只能是 query / inspector / rag
- 不需要的路径不要包含
- reason 用中文简要说明"""  # noqa: E501


def _build_previous_data_summary(prev_rounds: list) -> str:
    parts = []
    for rd in prev_rounds:
        parts.append(f"第 {rd.round_num} 轮:")
        parts.append(f"  激活路径: {rd.paths_requested}")
        if rd.query_result:
            parts.append(f"  查询结果: {json.dumps(rd.query_result, ensure_ascii=False)[:500]}")
        if rd.inspector_result:
            parts.append(f"  巡检报告(前200字): {rd.inspector_result[:200]}")
        if rd.rag_results:
            ir = rd.rag_results.get("inspection", [])
            wr = rd.rag_results.get("work_record", [])
            if ir:
                parts.append(f"  历史巡检报告: {len(ir)} 条")
            if wr:
                parts.append(f"  工作记录: {len(wr)} 条")
        parts.append("")
    return "\n".join(parts)


def _build_rag_context(prev_rounds: list) -> str:
    parts = []
    for rd in prev_rounds:
        if rd.rag_results:
            ir = rd.rag_results.get("inspection", [])
            wr = rd.rag_results.get("work_record", [])
            if ir:
                parts.append(f"第{rd.round_num}轮 历史巡检报告 ({len(ir)} 条):")
                for i, r in enumerate(ir[:3]):
                    parts.append(f"  [{i + 1}] score={r['score']:.4f}\n  {r['content'][:200]}")
            if wr:
                parts.append(f"第{rd.round_num}轮 工作记录 ({len(wr)} 条):")
                for i, r in enumerate(wr[:3]):
                    parts.append(f"  [{i + 1}] score={r['score']:.4f}\n  {r['content'][:200]}")
    return "\n".join(parts) if parts else "（无）"


def _build_selector_context(
    prev_reason: str,
    prev_need_more: str,
    previous_data: str = "",
) -> str:
    lines = ["上一轮诊断未能完成任务。"]
    if previous_data:
        lines.append(f"\n## 上一轮已收集到的数据\n{previous_data}")
    lines.append(f"\n不能完成的原因: {prev_reason}")
    lines.append(f"还需要的数据: {prev_need_more}")
    return "\n".join(lines)


# ============================================================
# 推理判断提示词
# ============================================================

REASONING_PROMPT = """你是运维诊断专家。根据调度任务和收集到的数据，判断是否可以完成任务，如果可以则生成最终诊断报告。

## 调度任务
{task_description}

开始时间: {start_time}
当前时间: {current_time}

## 收集到的数据
{data_text}

## 输出要求

只输出一个 JSON 对象:

{{
  "can_complete": false,
  "reason": "还缺少某些关键数据",
  "need_more": "需要额外收集的数据说明",
  "report": "如果 can_complete 为 true，这里写完整的最终诊断报告"
}}

要求:
- can_complete: 是否已有足够数据完成任务
- reason: 若无法完成任务，说明原因
- need_more: 若无法完成任务，说明还需要什么数据
- report: 若可以完成，给出完整的最终报告"""

FORCE_REPORT_PROMPT = """你是运维诊断专家。已进行 {total_rounds} 轮数据收集，现在必须基于已有数据生成最终报告。

## 调度任务
{task_description}

开始时间: {start_time}
当前时间: {current_time}

## 收集到的数据
{data_text}

## 输出要求

只输出一个 JSON 对象:

{{
  "can_complete": true,
  "reason": "",
  "need_more": "",
  "report": "基于已有数据生成的最终报告"
}}

要求:
- report 必须是完整的最终报告（按任务要求生成对应格式的报告）
- 如果数据不足，在报告中诚实说明信息缺失的部分
- 证据来自收集到的数据，不要编造"""  # noqa: E501


# ============================================================
# RAG 巡检报告时间段判断提示词
# ============================================================

INSPECTION_TIMERANGE_PROMPT = """你是运维诊断的时间分析专家。根据当前时间和任务描述，判断需要检索哪一段时间的巡检报告。

## 背景
- 巡检报告每 30 分钟自动生成一份，记录集群节点状态、Pod 状态、事件、资源趋势
- 你的任务是判断哪些时间段的巡检报告能帮助诊断当前问题

## 输入
当前时间: {current_time}
任务描述: {task_description}
{failure_context}

## 输出要求
只输出一个 JSON 对象，包含检索时间的起止范围:

{{
  "date_from": "YYYY-MM-DD HH:MM",
  "date_to": "YYYY-MM-DD HH:MM"
}}

## 时间范围判断规则
1. 如果任务涉及突发故障（Pod 重启、节点异常、资源飙升等），检索最近 1-3 小时的报告
2. 如果任务涉及长期趋势（资源消耗趋势、历史对比），检索最近 24-48 小时
3. 如果任务涉及特定时间段（如"昨天的故障"），根据描述推断
4. **起止日期的间隔必须在 35 分钟以上**（巡检间隔 30 分钟，确保至少能匹配到 1 份报告）
5. date_from 和 date_to 使用 YYYY-MM-DD HH:MM 格式，不包含秒
6. 如果额外上下文中提示数据不足，适当扩大时间范围；如果已收集到足够数据，缩小范围聚焦"""


# ============================================================
# 查询Agent 指令生成提示词
# ============================================================

QUERY_INSTRUCTION_PROMPT = """你是运维诊断的查询指令生成专家。基于以下信息，生成一条完整的K8s集群查询指令。

## 原始任务
{task_description}

## 前面已收集的数据
{previous_data}

## 上一轮诊断失败
原因: {prev_reason}
还需要: {prev_need_more}

## 要求
生成一条完整的K8s查询指令，用于查询Agent执行。
指令必须明确包含：
1. 具体的命名空间名称（不要说"相关命名空间"或"对应命名空间"）
2. 具体的Pod名称或资源类型（如果已知）
3. 明确要查询什么：日志、事件、状态、资源使用量、配置等
4. 所有必要参数，让查询Agent无需推测就能直接调用MCP工具

只输出查询指令文本，不要JSON或其他格式。"""


# ============================================================
# RAG 检索优化提示词
# ============================================================

RAG_OPTIMIZE_PROMPT = """你是运维知识检索优化专家。基于以下信息，生成一条用于检索历史运维知识库的查询文本。

## 原始任务
{task_description}

## 之前RAG检索到的内容
{previous_rag_results}

## 上一轮诊断失败
原因: {prev_reason}
还需要: {prev_need_more}

## 要求
生成一条优化后的检索查询文本，用于在向量知识库中进行语义匹配。
查询文本应包含关键术语、症状描述、时间段信息等，以提高匹配精度。
如果之前检索结果不够精确，调整关键词和描述方式。

只输出查询文本，不要JSON或其他格式。"""


# ============================================================
# 数据累积
# ============================================================

class RoundData:
    """单轮数据"""

    def __init__(self, round_num: int):
        self.round_num = round_num
        self.what_triggered = ""
        self.paths_requested: list = []
        self.query_result: dict | None = None
        self.inspector_result: str | None = None
        self.rag_results: dict | None = None
        self.can_complete: bool = False
        self.reason: str = ""
        self.need_more: str = ""

    def build_text_block(self) -> str:
        lines = [f"--- 第 {self.round_num} 轮 ---"]
        if self.what_triggered:
            lines.append(f"触发原因: {self.what_triggered}")

        if self.query_result:
            lines.append("查询Agent 返回:")
            lines.append(json.dumps(self.query_result, ensure_ascii=False, indent=2))

        if self.inspector_result:
            lines.append("巡检Agent 返回 (巡检报告):")
            lines.append(self.inspector_result)

        if self.rag_results:
            lines.append("RAG 知识库返回:")
            ir = self.rag_results.get("inspection", [])
            wr = self.rag_results.get("work_record", [])
            if ir:
                lines.append(f"  历史巡检报告 ({len(ir)} 条):")
                for i, r in enumerate(ir):
                    lines.append(f"    [{i + 1}] score={r['score']:.4f}\n{r['content'][:300]}")
            if wr:
                lines.append(f"  工作记录 ({len(wr)} 条):")
                for i, r in enumerate(wr):
                    lines.append(f"    [{i + 1}] score={r['score']:.4f}\n{r['content'][:300]}")

        return "\n".join(lines)


# ============================================================
# 核心诊断循环
# ============================================================

class DiagnosisAgent:

    def __init__(self, rag: RAGKnowledgeBase | None = None):
        self._rag = rag or RAGKnowledgeBase()

    async def run(self, task_description: str) -> dict:
        """
        执行诊断流程

        参数:
            task_description: 调度Agent 分派的任务描述（末尾附"开始时间: YYYY-MM-DD HH:MM"）

        返回:
            {
                "report": "最终报告字符串",
                "total_rounds": N,
                "rounds_detail": [{...}, ...]
            }
        """
        self._start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        clean_td = task_description

        if "开始时间:" in task_description:
            lines = task_description.split("\n")
            for line in lines:
                if line.startswith("开始时间:"):
                    self._start_time = line.replace("开始时间:", "").strip()
            clean_td = "\n".join(
                line for line in lines if not line.startswith("开始时间:")
            ).strip()

        rounds: list[RoundData] = []
        prev_reason = ""
        prev_need_more = ""

        for round_num in range(1, DIAGNOSIS_MAX_ROUNDS + 1):
            rd = RoundData(round_num)
            rounds.append(rd)

            if round_num == 1:
                rd.what_triggered = "首次诊断"
                context = ""
            else:
                rd.what_triggered = (
                    f"不能完成的原因: {prev_reason}\n还需要的数据: {prev_need_more}"
                )
                prev_data = _build_previous_data_summary(rounds[:-1])
                context = _build_selector_context(
                    prev_reason, prev_need_more, prev_data
                )

            rd.paths_requested = await self._select_paths(
                clean_td, context
            )

            path_instructions = {}
            if round_num >= 2:
                prev_rounds = rounds[:-1]
                if "query" in rd.paths_requested:
                    path_instructions["query"] = await self._generate_query_instruction(
                        clean_td, prev_rounds, prev_reason, prev_need_more
                    )
                if "rag" in rd.paths_requested:
                    path_instructions["rag"] = await self._generate_rag_query(
                        clean_td, prev_rounds, prev_reason, prev_need_more
                    )
                    path_instructions["rag_failure_context"] = (
                        f"上一轮失败原因: {prev_reason}\n还需要: {prev_need_more}"
                    )
                if "inspector" in rd.paths_requested:
                    insp_ctx = _build_previous_data_summary(prev_rounds)
                    if prev_reason:
                        insp_ctx += f"\n上一轮失败原因: {prev_reason}"
                    path_instructions["inspector"] = insp_ctx

            round_data = await self._collect_parallel(
                clean_td, rd.paths_requested, path_instructions
            )
            rd.query_result = round_data.get("query")
            rd.inspector_result = round_data.get("inspector")
            rd.rag_results = round_data.get("rag")

            is_last_round = (round_num == DIAGNOSIS_MAX_ROUNDS)
            decision = await self._reason(rounds, clean_td, is_last_round)

            rd.can_complete = decision.get("can_complete", False)
            rd.reason = decision.get("reason", "")
            rd.need_more = decision.get("need_more", "")

            prev_reason = rd.reason
            prev_need_more = rd.need_more

            if rd.can_complete or is_last_round:
                return {
                    "report": decision.get("report", "无法生成报告"),
                    "total_rounds": len(rounds),
                    "rounds_detail": [
                        {
                            "round": r.round_num,
                            "paths": r.paths_requested,
                            "trigger": r.what_triggered,
                            "can_complete": r.can_complete,
                            "reason": r.reason,
                            "need_more": r.need_more,
                        }
                        for r in rounds
                    ],
                }

        return {
            "report": "诊断流程异常退出",
            "total_rounds": 0,
            "rounds_detail": [],
        }

    # ============================================================
    # 策略选择
    # ============================================================

    async def _select_paths(self, task_description: str, context: str) -> list:
        path_desc = "\n".join(f"- **{name}**: {desc}" for name, desc in AVAILABLE_PATHS)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt = SELECTOR_PROMPT.format(
            path_list=path_desc,
            task_description=task_description,
            context=context,
            start_time=self._start_time,
            current_time=current_time,
        )

        _log.info("策略选择 (selector LLM)...")

        model = get_model("diagnosis_selector")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请决定本轮需要激活哪些路径"},
        ])
        text = get_llm_text(response)
        decision = parse_json_from_llm(text)

        paths = decision.get("paths", [])
        reason = decision.get("reason", "")
        _log.info(f"选择路径: {paths} (理由: {reason})")
        return paths

    # ============================================================
    # 并行采集
    # ============================================================

    async def _collect_parallel(
        self, task_description: str, paths: list,
        path_instructions: dict | None = None,
    ) -> dict:
        instr = path_instructions or {}
        tasks = {}

        if "query" in paths:
            qd = instr.get("query", task_description)
            tasks["query"] = asyncio.create_task(self._run_query(qd))

        if "inspector" in paths:
            insp_ctx = instr.get("inspector", "")
            tasks["inspector"] = asyncio.create_task(self._run_inspector(insp_ctx))

        if "rag" in paths:
            rag_q = instr.get("rag", task_description)
            rag_fc = instr.get("rag_failure_context", "")
            tasks["rag"] = asyncio.create_task(self._run_rag(rag_q, rag_fc))

        if not tasks:
            return {}

        _log.info(f"并行采集开始 (第{round_num}轮, 路径: {list(tasks.keys())}, timeout={DIAGNOSIS_TIMEOUT}s)...")

        def _on_timeout(name, t):
            _log.error(f"路径 {name} 超时 ({t}s)，已取消")

        results = await run_parallel(tasks, DIAGNOSIS_TIMEOUT, on_timeout=_on_timeout)

        for name in results:
            if results[name] is None and not isinstance(results[name], dict):
                _log.error(f"路径 {name} 执行失败")

        _log.info(f"并行采集完成，成功: {list(results.keys())}")
        return results

    async def _run_query(self, task_description: str) -> dict:
        _log.info("调用查询Agent...")
        from agents.query_agent import execute_query
        return await execute_query(task_description)

    async def _run_inspector(self, context: str = "") -> str:
        _log.info("调用巡检Agent 生成即时报告...")
        from agents.inspector_agent import generate_report
        return await generate_report(rag=None, context=context)

    async def _run_rag(
        self, task_description: str, failure_context: str = "",
    ) -> dict:
        """
        检索 RAG 知识库，同时查两个集合并合并返回

        巡检报告: task_description 原样做语义匹配 + LLM 判断时间段做 metadata.date 精确过滤
        工作记录: task_description 原样做语义匹配

        返回: {"inspection": [...], "work_record": [...]}
        """
        _log.info("检索 RAG 知识库...")

        timerange_prompt = INSPECTION_TIMERANGE_PROMPT.replace(
            "{failure_context}", ""
        )
        if failure_context:
            timerange_prompt = INSPECTION_TIMERANGE_PROMPT.replace(
                "{failure_context}",
                f"\n## 额外上下文\n{failure_context}",
            )

        work_task = asyncio.create_task(
            self._rag.retrieve_work_record(task_description)
        )
        inspection_task = asyncio.create_task(
            retrieve_inspection_with_timerange(
                rag=self._rag,
                task_description=task_description,
                timerange_prompt=timerange_prompt,
                model_name="diagnosis_selector",
                logger=_log,
            )
        )

        work_record = await work_task
        inspection = await inspection_task

        _log.info(f"RAG 合并结果: inspection={len(inspection)}条 work_record={len(work_record)}条")
        return {"inspection": inspection, "work_record": work_record}

    # ============================================================
    # 指令生成（第2/3轮用）
    # ============================================================

    async def _generate_query_instruction(
        self, clean_td: str, prev_rounds: list,
        prev_reason: str, prev_need_more: str,
    ) -> str:
        previous_data = _build_previous_data_summary(prev_rounds)
        prompt = QUERY_INSTRUCTION_PROMPT.format(
            task_description=clean_td,
            previous_data=previous_data or "（无）",
            prev_reason=prev_reason,
            prev_need_more=prev_need_more,
        )

        _log.info("查询指令生成 (selector LLM)...")
        model = get_model("diagnosis_selector")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请生成完整的K8s查询指令"},
        ])
        instruction = get_llm_text(response).strip()
        _log.info(f"查询指令: {instruction[:120]}...")
        return instruction

    async def _generate_rag_query(
        self, clean_td: str, prev_rounds: list,
        prev_reason: str, prev_need_more: str,
    ) -> str:
        previous_rag_results = _build_rag_context(prev_rounds)
        prompt = RAG_OPTIMIZE_PROMPT.format(
            task_description=clean_td,
            previous_rag_results=previous_rag_results,
            prev_reason=prev_reason,
            prev_need_more=prev_need_more,
        )

        _log.info("RAG 检索优化 (selector LLM)...")
        model = get_model("diagnosis_selector")
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请生成优化后的检索查询文本"},
        ])
        query = get_llm_text(response).strip()
        _log.info(f"RAG 优化查询: {query[:120]}...")
        return query

    # ============================================================
    # 推理判断
    # ============================================================

    async def _reason(
        self, rounds: list[RoundData], task_description: str, is_last: bool
    ) -> dict:
        data_text = "\n\n".join(rd.build_text_block() for rd in rounds)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        if is_last:
            prompt = FORCE_REPORT_PROMPT.format(
                total_rounds=len(rounds),
                task_description=task_description,
                data_text=data_text,
                start_time=self._start_time,
                current_time=current_time,
            )
        else:
            prompt = REASONING_PROMPT.format(
                task_description=task_description,
                data_text=data_text,
                start_time=self._start_time,
                current_time=current_time,
            )

        model_name = "diagnosis_reasoning"
        _log.info(f"推理判断 (reasoning LLM) 第{len(rounds)}轮...")

        model = get_model(model_name)
        response = await model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请根据收集到的数据进行判断"},
        ])
        text = get_llm_text(response)
        decision = parse_json_from_llm(text)

        can_complete = decision.get("can_complete", False)
        if can_complete:
            _log.info("可以完成任务，生成最终报告")
        else:
            reason = decision.get("reason", "")
            need_more = decision.get("need_more", "")
            _log.info(f"不能完成任务: {reason[:80]}...")
            _log.info(f"还需要: {need_more[:80]}...")

        return decision
