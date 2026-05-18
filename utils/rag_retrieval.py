"""
RAG 巡检报告检索模块 — LLM 时间判断 + RAG 语义匹配 的通用封装

将"LLM 判断时间段 → 向量语义匹配 → metadata.date 精确过滤"的三步流程
封装为一个函数，供诊断Agent（selector 路径）和调度Agent（意图识别路径）共用。

使用方式:
    from utils.rag_retrieval import retrieve_inspection_with_timerange

    results = await retrieve_inspection_with_timerange(
        rag=rag_instance,
        task_description="诊断Pod重启原因",
        timerange_prompt=INSPECTION_TIMERANGE_PROMPT,
        model_name="dispatcher",
    )
    # results = [{"content": "...", "score": 0.85, "metadata": {...}}, ...]
"""

from datetime import datetime

from utils.llm import get_model
from utils.llm_utils import get_llm_text, parse_json_from_llm


async def retrieve_inspection_with_timerange(
    rag,
    task_description: str,
    timerange_prompt: str,
    model_name: str,
    logger=None,
) -> list:
    """
    检索巡检报告：LLM 自动判断时间段 + 语义匹配 + metadata.date 精确过滤

    三步流程:
        1. 注入当前时间 + task_description 到 timerange_prompt，
           调用 LLM 判断需要检索的 date_from ~ date_to
        2. 用 task_description 原样传入 retrieve_inspection 做向量语义匹配
        3. retrieve_inspection 内部用 metadata.date 做精确日期过滤，
           排除不在 [date_from, date_to] 范围内的结果

    参数:
        rag:                  RAGKnowledgeBase 实例
        task_description:     原始任务描述，直接作为语义匹配的 query 文本
        timerange_prompt:     LLM 时间判断的提示词模板，
                              需包含 {current_time} 和 {task_description} 两个占位符
        model_name:           用于时间判断的 LLM 模型名称（如 "dispatcher" / "diagnosis_selector"）
        logger:               可选的日志记录器，传入后自动记录时间范围和结果数量

    返回:
        检索结果的 list，格式:
        [
            {
                "content": "巡检报告正文片段",
                "score": 0.87,
                "metadata": {"date": "2026-05-17", "filename": "...", "type": "inspection"}
            },
            ...
        ]
    """
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M")

    prompt = timerange_prompt.format(
        current_time=current_time,
        task_description=task_description,
    )

    if logger:
        logger.info(f"巡检报告时间范围判断 (当前时间={current_time})...")

    model = get_model(model_name)
    response = await model([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "请判断需要检索的巡检报告时间范围"},
    ])

    text = get_llm_text(response)
    decision = parse_json_from_llm(text)

    date_from = decision.get("date_from", now.strftime("%Y-%m-%d"))
    date_to = decision.get("date_to", now.strftime("%Y-%m-%d"))

    if logger:
        logger.info(f"巡检报告时间范围: {date_from} ~ {date_to}")

    results = await rag.retrieve_inspection(
        task_description,
        date_from=date_from,
        date_to=date_to,
    )

    if logger:
        logger.info(f"巡检报告检索完成: {len(results)} 条")

    return results
