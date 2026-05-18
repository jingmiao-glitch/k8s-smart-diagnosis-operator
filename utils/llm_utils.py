"""
LLM 通用工具 — 跨 Agent 共用的 LLM 响应解析函数

原本在 4 个 Agent 文件中各重复定义了一份 _get_text()，
以及在 6 处重复了 JSON 清理 + json.loads() 逻辑。
统一提取到此模块，消除代码重复。

使用方式:
    from utils.llm_utils import get_llm_text, parse_json_from_llm

    response = await model(messages)
    text = get_llm_text(response)
    data = parse_json_from_llm(text)
"""

import json


def get_llm_text(response) -> str:
    """
    从 AgentScope OpenAIChatModel 的 response 中提取文本内容

    response.content 是一个列表，每个元素是 dict，其中 "type" 字段区分内容类型。
    LLM 的文字回复类型为 "text"，本函数只提取第一个 text 块。

    参数:
        response: AgentScope OpenAIChatModel 调用后的响应对象

    返回:
        LLM 生成的纯文本字符串。如果 response.content 中没有任何 text 块，返回空字符串。
    """
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block["text"]
    return ""


def parse_json_from_llm(text: str) -> dict:
    """
    从 LLM 返回的文本中解析 JSON

    LLM 输出的 JSON 通常被包裹在 ```json ... ``` 代码块中，前后可能有空格。
    本函数依次剥离这些包裹物，然后调用 json.loads()。

    参数:
        text: LLM 返回的原始文本（通常来自 get_llm_text()）

    返回:
        解析后的 Python dict
    """
    cleaned = text.strip().strip("```json").strip("```").strip()
    return json.loads(cleaned)
