"""
LLM 配置加载与模型工厂

读取 config/llm_config.json，为每个 Agent 提供专属的 OpenAIChatModel 实例，
同时提供 LlamaIndex 兼容的 Embedding 模型创建。

所有模型通过火山引擎 OpenAI 兼容 API (openai_compatible) 调用。

使用方式:
    from utils.llm import get_model

    model = get_model("inspector")          # 巡检Agent 的 LLM
    response = await model([{"role": "...", "content": "..."}])
    text = response.content[0]["text"]
"""

import json
import os

from agentscope.model import OpenAIChatModel

PROVIDER = "openai_compatible"
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_config.json")

_config_cache = None


def _load_config():
    """读取 llm_config.json，进程生命周期内只读一次（带缓存）"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = json.load(f)
    return _config_cache


def _build_openai_chat_model(model_config: dict) -> OpenAIChatModel:
    """
    从单个 model 配置字典构建 AgentScope OpenAIChatModel 实例

    参数来源: llm_config.json 中 llm.{name} 节点
    自动处理: temperature / enable_thinking → extra_body.thinking
    """
    model_name = model_config["model_name"]
    api_key = model_config["api_key"]
    base_url = model_config.get("base_url", "")
    stream = model_config.get("stream", False)
    enable_thinking = model_config.get("enable_thinking", False)
    temperature = model_config.get("temperature", 0.0)

    generate_kwargs = {
        "temperature": temperature,
        "extra_body": {
            "thinking": {"type": "enabled" if enable_thinking else "disabled"}
        },
    }

    return OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=stream,
        client_kwargs={"base_url": base_url},
        generate_kwargs=generate_kwargs,
    )


# ============================================================
# 4 个 Agent 各自的 LLM 工厂函数（诊断Agent 使用两个模型）
# ============================================================

def get_dispatcher_model() -> OpenAIChatModel:
    """调度Agent模型 — 信息整合 + Agent调用判断"""
    config = _load_config()
    return _build_openai_chat_model(config["llm"]["dispatcher"])


def get_query_model() -> OpenAIChatModel:
    """查询模型 — 调用工具查询 + 内容整合，能力要求不高"""
    config = _load_config()
    return _build_openai_chat_model(config["llm"]["query"])


def get_inspector_model() -> OpenAIChatModel:
    """巡检模型 — 采集数据 + 生成巡检报告"""
    config = _load_config()
    return _build_openai_chat_model(config["llm"]["inspector"])


def get_diagnosis_selector_model() -> OpenAIChatModel:
    """诊断调度模型 — 判断需要哪些Agent执行诊断"""
    config = _load_config()
    return _build_openai_chat_model(config["llm"]["diagnosis_selector"])


def get_diagnosis_reasoning_model() -> OpenAIChatModel:
    """诊断推理模型 — 强推理，逻辑严谨，分析诊断根因"""
    config = _load_config()
    return _build_openai_chat_model(config["llm"]["diagnosis_reasoning"])


_llm_registry = {
    "dispatcher": get_dispatcher_model,
    "query": get_query_model,
    "inspector": get_inspector_model,
    "diagnosis_selector": get_diagnosis_selector_model,
    "diagnosis_reasoning": get_diagnosis_reasoning_model,
}


def get_model(name: str) -> OpenAIChatModel:
    """
    根据名称获取对应 Agent 的 LLM 实例

    name 可选: dispatcher / query / inspector / diagnosis_selector / diagnosis_reasoning
    """
    if name not in _llm_registry:
        raise KeyError(f"未知模型名: {name}，可选: {list(_llm_registry.keys())}")
    return _llm_registry[name]()


def get_embedding_config(kb_name: str) -> dict:
    """
    获取指定知识库的 embedding 配置（含 model + ingest + retrieve）

    kb_name 可选: inspection_reports / work_records
    """
    config = _load_config()
    if kb_name not in config["embedding"]:
        raise KeyError(f"未知知识库: {kb_name}，可选: {list(config['embedding'].keys())}")
    return config["embedding"][kb_name]


def create_embeddings():
    """
    创建两个知识库对应的 LlamaIndex VolCanoEmbedding 实例

    返回:
        {
            "inspection_reports": VolCanoEmbedding(...),
            "work_records": VolCanoEmbedding(...),
        }

    通过 OpenAI SDK 调用火山引擎 embedding API
    """
    from openai import OpenAI
    from llama_index.core.embeddings import BaseEmbedding

    config = _load_config()

    inspection_config = config["embedding"]["inspection_reports"]["model"]
    work_records_config = config["embedding"]["work_records"]["model"]

    class VolCanoEmbedding(BaseEmbedding):
        """火山引擎 doubao-embedding-vision 兼容的 LlamaIndex Embedding"""

        def __init__(self, model_name, api_key, base_url, dimensions):
            super().__init__(model_name=model_name)
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._model = model_name
            self._dimensions = dimensions

        def _get_text_embedding(self, text):
            resp = self._client.embeddings.create(
                input=[text],
                model=self._model,
                dimensions=self._dimensions,
            )
            return resp.data[0].embedding

        def _get_query_embedding(self, query):
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query):
            return self._get_text_embedding(query)

        async def _aget_text_embedding(self, text):
            return self._get_text_embedding(text)

    return {
        "inspection_reports": VolCanoEmbedding(
            model_name=inspection_config["model_name"],
            api_key=inspection_config["api_key"],
            base_url=inspection_config["base_url"],
            dimensions=inspection_config["dimensions"],
        ),
        "work_records": VolCanoEmbedding(
            model_name=work_records_config["model_name"],
            api_key=work_records_config["api_key"],
            base_url=work_records_config["base_url"],
            dimensions=work_records_config["dimensions"],
        ),
    }
