"""
utils/ — 项目级可复用工具模块

子模块:
    llm.py              LLM 配置加载 + 模型工厂 + Embedding 创建
    llm_utils.py        LLM 通用工具（跨 Agent 共用）: get_llm_text / parse_json_from_llm
    rag.py              RAG 知识库 (ChromaDB 持久化) — 巡检报告 + 工作记录
    rag_retrieval.py    RAG 巡检报告检索封装 — LLM 时间判断 + 语义匹配 + metadata.date 过滤
    write_guard.py      写入保护计数器 — Ctrl+C 优雅退出时防止关键文件写入中断
    async_utils.py      异步并发工具: run_parallel
    conversation.py     会话管理 (ConversationManager)
    task_context.py     Agent 间任务上下文 (TaskContext)
"""
