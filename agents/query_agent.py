"""
查询Agent — 接收任务描述，通过 MCP 协议调用 K8s Query Server 查询集群数据，整合后返回结构化结果

核心逻辑:
    1. 第一次 LLM: 根据任务描述决定调用哪些工具及参数
       （工具列表通过 MCP tools/list 协议动态拉取，无需手动维护）
    2. 通过 MCP streamable-http 协议调用真实 K8s 工具收集数据
    3. 第二次 LLM: 整合所有工具返回数据，生成结构化 dict
    4. 返回结果给调用方（调度Agent / 诊断Agent）

使用方式:
    from agents.query_agent import QueryAgent

    agent = QueryAgent()
    result = await agent.execute_query("查 default 命名空间下所有异常 Pod")

    # result 格式:
    # {"success": True, "total_commands": 2, "results": [...], "summary": "..."}

依赖:
    MCP Server 同进程运行在 127.0.0.1:1315
"""

from utils.llm import get_model
from utils.llm_utils import get_llm_text, parse_json_from_llm
from logger import get_agent_logger

_log = get_agent_logger("query")


# ============================================================
# MCP 工具动态发现 — 通过 tools/list 拉取，无需手动维护
# ============================================================

from tools.mcp_utils import fetch_mcp_tools, call_mcp_tools


TOOL_SELECTION_PROMPT = """你是 Kubernetes 集群运维专家。根据用户的任务描述，判断需要调用哪些 MCP 工具来完成查询任务。

## 重要规则

1. 如果任务是排查"集群内"、"所有Pod"、"全集群"的故障，**必须先调用 get_namespaces 获取所有命名空间**，再对每个命名空间调 get_pods，不要只查 default。
2. 如果任务需要查 Pod 日志或详情，先用 get_pods 列出目标命名空间的 Pod，从结果中提取 Pod 名称，再调 describe_pod 或 logs。
3. 如果任务涉及资源使用趋势，优先调 Prometheus 工具（node_cpu_usage 等）而不是 top_node/top_pod。

## 可用工具
{tool_list}

## 输出要求
只输出一个 JSON 数组，每个元素包含 "tool"（工具名）和 "args"（参数字典）。
按照合理的执行顺序排列。

示例:
用户任务: "查 default 命名空间下所有异常 Pod"
输出: [
  {{"tool": "get_pods", "args": {{"namespace": "default"}}}},
  {{"tool": "get_events", "args": {{"namespace": "default"}}}}
]

用户任务: "分析集群内Pod重启次数过多的根因"
输出: [
  {{"tool": "get_namespaces", "args": {{}}}},
  {{"tool": "get_pods", "args": {{"namespace": "kube-system"}}}},
  {{"tool": "get_events", "args": {{"namespace": "kube-system"}}}},
  {{"tool": "get_pods", "args": {{"namespace": "default"}}}}
]

用户任务: "查 node3 的详细信息和资源使用"
输出: [
  {{"tool": "describe_node", "args": {{"name": "node3"}}}},
  {{"tool": "top_node", "args": {{}}}}
]

用户任务: "查 nginx-deploy-7d4f8b-x9k2m Pod 的日志"
输出: [
  {{"tool": "logs", "args": {{"pod_name": "nginx-deploy-7d4f8b-x9k2m", "namespace": "default", "tail_lines": 100}}}}
]

用户任务: "查集群所有节点"
输出: [
  {{"tool": "get_nodes", "args": {{}}}}
]

用户任务: "查 Prometheus 监控目标 up 状态"
输出: [
  {{"tool": "prom_query_instant", "args": {{"query": "up"}}}}
]

用户任务: "查 node3 最近30分钟的 CPU 使用率趋势"
输出: [
  {{"tool": "node_cpu_usage", "args": {{"node_name": "node3", "minutes": 30}}}}
]

用户任务: "查 redis-master-0 Pod 最近1小时内存使用量变化"
输出: [
  {{"tool": "pod_memory_usage", "args": {{"pod_name": "redis-master-0", "namespace": "default"}}}},
  {{"tool": "pod_cpu_usage", "args": {{"pod_name": "redis-master-0", "namespace": "default", "minutes": 60}}}}
]

用户任务: "查 kube-system 命名空间 Controller 是否都正常"
输出: [
  {{"tool": "get_workloads", "args": {{"kind": "all", "namespace": "kube-system", "wide": true}}}}
]"""

RESULT_INTEGRATION_PROMPT = """你是 Kubernetes 集群运维专家。请根据工具返回的原始数据，整合生成一个结构化的查询报告。

## 输出格式（严格 JSON）

{{
  "success": true/false,
  "total_commands": N,
  "results": [
    {{"tool": "工具名", "data": "工具返回的原始数据"}}
  ],
  "summary": "一句话总结查询发现的关键信息"
}}

## 要求
- results 数组中按调用顺序列出每个工具名和返回数据
- data 字段保留工具的完整返回内容（表格等格式）
- summary 总结异常点、关键指标，不要重复整个数据
- 如果所有工具都正常执行，success 为 true

以下是工具返回数据："""

# ============================================================
# MCP 客户端（streamable-http 协议）
# ============================================================


async def _mcp_call_tools(tool_calls: list) -> list:
    """
    通过共享 MCP 客户端调用工具

    返回: [{"tool": "get_pods", "data": "..."}, ...]
    """
    results_dict = await call_mcp_tools(tool_calls, agent_name="query")
    return [{"tool": name, "data": data} for name, data in results_dict.items()]


# ============================================================
# LLM 工具选择
# ============================================================

async def _select_tools(task_description: str) -> list:
    """
    第一次 LLM 调用: 根据任务描述决定调用哪些工具及参数

    工具列表从 MCP Server 动态拉取（tools/list 协议），
    在 mcp_server.py 中新增工具后，重启服务即可自动生效，无需修改此处代码。

    输入: 任务描述字符串
    输出: [{"tool": "get_pods", "args": {"namespace": "default"}}, ...]
    """
    tools = await fetch_mcp_tools()
    tool_desc = "\n".join(
        f"- {t['name']}: {t.get('description', '无描述')}" for t in tools
    )
    prompt = TOOL_SELECTION_PROMPT.format(tool_list=tool_desc)

    _log.info(f"任务: {task_description}")
    _log.info("第1次 LLM 调用 → 选择工具...")

    model = get_model("query")
    response = await model([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"任务描述: {task_description}"},
    ])
    text = get_llm_text(response)
    calls = parse_json_from_llm(text)
    names = [c["tool"] for c in calls]
    _log.info(f"已选择工具: {names}")
    return calls


# ============================================================
# 结果整合
# ============================================================

async def _integrate_results(tool_results: list) -> dict:
    """
    第二次 LLM 调用: 整合工具返回数据，生成结构化 dict

    输入: [{"tool": "get_pods", "data": "..."}, ...]
    输出: {"success": True, "total_commands": 2, "results": [...], "summary": "..."}
    """
    data_text = "\n\n---\n\n".join(
        f"工具: {r['tool']}\n{r['data']}"
        for r in tool_results
    )

    _log.info("第2次 LLM 调用 → 整合结果...")

    model = get_model("query")
    response = await model([
        {"role": "system", "content": RESULT_INTEGRATION_PROMPT},
        {"role": "user", "content": data_text},
    ])
    text = get_llm_text(response)
    result = parse_json_from_llm(text)
    _log.info(f"整合完成, success={result.get('success')}")
    return result


# ============================================================
# 核心函数
# ============================================================

async def execute_query(task_description: str) -> dict:
    """
    查询Agent 核心函数

    参数:
        task_description: 任务描述字符串

    返回:
        {"success": True, "total_commands": N, "results": [...], "summary": "..."}

    流程:
        1. 第一次 LLM → 选择需要调用的工具及参数
        2. 通过 MCP 协议调用 K8s 工具收集数据
        3. 第二次 LLM → 整合为结构化 dict
        4. 返回结果
    """
    tool_calls = await _select_tools(task_description)
    tool_results = await _mcp_call_tools(tool_calls)

    if not tool_results:
        return {
            "success": False,
            "total_commands": 0,
            "results": [],
            "summary": "未执行任何工具",
        }

    result = await _integrate_results(tool_results)
    result["total_commands"] = len(tool_results)
    return result


class QueryAgent:
    """
    查询Agent 实例封装

    使用方式:
        agent = QueryAgent()
        # 被调度Agent / 诊断Agent 调用
        result = await agent.execute_query("查 default 命名空间下所有异常 Pod")
    """

    async def execute_query(self, task_description: str) -> dict:
        return await execute_query(task_description)
