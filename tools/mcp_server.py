"""
MCP Server 统一入口 — FastMCP 实例 + add_tool 注册

将 k8s_tools（11 个 K8s 查询工具）和 prometheus_tools（6 个 PromQL 工具）
通过 FastMCP.add_tool() 注册到同一个 MCP 实例，监听 127.0.0.1:1315 供 Agent 进程内调用。

启动方式（同进程内）：
    from tools.mcp_server import run_mcp
    await run_mcp()  # 阻塞，启动 uvicorn 服务器

也可以通过 stdin/stdout 运行（开发调试）：
    python tools/mcp_server.py --transport streamable-http
"""

from mcp.server.fastmcp import FastMCP
from starlette.responses import Response
import asyncio

from . import k8s_tools, prometheus_tools

mcp = FastMCP(
    "k8s-agent-tools",
    host="127.0.0.1",
    port=1315,
    log_level="WARNING",
)

mcp.add_tool(k8s_tools.get_nodes, name="get_nodes", description="查询集群所有节点状态")
mcp.add_tool(k8s_tools.get_pods, name="get_pods", description="查询指定命名空间的 Pod 列表")
mcp.add_tool(k8s_tools.describe_pod, name="describe_pod", description="查询单个 Pod 的详细信息")
mcp.add_tool(k8s_tools.describe_node, name="describe_node", description="查询单个节点的详细信息")
mcp.add_tool(k8s_tools.get_events, name="get_events", description="查询指定命名空间的事件")
mcp.add_tool(k8s_tools.get_namespaces, name="get_namespaces", description="查询所有命名空间")
mcp.add_tool(k8s_tools.get_workloads, name="get_workloads", description="查询控制器列表（deployment/statefulset/daemonset）")
mcp.add_tool(k8s_tools.describe_controller, name="describe_controller", description="查询控制器详细信息")
mcp.add_tool(k8s_tools.logs, name="logs", description="获取 Pod 容器日志")
mcp.add_tool(k8s_tools.top_node, name="top_node", description="查看节点资源使用量（需 metrics-server）")
mcp.add_tool(k8s_tools.top_pod, name="top_pod", description="查看 Pod 资源使用量（需 metrics-server）")

mcp.add_tool(prometheus_tools.prom_query_instant, name="prom_query_instant", description="PromQL 即时查询")
mcp.add_tool(prometheus_tools.prom_query_range, name="prom_query_range", description="PromQL 范围查询（时序数据）")
mcp.add_tool(prometheus_tools.node_cpu_usage, name="node_cpu_usage", description="节点 CPU 使用率趋势")
mcp.add_tool(prometheus_tools.node_memory_usage, name="node_memory_usage", description="节点内存使用率趋势")
mcp.add_tool(prometheus_tools.pod_cpu_usage, name="pod_cpu_usage", description="Pod CPU 使用率趋势")
mcp.add_tool(prometheus_tools.pod_memory_usage, name="pod_memory_usage", description="Pod 内存使用率趋势")


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    return Response("ok\n", media_type="text/plain", status_code=200)


async def run_mcp():
    """同进程内启动 MCP Server（uvicorn 阻塞直到信号退出）"""
    await mcp.run(transport="streamable-http")


if __name__ == "__main__":
    asyncio.run(mcp.run(transport="streamable-http"))
