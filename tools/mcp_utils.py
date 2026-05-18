"""
MCP 客户端共享工具 — 供 query_agent 和 inspector_agent 共用

提供：
  - fetch_mcp_tools()    通过 tools/list 动态拉取可用工具列表（带缓存）
  - call_mcp_tools()     调用 MCP 工具的统一入口（initialize → tools/call）
"""

import json
from datetime import datetime
from pathlib import Path

from config import MCP_SERVER_URL, MCP_LOG_DIR

_tool_cache: list | None = None


async def fetch_mcp_tools() -> list:
    """
    通过 MCP tools/list 协议从 MCP Server 动态拉取可用工具列表

    返回: [{"name": "...", "description": "...", "inputSchema": {...}}, ...]

    进程生命周期内只请求一次（缓存），新增工具后重启服务即可生效。
    无需在 agent 代码中手动维护工具列表。
    """
    global _tool_cache
    if _tool_cache is not None:
        return _tool_cache

    import aiohttp

    async with aiohttp.ClientSession() as http:
        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-client", "version": "1.0"},
            },
            "id": 1,
        }
        async with http.post(
            f"{MCP_SERVER_URL}/mcp",
            json=init_payload,
            headers={"Accept": "application/json, text/event-stream"},
        ) as resp:
            session_id = resp.headers.get("mcp-session-id")

        await http.post(
            f"{MCP_SERVER_URL}/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id or "",
            },
        )

        list_payload = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
        async with http.post(
            f"{MCP_SERVER_URL}/mcp",
            json=list_payload,
            headers={
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id or "",
            },
        ) as resp:
            body = await resp.text()
            for line in body.split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    _tool_cache = data.get("result", {}).get("tools", [])
                    break

    return _tool_cache


async def call_mcp_tools(tool_calls: list, agent_name: str = "agent") -> dict:
    """
    通过 MCP streamable-http 协议调用 MCP Server 的工具

    参数:
      tool_calls  [{"tool": "get_nodes", "args": {}}, ...]
      agent_name  调用方名称（用于日志区分）

    返回: {"tool_name": "result_text", ...}
    """
    import aiohttp

    results = {}
    session_id = None

    async with aiohttp.ClientSession() as http:
        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": f"{agent_name}-agent", "version": "1.0"},
            },
            "id": 1,
        }
        async with http.post(
            f"{MCP_SERVER_URL}/mcp",
            json=init_payload,
            headers={"Accept": "application/json, text/event-stream"},
        ) as resp:
            session_id = resp.headers.get("mcp-session-id")

        await http.post(
            f"{MCP_SERVER_URL}/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id or "",
            },
        )

        for idx, call in enumerate(tool_calls):
            name = call["tool"]
            args = call.get("args", {})

            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args},
                    "id": idx + 2,
                }
                async with http.post(
                    f"{MCP_SERVER_URL}/mcp",
                    json=payload,
                    headers={
                        "Accept": "application/json, text/event-stream",
                        "mcp-session-id": session_id or "",
                    },
                ) as resp:
                    body = await resp.text()
                    result_text = body
                    for line in body.split("\n"):
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data.get("result") and "content" in data["result"]:
                                parts = []
                                for c in data["result"]["content"]:
                                    if isinstance(c, dict) and "text" in c:
                                        parts.append(c["text"])
                                result_text = "\n".join(parts)

                    results[name] = result_text
            except Exception as e:
                results[name] = f"错误: {e}"

    _write_mcp_log(agent_name, results)
    return results


def _write_mcp_log(agent_name: str, results: dict):
    MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = MCP_LOG_DIR / f"{ts}_{agent_name}.log"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"调用方: {agent_name}\n")
        f.write(f"时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"工具数: {len(results)}\n")
        f.write("=" * 60 + "\n\n")
        for name, text in results.items():
            f.write(f"工具: {name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{text}\n")
            f.write("=" * 60 + "\n\n")
