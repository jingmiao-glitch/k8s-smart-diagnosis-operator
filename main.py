"""
AI-Kubenetes 4-Agent 协作系统 — 主入口

启动:
    python main.py

HTTP 接口:
    POST /chat          {"message": "...", "session_id": "..."}
    GET  /healthz        健康检查

MCP Server:
    内嵌 FastMCP Server 在 127.0.0.1:1315 启动（供 Agent 内部调用）

会话管理:
    首次请求自动创建会话（文件名: YYYY-MM-DD-HH:MM.json）
    后续不传 session_id 则复用当前会话
    LLM 识别到 new_session 意图时创建新会话

日志系统:
    5 文件日志体系: logs/agent/{all,dispatcher,query,inspector,diagnosis}.log
    控制台同步输出，可通过 set_agent_debug("name") 按 Agent 切换 DEBUG 级别
"""

import asyncio
import json
import os
import signal
import sys

from aiohttp import web

from agents.dispatcher_agent import DispatcherAgent
from agents.inspector_agent import InspectorAgent
from logger import get_agent_logger
from utils.rag import RAGKnowledgeBase
from utils.write_guard import is_writing

_log = get_agent_logger("dispatcher")

dispatcher: DispatcherAgent | None = None
inspector: InspectorAgent | None = None
rag: RAGKnowledgeBase | None = None


def _json_response(data, status=200):
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        content_type="application/json",
        status=status,
    )


async def handle_chat(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        user_message = body.get("message", "").strip()
        session_id = body.get("session_id", "")

        if not user_message:
            return _json_response({"error": "message 不能为空"}, status=400)

        result = await dispatcher.handle(user_message, session_id)

        return _json_response(result)

    except Exception as e:
        _log.error(f"处理请求出错: {e}")
        return _json_response({"error": str(e)}, status=500)


async def handle_healthz(request: web.Request) -> web.Response:
    return web.Response(text="ok")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/chat", handle_chat)
    app.router.add_get("/healthz", handle_healthz)
    return app


async def main():
    global dispatcher, inspector

    _log.info("K8s故障智能排查系统启动中...")

    import logging as _logging
    for _noisy in ("uvicorn.access", "uvicorn.error", "mcp.server", "mcp"):
        _logging.getLogger(_noisy).setLevel(_logging.WARNING)

    from tools.mcp_server import mcp as mcp_server

    global rag
    rag = RAGKnowledgeBase()
    await rag.init_from_disk()

    global dispatcher, inspector
    dispatcher = DispatcherAgent(rag=rag)
    inspector = InspectorAgent(rag=rag)

    inspector.start_schedule(interval_seconds=1800)

    mcp_task = asyncio.create_task(
        asyncio.to_thread(mcp_server.run, transport="streamable-http")
    )

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 1314)

    loop = asyncio.get_event_loop()

    async def _shutdown():
        _log.info("收到退出信号，正在关闭...")
        inspector.stop_schedule()
        _log.info("巡检定时器停止中")

        current_task = asyncio.current_task()
        other_tasks = [t for t in asyncio.all_tasks(loop) if t is not current_task]
        for t in other_tasks:
            t.cancel()
        _log.info(f"已取消 {len(other_tasks)} 个任务")

        SHUTDOWN_TIMEOUT = 30
        waited = 0
        while is_writing() and waited < SHUTDOWN_TIMEOUT:
            _log.info(f"等待写入完成... ({waited}s / {SHUTDOWN_TIMEOUT}s)")
            await asyncio.sleep(1)
            waited += 1

        if is_writing():
            _log.warning(f"写入超时 ({SHUTDOWN_TIMEOUT}s)，强制退出")
        else:
            _log.info("服务已停止")

        os._exit(0)

    def _signal_handler():
        asyncio.ensure_future(_shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await site.start()
    _log.info("HTTP 服务已启动: http://0.0.0.0:1314")
    _log.info("MCP Server 已启动: http://127.0.0.1:1315")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
