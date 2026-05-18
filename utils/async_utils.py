"""
异步并发工具 — 通用 asyncio 并行执行封装

原本在 dispatcher_agent 和 diagnosis_agent 中各有一份结构高度相似的
asyncio.wait() 并行执行模式，差异仅在于超时值和超时处理方式。
统一提取到此模块。

使用方式:
    from utils.async_utils import run_parallel

    tasks = {
        "query": asyncio.create_task(query_func()),
        "inspector": asyncio.create_task(inspector_func()),
    }
    results = await run_parallel(tasks, timeout=40)
    # results = {"query": {...}, "inspector": "..."}
"""

import asyncio


async def run_parallel(tasks: dict, timeout: float, on_timeout=None) -> dict:
    """
    并行执行多个 asyncio Task，统一处理结果收集和超时取消

    参数:
        tasks:   {"name": asyncio.Task, ...}  任务字典
        timeout: 最大等待秒数
        on_timeout: 可选回调 on_timeout(name, timeout) → value，
                   用于定制超时任务的返回内容。
                   不传则超时任务不出现在返回的 dict 中。

    返回:
        {"name": task_result, ...}
        - 正常完成: 返回 task.result()
        - 执行异常: 返回 None
        - 超时:     调用 on_timeout(name, timeout)，不传则跳过

    注意:
        异常信息不会在此函数内记录日志，调用方自行决定如何处理。
    """
    done, pending = await asyncio.wait(tasks.values(), timeout=timeout)

    results = {}
    for name, task in tasks.items():
        if task in done:
            try:
                results[name] = task.result()
            except Exception:
                results[name] = None
        else:
            task.cancel()
            if on_timeout is not None:
                results[name] = on_timeout(name, timeout)

    return results
