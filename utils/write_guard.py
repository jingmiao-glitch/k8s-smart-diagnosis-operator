"""
写入保护计数器 — Ctrl+C 优雅退出时确保关键写入不被中断

使用方式:
    from utils.write_guard import critical_write

    with critical_write():
        filepath.write_text(content, encoding="utf-8")

    # 或者手动 enter/exit（用于需要精细化控制的场景）
    from utils.write_guard import enter_write, exit_write
    enter_write()
    try:
        ...
    finally:
        exit_write()

    # shutdown 时查询
    from utils.write_guard import is_writing
    while is_writing():
        await asyncio.sleep(1)
"""

import threading

_count = 0
_lock = threading.Lock()


def enter_write():
    """写入开始，计数加 1"""
    global _count
    with _lock:
        _count += 1


def exit_write():
    """写入完成，计数减 1"""
    global _count
    with _lock:
        _count -= 1


def is_writing() -> bool:
    """是否有正在进行的写入操作"""
    global _count
    with _lock:
        return _count > 0


class critical_write:
    """上下文管理器，自动 enter/exit"""

    def __enter__(self):
        enter_write()
        return self

    def __exit__(self, *args):
        exit_write()
