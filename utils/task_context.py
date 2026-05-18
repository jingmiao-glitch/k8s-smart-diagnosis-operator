"""
TaskContext — Agent 间任务上下文

Python 标准库 dataclass，作为一次用户请求在 4 个 Agent 之间流转的数据篮子。
分配Agent 创建并写入任务描述，子Agent 读取任务、写入中间结果，分配Agent 最终汇总。

使用方式:
    from utils.task_context import TaskContext, INTENT_COMPLEX

    ctx = TaskContext(
        task_description="诊断 default 命名空间下的 Pod 异常原因",
        intent=INTENT_COMPLEX,
    )
    ctx.start()

    # 诊断Agent 写入结果
    ctx.set_result("diagnosis", {"root_cause": "...", "confidence": "high"})
    ctx.complete()

    # 分配Agent 读取结果
    result = ctx.get_result("diagnosis")
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Any

# 意图常量
INTENT_SIMPLE = "simple"            # 简单查询，直接转给查询Agent
INTENT_COMPLEX = "complex"          # 复杂诊断，转给诊断Agent
INTENT_INSPECTION = "inspection"    # 巡检请求，转给巡检Agent

# 任务状态常量
STATUS_PENDING = "pending"          # 已创建，等待执行
STATUS_RUNNING = "running"          # 执行中
STATUS_COMPLETED = "completed"      # 已完成
STATUS_FAILED = "failed"            # 执行失败


@dataclass
class TaskContext:
    """
    任务上下文 — 4 个 Agent 之间传递的数据篮子

    字段说明:
        task_id:            自动生成的12位唯一任务ID
        task_description:   分配Agent 写给子Agent 的任务描述（不是用户原话）
        intent:             意图类型（simple / complex / inspection）
        status:             任务生命周期状态
        intermediate_results: 各子Agent 写入的中间结果 {"agent_name": result}
        max_retry:          诊断Agent 最大重试次数
        current_retry:      诊断Agent 当前重试次数
        created_at:         任务创建时间戳
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_description: str = ""
    intent: str = ""
    status: str = STATUS_PENDING
    intermediate_results: dict = field(default_factory=dict)
    max_retry: int = 1
    current_retry: int = 0
    created_at: float = field(default_factory=time.time)

    # ---- 生命周期管理 ----

    def start(self):
        """标记任务开始执行"""
        self.status = STATUS_RUNNING

    def complete(self):
        """标记任务执行完成"""
        self.status = STATUS_COMPLETED

    def fail(self):
        """标记任务执行失败"""
        self.status = STATUS_FAILED

    # ---- 中间结果读写 ----

    def set_result(self, agent_name: str, result: Any):
        """
        子Agent 写入中间结果

        agent_name: 如 "query" / "diagnosis" / "inspection"
        result: 任意可序列化数据
        """
        self.intermediate_results[agent_name] = result

    def get_result(self, agent_name: str):
        """
        分配Agent 读取某个子Agent 的中间结果

        返回 None 如果该 Agent 尚未写入
        """
        return self.intermediate_results.get(agent_name)

    # ---- 重试控制（诊断Agent 专用） ----

    def can_retry(self) -> bool:
        """诊断Agent 判断是否还能重试"""
        return self.current_retry < self.max_retry

    def inc_retry(self):
        """诊断Agent 重试计数 +1"""
        self.current_retry += 1
