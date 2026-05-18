"""
会话记录管理 — 持久化用户与调度Agent 的自然语言对话历史

存储位置: config/conversations/{YYYY-MM-DD-HH:MM}.json
仅存储发送给用户的内容（assistant 回复），不存储 Agent 间原始数据。

会话生命周期:
    - 服务启动后首次请求自动创建第一个会话
    - LLM 识别到 new_session 意图时创建新会话
    - 其他情况下复用当前会话

使用方式:
    from utils.conversation import ConversationManager

    cm = ConversationManager()
    session_id = cm.create_session()
    cm.append(session_id, "user", "集群最近不稳定")
    cm.append(session_id, "assistant", "正在诊断...")
    history = cm.get_history(session_id)
"""

import json
from datetime import datetime
from pathlib import Path

from config import CONVERSATIONS_DIR
from utils.write_guard import critical_write


class ConversationManager:

    def __init__(self):
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        """
        创建新会话，文件名格式: YYYY-MM-DD-HH:MM.json
        如果同一分钟已有文件，自动追加 -2, -3 序号
        """
        now = datetime.now()
        base = now.strftime("%Y-%m-%d-%H:%M")
        session_id = base
        idx = 2
        while (CONVERSATIONS_DIR / f"{session_id}.json").exists():
            session_id = f"{base}-{idx}"
            idx += 1
        path = CONVERSATIONS_DIR / f"{session_id}.json"
        with critical_write():
            path.write_text("[]", encoding="utf-8")
        return session_id

    def exists(self, session_id: str) -> bool:
        return (CONVERSATIONS_DIR / f"{session_id}.json").exists()

    def append(self, session_id: str, role: str, content: str):
        path = CONVERSATIONS_DIR / f"{session_id}.json"
        if path.exists():
            messages = json.loads(path.read_text(encoding="utf-8"))
        else:
            messages = []
        messages.append({"role": role, "content": content})
        with critical_write():
            path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_history(self, session_id: str) -> list:
        path = CONVERSATIONS_DIR / f"{session_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def clear(self, session_id: str):
        path = CONVERSATIONS_DIR / f"{session_id}.json"
        if path.exists():
            path.unlink()
