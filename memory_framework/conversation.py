"""对话数据加载与灌入。

从 JSON 文件读取一段对话(校验结构),并可将其灌入 ``MemoryStore``
由 mem0 抽取长期记忆。JSON 结构::

    {"user_id": "alice", "messages": [{"role": "user", "content": "..."}, ...]}
"""

import json


class InvalidConversationError(ValueError):
    """对话 JSON 结构非法时抛出。"""


def load_conversation(path: str) -> dict:
    """读取并校验一段对话 JSON,返回 ``{"user_id", "messages"}``。

    Raises:
        InvalidConversationError: 缺少/非法的 ``user_id``,或 ``messages`` 非列表,
            或某条 message 缺 ``role``/``content``。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "user_id" not in data or not isinstance(data["user_id"], str):
        raise InvalidConversationError(f"{path}: 缺少或非法的 user_id")
    if "messages" not in data or not isinstance(data["messages"], list):
        raise InvalidConversationError(f"{path}: messages 必须是列表")
    for m in data["messages"]:
        if "role" not in m or "content" not in m:
            raise InvalidConversationError(f"{path}: message 缺 role/content")
    return data


def ingest(store, conv: dict) -> dict:
    """把一段对话灌入 store,由 mem0 抽取记忆。返回 mem0 的 add 结果。"""
    return store.add(messages=conv["messages"], user_id=conv["user_id"])
