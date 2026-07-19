"""长期记忆存储封装。

``MemoryStore`` 是对 mem0ai 2.0.11 ``mem0.Memory`` 的薄封装,隔离 mem0 内部
细节,对外提供稳定的增删改查接口。

与 mem0 2.0.11 真实 API 的差异(以 ``inspect.signature`` 探查为准):

- ``Memory.search`` / ``Memory.get_all`` 使用 ``filters={"user_id": ...}`` 传递
  实体 ID,而**不接受**顶层 ``user_id=`` 关键字参数(``get_all`` 内部有
  ``_reject_top_level_entity_params`` 会拒绝)。
- ``search`` 的结果数量参数是 ``top_k``(非 ``limit``)。
- ``search`` / ``get_all`` 返回 ``{"results": [...]}`` 结构,本封装统一解包为 list。
- ``update`` 的第二参 ``data`` 为位置参数。

创建 ``Memory`` 前必须先让 litellm 环境变量生效,故 ``__init__`` 使用
``build_config_and_apply_env()``。
"""

from mem0 import Memory

from memory_framework.config import build_config_and_apply_env


class MemoryStore:
    """对 mem0 ``Memory`` 的薄封装,提供干净的记忆增删改查接口。"""

    def __init__(self) -> None:
        # 必须先 apply_litellm_env(经 build_config_and_apply_env 完成),
        # 否则 litellm 拿不到中转站基址/密钥,真实 API 调用会失败。
        self._memory = Memory.from_config(build_config_and_apply_env())

    def add(self, messages: list[dict], user_id: str) -> dict:
        """写入一批对话消息,由 mem0 抽取并存储长期记忆。"""
        return self._memory.add(messages, user_id=user_id)

    def search(self, query: str, user_id: str, limit: int = 5) -> list:
        """按语义检索某用户的记忆,统一返回 list。"""
        res = self._memory.search(
            query, filters={"user_id": user_id}, top_k=limit
        )
        return self._unwrap(res)

    def get_all(self, user_id: str) -> list:
        """列出某用户的全部记忆,统一返回 list。"""
        res = self._memory.get_all(filters={"user_id": user_id})
        return self._unwrap(res)

    def update(self, memory_id: str, data: str) -> dict:
        """更新指定记忆的内容。"""
        return self._memory.update(memory_id, data)

    def delete(self, memory_id: str) -> None:
        """删除指定记忆。"""
        self._memory.delete(memory_id)

    def delete_all(self, user_id: str) -> None:
        """删除某用户的全部记忆。"""
        self._memory.delete_all(user_id=user_id)

    @staticmethod
    def _unwrap(res) -> list:
        """把 mem0 的 ``{"results": [...]}`` 或裸 list 统一解包为 list。"""
        if isinstance(res, dict):
            return res.get("results", [])
        return res
