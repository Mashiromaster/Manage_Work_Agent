"""项目 Q&A 长期记忆测试。假 store 断言 proj:: 命名空间 + prompt 注入;LLM mock。"""

from unittest.mock import patch

import memory_framework.project_chat as pc
from memory_framework.project_chat import (
    persist_project_turn,
    project_namespace,
    project_reply,
    sediment_analysis,
    sediment_change,
)


class FakeStore:
    """记录 add/search 调用的假 MemoryStore。"""

    def __init__(self, search_result=None):
        self.added = []          # [(user_id, messages)]
        self._search_result = search_result or []
        self.searched = []       # [(user_id, query)]

    def add(self, messages, user_id):
        self.added.append((user_id, messages))

    def search(self, query, user_id, limit=5):
        self.searched.append((user_id, query))
        return self._search_result


class FakeItem:
    def __init__(self, dimension, text):
        self.dimension = dimension
        self.text = text


class FakeProfile:
    def __init__(self, items):
        self.items = items


class FakeProjectStore:
    def __init__(self, items):
        self._items = items

    def get_project(self, project_id):
        return FakeProfile(self._items)


def test_namespace():
    assert project_namespace("Mem0") == "proj::Mem0"


@patch("memory_framework.project_chat.litellm.completion")
def test_reply_uses_namespace_and_injects_context(mock_completion):
    mock_completion.return_value = {
        "choices": [{"message": {"content": "回答"}}]}
    store = FakeStore(search_result=[{"memory": "上次讨论过登录流程"}])
    pstore = FakeProjectStore([
        FakeItem("progress", "完成结构化画像"),
        FakeItem("todo", "接 MCP"),
    ])

    # 注入代码摘要,验证一并进 prompt
    with patch("memory_framework.project_chat.load_code_analysis",
               return_value={"summaries": [
                   {"file": "app.py", "role": "UI", "summary": "Gradio 入口"}]}):
        answer = project_reply(store, pstore, "Mem0", "登录怎么做的?", model="test/m")

    assert answer == "回答"
    # 检索用了 proj:: 命名空间
    assert store.searched == [("proj::Mem0", "登录怎么做的?")]
    # system prompt 注入了进度四维 + 代码摘要 + 历史记忆
    sys_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "完成结构化画像" in sys_prompt
    assert "接 MCP" in sys_prompt
    assert "app.py" in sys_prompt
    assert "上次讨论过登录流程" in sys_prompt


@patch("memory_framework.project_chat.litellm.completion")
def test_reply_no_context_still_works(mock_completion):
    mock_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]}
    store = FakeStore()
    with patch("memory_framework.project_chat.load_code_analysis",
               return_value=None):
        answer = project_reply(store, None, "New", "你好", model="test/m")
    assert answer == "ok"
    sys_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "当前项目:New" in sys_prompt


def test_persist_turn_namespace():
    store = FakeStore()
    persist_project_turn(store, "Mem0", "问", "答")
    assert len(store.added) == 1
    uid, msgs = store.added[0]
    assert uid == "proj::Mem0"
    assert msgs[0]["content"] == "问" and msgs[1]["content"] == "答"


def test_sediment_analysis():
    store = FakeStore()
    sediment_analysis(store, "Mem0",
                      capinfo={"analyzed": 5, "incremental": False},
                      summaries=[{"file": "app.py"}, {"file": "chat.py"}])
    uid, msgs = store.added[0]
    assert uid == "proj::Mem0"
    text = msgs[0]["content"]
    assert "全量" in text and "5 个" in text and "app.py" in text


def test_sediment_analysis_incremental():
    store = FakeStore()
    sediment_analysis(store, "Mem0",
                      capinfo={"analyzed": 1, "incremental": True},
                      summaries=[{"file": "a.py"}])
    assert "增量" in store.added[0][1][0]["content"]


def test_sediment_change():
    store = FakeStore()
    sediment_change(store, "Mem0",
                    {"added": ["c.py"], "changed": ["a.py"], "removed": ["b.py"]})
    text = store.added[0][1][0]["content"]
    assert "新增 c.py" in text and "修改 a.py" in text and "删除 b.py" in text


def test_sediment_change_empty_noop():
    store = FakeStore()
    sediment_change(store, "Mem0", {"added": [], "changed": [], "removed": []})
    assert store.added == []
