"""chat.reply 人设 + 注入开关测试。mock litellm 与 store,只断 system prompt。"""

from unittest.mock import patch

from memory_framework.chat import _build_system_prompt, reply


class FakeStore:
    def __init__(self, mems):
        self._mems = mems
        self.searched = 0

    def search(self, query, user_id, limit=5):
        self.searched += 1
        return self._mems


def test_build_prompt_persona_and_memories():
    sp = _build_system_prompt([{"memory": "用户叫小李"}], persona="回答尽量简洁")
    assert "回答尽量简洁" in sp
    assert "用户叫小李" in sp
    assert "人设" in sp


def test_build_prompt_no_persona():
    sp = _build_system_prompt([{"memory": "x"}], persona="")
    assert "人设" not in sp
    assert "x" in sp


@patch("memory_framework.chat.litellm.completion")
def test_reply_injects_persona_and_memory(mock_completion):
    mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
    store = FakeStore([{"memory": "用户是后端工程师"}])
    reply(store, "u", "你好", model="test/m",
          persona="你是我的中文编程助手", inject=True)
    sys_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "你是我的中文编程助手" in sys_prompt
    assert "用户是后端工程师" in sys_prompt
    assert store.searched == 1


@patch("memory_framework.chat.litellm.completion")
def test_reply_inject_off_skips_everything(mock_completion):
    mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
    store = FakeStore([{"memory": "用户是后端工程师"}])
    reply(store, "u", "你好", model="test/m",
          persona="你是我的中文编程助手", inject=False)
    sys_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    # 关掉注入:人设与记忆都不进 system,且不检索记忆(省一次 search)
    assert "你是我的中文编程助手" not in sys_prompt
    assert "用户是后端工程师" not in sys_prompt
    assert store.searched == 0
