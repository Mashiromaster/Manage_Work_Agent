"""项目维度提炼测试。mock litellm,不触网。"""

import json
from unittest.mock import patch

import pytest

import memory_framework.project_extractor as pe
from memory_framework.project_dims import PROJECT_DIMENSIONS
from memory_framework.project_extractor import synthesize_project_dims


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """重试退避的 sleep 置空,避免失败/空返回用例真实等待。"""
    monkeypatch.setattr(pe.time, "sleep", lambda *_: None)


class TestSynthesizeProjectDims:
    def test_empty_memories_returns_empty(self):
        assert synthesize_project_dims([], model="test/model") == []

    @patch("memory_framework.project_extractor.litellm.completion")
    def test_returns_parsed_project_items(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"dimension": "progress", "text": "聊天延迟降到 4s", "importance": 8, "evidence": "14s→4s"},
                {"dimension": "todo", "text": "换更快模型", "importance": 5, "evidence": "以后再优化"},
                {"dimension": "decision", "text": "回复先返回后台落盘", "importance": 7, "evidence": "降感知延迟"},
            ])}}]
        }
        items = synthesize_project_dims(
            memories=["把延迟从 14s 降到 4s", "还想换更快的模型"],
            model="test/model",
        )
        assert len(items) == 3
        dims = {it["dimension"] for it in items}
        assert dims == {"progress", "todo", "decision"}
        assert all(d in PROJECT_DIMENSIONS for d in dims)

    @patch("memory_framework.project_extractor.litellm.completion")
    def test_rejects_non_project_dimension(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"dimension": "progress", "text": "有效", "importance": 6},
                {"dimension": "personality", "text": "用户画像维度应被拒", "importance": 5},
            ])}}]
        }
        items = synthesize_project_dims(memories=["m"], model="test/model")
        assert len(items) == 1
        assert items[0]["dimension"] == "progress"

    @patch("memory_framework.project_extractor.litellm.completion")
    def test_llm_exception_returns_empty(self, mock_completion):
        mock_completion.side_effect = RuntimeError("API 不可用")
        assert synthesize_project_dims(memories=["m"], model="test/model") == []
        # 重试:1 次首发 + 3 次退避 = 4 次调用
        assert mock_completion.call_count == 4

    @patch("memory_framework.project_extractor.litellm.completion")
    def test_retry_then_succeed(self, mock_completion):
        # 前两次 429,第三次成功 → 重试把结果救回来
        calls = {"n": 0}

        def _resp(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429")
            return {"choices": [{"message": {"content": json.dumps([
                {"dimension": "progress", "text": "ok", "importance": 6}])}}]}
        mock_completion.side_effect = _resp
        items = synthesize_project_dims(memories=["m"], model="test/model")
        assert len(items) == 1 and items[0]["dimension"] == "progress"
        assert calls["n"] == 3

    @patch("memory_framework.project_extractor.litellm.completion")
    def test_passes_existing_items(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }
        existing = [{"dimension": "blocker", "text": "Qdrant 单进程锁", "importance": 8}]
        synthesize_project_dims(memories=["m"], existing_items=existing, model="test/model")
        user_content = mock_completion.call_args[1]["messages"][1]["content"]
        assert "Qdrant 单进程锁" in user_content
        assert "现有项目记录" in user_content
