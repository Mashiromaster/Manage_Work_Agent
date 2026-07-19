"""画像提炼模块测试。"""

import json
from unittest.mock import patch

from memory_framework.profile_extractor import (
    _parse_json_response,
    _validate_items,
    synthesize_profile,
)


class TestParseJsonResponse:
    def test_direct_json_array(self):
        raw = json.dumps([
            {"dimension": "personality", "text": "坚毅", "importance": 8, "evidence": "跑完全马"}
        ])
        items = _parse_json_response(raw)
        assert len(items) == 1
        assert items[0]["text"] == "坚毅"

    def test_markdown_code_block(self):
        raw = '```json\n[{"dimension": "taboo", "text": "不吃香菜", "importance": 9, "evidence": "我不吃香菜"}]\n```'
        items = _parse_json_response(raw)
        assert len(items) == 1
        assert items[0]["dimension"] == "taboo"

    def test_json_in_text(self):
        raw = '好的,以下是用户画像:\n[{"dimension": "interest", "text": "热爱音乐", "importance": 6, "evidence": "每天练琴"}]\n还有其他需要吗?'
        items = _parse_json_response(raw)
        assert len(items) == 1
        assert items[0]["text"] == "热爱音乐"

    def test_invalid_json_returns_empty(self):
        assert _parse_json_response("这不是 JSON") == []
        assert _parse_json_response("") == []

    def test_nested_array_extraction(self):
        raw = 'prefix [{"dimension": "personality", "text": "自律", "importance": 7}] suffix'
        items = _parse_json_response(raw)
        assert len(items) == 1
        assert items[0]["text"] == "自律"


class TestValidateItems:
    def test_valid_items_pass(self):
        items = [
            {"dimension": "personality", "text": "坚毅", "importance": 8, "evidence": "跑完全马"},
            {"dimension": "taboo", "text": "海鲜过敏", "importance": 9, "evidence": "对海鲜过敏"},
        ]
        result = _validate_items(items)
        assert len(result) == 2

    def test_invalid_dimension_filtered(self):
        items = [
            {"dimension": "personality", "text": "坚毅", "importance": 8},
            {"dimension": "INVALID", "text": "某特征", "importance": 5},
        ]
        result = _validate_items(items)
        assert len(result) == 1
        assert result[0]["dimension"] == "personality"

    def test_empty_text_filtered(self):
        items = [
            {"dimension": "personality", "text": "", "importance": 5},
            {"dimension": "interest", "text": "音乐", "importance": 6},
        ]
        result = _validate_items(items)
        assert len(result) == 1

    def test_non_dict_filtered(self):
        items = [
            "not a dict",
            {"dimension": "taboo", "text": "过敏", "importance": 9},
        ]
        result = _validate_items(items)
        assert len(result) == 1

    def test_missing_importance_defaults(self):
        items = [{"dimension": "event", "text": "离职创业"}]
        result = _validate_items(items)
        assert result[0]["importance"] == 5

    def test_missing_evidence_defaults(self):
        items = [{"dimension": "personality", "text": "乐观"}]
        result = _validate_items(items)
        assert result[0]["evidence"] == ""


class TestSynthesizeProfile:
    def test_empty_memories_returns_empty(self):
        items = synthesize_profile([], model="test/model")
        assert items == []

    @patch("memory_framework.profile_extractor.litellm.completion")
    def test_returns_parsed_items(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"dimension": "personality", "text": "坚毅自律", "importance": 8, "evidence": "坚持长跑训练半年并完赛全马"},
                {"dimension": "taboo", "text": "不吃香菜,海鲜过敏", "importance": 9, "evidence": "我不吃香菜,对海鲜过敏"},
            ])}}]
        }
        items = synthesize_profile(
            memories=["用户不吃香菜,对海鲜过敏", "用户坚持长跑训练半年并完赛全马"],
            model="test/model",
        )
        assert len(items) == 2
        assert items[0]["dimension"] == "personality"
        assert items[0]["text"] == "坚毅自律"
        assert "全马" in items[0]["evidence"]
        assert items[1]["dimension"] == "taboo"
        assert "香菜" in items[1]["text"]

    @patch("memory_framework.profile_extractor.litellm.completion")
    def test_llm_exception_returns_empty(self, mock_completion):
        mock_completion.side_effect = RuntimeError("API 不可用")
        items = synthesize_profile(
            memories=["用户喜欢爵士乐"],
            model="test/model",
        )
        assert items == []

    @patch("memory_framework.profile_extractor.litellm.completion")
    def test_passes_existing_items_to_llm(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"dimension": "personality", "text": "坚毅", "importance": 8, "evidence": "跑马"}
            ])}}]
        }
        existing = [{"dimension": "interest", "text": "热爱烹饪", "importance": 6, "evidence": "学做菜"}]
        synthesize_profile(
            memories=["用户坚持跑步"],
            existing_items=existing,
            model="test/model",
        )
        # 验证 LLM 收到的消息中包含现有画像
        call_args = mock_completion.call_args[1]
        messages = call_args["messages"]
        user_content = messages[1]["content"]
        assert "热爱烹饪" in user_content
        assert "现有画像" in user_content
