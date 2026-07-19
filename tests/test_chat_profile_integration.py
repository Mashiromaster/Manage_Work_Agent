from datetime import datetime
from unittest.mock import patch

from memory_framework.chat import update_profile_from_memories


class FakeStore:
    def get_all(self, user_id):
        return [{"memory": "用户不吃香菜"}, {"memory": "用户喜欢爵士乐"}]


def test_update_profile_from_memories_uses_llm_synthesis(tmp_path):
    from memory_framework.profile_store import ProfileStore

    ps = ProfileStore(base_dir=str(tmp_path / "profiles"))
    mock_items = [
        {"dimension": "taboo", "text": "不吃香菜", "importance": 9,
         "evidence": "用户不吃香菜"},
        {"dimension": "interest", "text": "热爱爵士乐,有音乐品味", "importance": 6,
         "evidence": "用户喜欢爵士乐"},
    ]
    with patch("memory_framework.chat.synthesize_profile", return_value=mock_items):
        prof = update_profile_from_memories(
            ps, FakeStore(), "u1", model="test/model",
            now=datetime(2026, 7, 4),
        )
    texts = [i.text for i in prof.items]
    assert any("香菜" in t for t in texts)
    assert any("爵士" in t for t in texts)
    # 画像应包含 LLM 提炼后的特征描述(evidence 字段)
    evidence_texts = [i.evidence for i in prof.items if i.evidence]
    assert len(evidence_texts) >= 1


def test_update_profile_empty_memories_returns_empty_profile(tmp_path):
    from memory_framework.profile_store import ProfileStore

    ps = ProfileStore(base_dir=str(tmp_path / "profiles"))

    class EmptyStore:
        def get_all(self, user_id):
            return []

    with patch("memory_framework.chat.synthesize_profile") as mock_synth:
        prof = update_profile_from_memories(
            ps, EmptyStore(), "u1", model="test/model",
            now=datetime(2026, 7, 4),
        )
    # 无记忆时不应调用 LLM
    mock_synth.assert_not_called()
    assert prof.items == []


def test_update_profile_fallback_on_llm_failure(tmp_path):
    """LLM 提炼失败时回退到旧画像,不丢数据。"""
    from memory_framework.profile_store import ProfileStore, ProfileItem, UserProfile

    ps = ProfileStore(base_dir=str(tmp_path / "profiles"))
    # 先存入一条旧画像
    ps._save(UserProfile(user_id="u1", items=[
        ProfileItem(text="旧特征", dimension="personality", importance=5,
                    created_at="2026-07-04T10:00:00", last_seen="2026-07-04T10:00:00"),
    ]))

    with patch("memory_framework.chat.synthesize_profile", return_value=[]):
        prof = update_profile_from_memories(
            ps, FakeStore(), "u1", model="test/model",
            now=datetime(2026, 7, 4),
        )
    # 回退:返回旧画像
    assert any("旧特征" in i.text for i in prof.items)
