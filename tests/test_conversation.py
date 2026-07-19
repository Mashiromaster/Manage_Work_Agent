import json

import pytest

from memory_framework.conversation import InvalidConversationError, load_conversation


def test_load_valid(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps({"user_id": "u1", "messages": [{"role": "user", "content": "hi"}]}),
        encoding="utf-8",
    )
    conv = load_conversation(str(p))
    assert conv["user_id"] == "u1"
    assert conv["messages"][0]["content"] == "hi"


def test_load_missing_user_id_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"messages": []}), encoding="utf-8")
    with pytest.raises(InvalidConversationError):
        load_conversation(str(p))


def test_load_messages_not_list_raises(tmp_path):
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps({"user_id": "u", "messages": "nope"}), encoding="utf-8")
    with pytest.raises(InvalidConversationError):
        load_conversation(str(p))


def test_load_message_missing_content_raises(tmp_path):
    p = tmp_path / "bad3.json"
    p.write_text(
        json.dumps({"user_id": "u", "messages": [{"role": "user"}]}), encoding="utf-8"
    )
    with pytest.raises(InvalidConversationError):
        load_conversation(str(p))


def test_demo_datasets_are_valid():
    for name in ("alice", "bob"):
        conv = load_conversation(f"data/conversations/{name}.json")
        assert conv["user_id"] == name
        assert len(conv["messages"]) >= 1
