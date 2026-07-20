from datetime import datetime, timedelta

import pytest

from memory_framework.profile import TABOO
from memory_framework.profile_store import ProfileStore


@pytest.fixture
def store(tmp_path):
    return ProfileStore(base_dir=str(tmp_path / "profiles"))


def test_ingest_creates_items(store):
    now = datetime(2026, 7, 4)
    p = store.ingest("u1", ["我不吃香菜", "我喜欢爵士乐"], now=now)
    texts = [i.text for i in p.items]
    assert "我不吃香菜" in texts and "我喜欢爵士乐" in texts
    taboo = [i for i in p.items if i.dimension == TABOO][0]
    assert taboo.importance >= 9.0


def test_ingest_dedup_increments_mention(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["我喜欢爵士乐"], now=now)
    p = store.ingest("u1", ["我喜欢爵士乐"], now=now + timedelta(days=1))
    jazz = [i for i in p.items if "爵士" in i.text]
    assert len(jazz) == 1
    assert jazz[0].mention_count == 2


def test_get_profile_marks_forgotten(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["昨天我随便逛了逛"], now=now - timedelta(days=120))
    p = store.get_profile("u1", now=now, include_forgotten=True)
    assert any(i.forgotten for i in p.items)


def test_get_profile_hides_forgotten_by_default(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["昨天我随便逛了逛"], now=now - timedelta(days=120))
    visible = store.get_profile("u1", now=now).items
    assert all(not i.forgotten for i in visible)


def test_get_profile_sorted_by_survival(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["我不吃香菜", "我喜欢爵士乐"], now=now)
    items = store.get_profile("u1", now=now).items
    assert items == sorted(items, key=lambda i: i.importance, reverse=True) or len(items) <= 1


def test_delete_item_removes_matching(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["我不吃香菜", "我喜欢爵士乐"], now=now)
    item = store.get_profile("u1", now=now).items[0]
    ok = store.delete_item("u1", item.dimension, item.text)
    assert ok is True
    remaining = [i.text for i in store.get_profile("u1", now=now).items]
    assert item.text not in remaining


def test_delete_item_not_found(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["我喜欢爵士乐"], now=now)
    assert store.delete_item("u1", "interest", "不存在的条目") is False
    # 原条目仍在
    assert any("爵士" in i.text for i in store.get_profile("u1", now=now).items)


def test_replace_profile_preserves_locked_and_source(store):
    p = store.replace_profile("u1", [
        {"dimension": "progress", "text": "锁定项", "importance": 7,
         "locked": True, "source": "manual"},
        {"dimension": "todo", "text": "普通项", "importance": 5},
    ])
    by_text = {i.text: i for i in p.items}
    assert by_text["锁定项"].locked is True
    assert by_text["锁定项"].source == "manual"
    assert by_text["普通项"].locked is False
    assert by_text["普通项"].source == "llm"


from memory_framework.profile_store import ProjectStore


@pytest.fixture
def pstore(tmp_path):
    return ProjectStore(base_dir=str(tmp_path / "pt"))


def test_add_item_appends_manual_locked(pstore):
    p = pstore.add_item("proj", "todo", "  手动待办  ", importance=8)
    it = [i for i in p.items if i.text == "手动待办"][0]
    assert it.source == "manual" and it.locked is True and it.importance == 8


def test_add_item_blank_ignored(pstore):
    p = pstore.add_item("proj", "todo", "   ")
    assert p.items == []


def test_add_item_existing_becomes_manual_locked(pstore):
    pstore.replace_profile("proj", [
        {"dimension": "todo", "text": "x", "importance": 5}])
    p = pstore.add_item("proj", "todo", "x")
    xs = [i for i in p.items if i.text == "x"]
    assert len(xs) == 1 and xs[0].locked is True and xs[0].source == "manual"


def test_set_locked_toggles(pstore):
    pstore.replace_profile("proj", [
        {"dimension": "progress", "text": "a", "importance": 5}])
    assert pstore.set_locked("proj", "progress", "a", True) is True
    assert pstore.get_project("proj").items[0].locked is True
    assert pstore.set_locked("proj", "progress", "a", False) is True
    assert pstore.get_project("proj").items[0].locked is False


def test_set_locked_missing_returns_false(pstore):
    assert pstore.set_locked("proj", "progress", "nope", True) is False


def test_delete_item_removes(pstore):
    pstore.replace_profile("proj", [
        {"dimension": "todo", "text": "d", "importance": 5}])
    assert pstore.delete_item("proj", "todo", "d") is True
    assert pstore.get_project("proj").items == []
