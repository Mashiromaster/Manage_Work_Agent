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
