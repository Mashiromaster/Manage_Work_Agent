from memory_framework.profile import (
    DIMENSION_LABELS, DIMENSIONS, ProfileItem, TABOO, UserProfile,
)


def test_dimensions_has_four():
    assert len(DIMENSIONS) == 4
    assert TABOO in DIMENSIONS
    assert all(d in DIMENSION_LABELS for d in DIMENSIONS)


def test_profile_item_roundtrip():
    item = ProfileItem(text="不吃香菜", dimension=TABOO, importance=9.0,
                       created_at="2026-07-04T10:00:00", last_seen="2026-07-04T10:00:00",
                       evidence="用户明确说我不吃香菜")
    d = item.to_dict()
    back = ProfileItem.from_dict(d)
    assert back.text == "不吃香菜"
    assert back.dimension == TABOO
    assert back.importance == 9.0
    assert back.mention_count == 1
    assert back.forgotten is False
    assert back.evidence == "用户明确说我不吃香菜"


def test_user_profile_by_dimension_groups():
    p = UserProfile(user_id="u", items=[
        ProfileItem("不吃香菜", TABOO, 9.0, "2026-07-04T10:00:00", "2026-07-04T10:00:00"),
    ])
    grouped = p.by_dimension()
    assert grouped[TABOO][0].text == "不吃香菜"
    assert all(dim in grouped for dim in DIMENSIONS)


def test_user_profile_roundtrip():
    p = UserProfile(user_id="u", items=[
        ProfileItem("喜欢爵士", "interest", 6.0, "2026-07-04T10:00:00", "2026-07-04T10:00:00"),
    ])
    back = UserProfile.from_dict(p.to_dict())
    assert back.user_id == "u"
    assert back.items[0].text == "喜欢爵士"


def test_defaults_locked_false_source_llm():
    it = ProfileItem(text="t", dimension="progress", importance=5,
                     created_at="2026-07-20", last_seen="2026-07-20")
    assert it.locked is False
    assert it.source == "llm"


def test_to_dict_roundtrip_preserves_new_fields():
    it = ProfileItem(text="t", dimension="progress", importance=5,
                     created_at="c", last_seen="l", locked=True, source="manual")
    d = it.to_dict()
    assert d["locked"] is True and d["source"] == "manual"
    back = ProfileItem.from_dict(d)
    assert back.locked is True and back.source == "manual"


def test_from_dict_legacy_without_new_fields():
    d = {"text": "t", "dimension": "progress", "importance": 5,
         "created_at": "c", "last_seen": "l"}
    it = ProfileItem.from_dict(d)
    assert it.locked is False and it.source == "llm"
