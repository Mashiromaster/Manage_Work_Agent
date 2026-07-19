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
