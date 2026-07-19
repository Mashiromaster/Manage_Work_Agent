from datetime import datetime, timedelta

from memory_framework.profile import EVENT, INTEREST, PERSONALITY, TABOO
from memory_framework.scoring import (
    FORGET_THRESHOLD, HALFLIFE_DAYS, base_score, classify, survival_score,
    weighted_importance,
)


def test_classify_taboo():
    assert classify("我不吃香菜,对海鲜过敏") == TABOO
    assert classify("我特别讨厌开长会") == TABOO


def test_classify_event():
    assert classify("我报名了今年秋天的马拉松") == EVENT
    assert classify("昨天线上遇到一个诡异的 bug") == EVENT


def test_classify_personality():
    assert classify("我性格比较内向,喜欢独处") == PERSONALITY


def test_classify_interest_default():
    assert classify("我喜欢爵士乐") == INTEREST


def test_base_scores_ordered():
    assert base_score(TABOO) > base_score(PERSONALITY) > base_score(INTEREST) > base_score(EVENT)


def test_weighted_importance_grows_with_mentions():
    once = weighted_importance(INTEREST, 1)
    thrice = weighted_importance(INTEREST, 4)
    assert thrice > once
    assert weighted_importance(TABOO, 100) <= 10.0


def test_survival_decays_over_time():
    now = datetime(2026, 7, 4)
    fresh = survival_score(9.0, now.isoformat(), TABOO, now)
    old_event = survival_score(5.0, (now - timedelta(days=60)).isoformat(), EVENT, now)
    assert fresh > old_event


def test_taboo_decays_slower_than_event():
    now = datetime(2026, 7, 4)
    dt = (now - timedelta(days=60)).isoformat()
    assert survival_score(9.0, dt, TABOO, now) > survival_score(9.0, dt, EVENT, now)


def test_halflife_ordered():
    assert HALFLIFE_DAYS[TABOO] > HALFLIFE_DAYS[PERSONALITY] > HALFLIFE_DAYS[INTEREST] > HALFLIFE_DAYS[EVENT]
    assert FORGET_THRESHOLD == 1.0
