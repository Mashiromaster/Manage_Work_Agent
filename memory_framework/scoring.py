"""重要性打分与时间衰减(纯函数,不调 LLM,不隐式取当前时间)。

.. deprecated::
    ``classify()`` 已被 LLM 提炼替代,保留仅为向后兼容。
    ``survival_score()`` / ``weighted_importance()`` / ``base_score()`` / ``FORGET_THRESHOLD``
    仍用于画像读取时的遗忘计算和排序。
"""

import math
from datetime import datetime

from memory_framework.profile import EVENT, INTEREST, PERSONALITY, TABOO

_TABOO_KW = ["不吃", "过敏", "讨厌", "不喜欢", "忌", "反感", "受不了", "厌恶"]
_EVENT_KW = ["报名", "去了", "去过", "遇到", "昨天", "上周", "上个月", "计划", "打算",
             "买了", "入手", "参加", "完成了", "开始了"]
_PERSONALITY_KW = ["性格", "习惯", "内向", "外向", "自律", "急性子", "慢性子",
                   "喜欢独处", "喜欢一个人", "追求完美", "乐观", "悲观"]

_BASE = {TABOO: 9.0, PERSONALITY: 7.0, INTEREST: 6.0, EVENT: 5.0}
HALFLIFE_DAYS = {TABOO: 365.0, PERSONALITY: 180.0, INTEREST: 90.0, EVENT: 30.0}
FORGET_THRESHOLD = 1.0


def classify(text: str) -> str:
    if any(kw in text for kw in _TABOO_KW):
        return TABOO
    if any(kw in text for kw in _EVENT_KW):
        return EVENT
    if any(kw in text for kw in _PERSONALITY_KW):
        return PERSONALITY
    return INTEREST


def base_score(dimension: str) -> float:
    return _BASE.get(dimension, 5.0)


def weighted_importance(dimension: str, mention_count: int) -> float:
    bonus = math.log2(mention_count) if mention_count > 0 else 0.0
    return min(10.0, base_score(dimension) + bonus)


def survival_score(importance: float, last_seen: str, dimension: str,
                   now: datetime) -> float:
    last = datetime.fromisoformat(last_seen)
    delta_days = max(0.0, (now - last).total_seconds() / 86400.0)
    halflife = HALFLIFE_DAYS.get(dimension, 90.0)
    return importance * (0.5 ** (delta_days / halflife))
