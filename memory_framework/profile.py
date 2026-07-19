"""用户画像数据结构。

画像是向量记忆的衍生视图:每条 ProfileItem 归属一个维度,带重要性与时间戳。
UserProfile 是某个 user 的全部画像条目容器。
"""

from dataclasses import dataclass, field

INTEREST = "interest"
PERSONALITY = "personality"
EVENT = "event"
TABOO = "taboo"
DIMENSIONS = [INTEREST, PERSONALITY, EVENT, TABOO]
DIMENSION_LABELS = {
    INTEREST: "兴趣爱好",
    PERSONALITY: "性格特征",
    EVENT: "事件经历",
    TABOO: "禁忌与反感",
}


@dataclass
class ProfileItem:
    text: str
    dimension: str
    importance: float
    created_at: str  # ISO8601
    last_seen: str   # ISO8601
    mention_count: int = 1
    forgotten: bool = False
    evidence: str = ""  # 支撑该画像条目的原始记忆引用

    def to_dict(self) -> dict:
        return {
            "text": self.text, "dimension": self.dimension,
            "importance": self.importance, "created_at": self.created_at,
            "last_seen": self.last_seen, "mention_count": self.mention_count,
            "forgotten": self.forgotten, "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileItem":
        return cls(
            text=d["text"], dimension=d["dimension"], importance=d["importance"],
            created_at=d["created_at"], last_seen=d["last_seen"],
            mention_count=d.get("mention_count", 1), forgotten=d.get("forgotten", False),
            evidence=d.get("evidence", ""),
        )


@dataclass
class UserProfile:
    user_id: str
    items: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "items": [i.to_dict() for i in self.items]}

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(user_id=d["user_id"],
                   items=[ProfileItem.from_dict(i) for i in d.get("items", [])])

    def by_dimension(self) -> dict:
        grouped = {dim: [] for dim in DIMENSIONS}
        for item in self.items:
            grouped.setdefault(item.dimension, []).append(item)
        return grouped
