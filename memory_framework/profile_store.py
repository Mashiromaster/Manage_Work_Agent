"""画像持久化与增量更新。

画像存 JSON(profiles/<user_id>.json)。ingest 对新记忆分类、查重加频次、
打分、写盘;get_profile 读盘并按 survival 重算遗忘标记与排序。
"""

import json
import os
from datetime import datetime

from memory_framework.profile import ProfileItem, UserProfile
from memory_framework.scoring import (
    FORGET_THRESHOLD, classify, survival_score, weighted_importance,
)


class ProfileStore:
    def __init__(self, base_dir: str = "./profiles") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe = user_id.replace("/", "_")
        return os.path.join(self.base_dir, f"{safe}.json")

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(text.split()).strip("。,.!?;:")

    def _load(self, user_id: str) -> UserProfile:
        path = self._path(user_id)
        if not os.path.exists(path):
            return UserProfile(user_id=user_id, items=[])
        with open(path, encoding="utf-8") as f:
            return UserProfile.from_dict(json.load(f))

    def _save(self, profile: UserProfile) -> None:
        with open(self._path(profile.user_id), "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

    def delete_item(self, user_id: str, dimension: str, text: str) -> bool:
        """删除某用户画像里 (dimension, text) 匹配的条目;返回是否删掉了。

        画像条目无独立 id,按 (维度, 文本) 定位。删完即写盘。找不到返回 False。
        """
        profile = self._load(user_id)
        before = len(profile.items)
        profile.items = [
            it for it in profile.items
            if not (it.dimension == dimension and it.text == text)
        ]
        if len(profile.items) == before:
            return False
        self._save(profile)
        return True

    def replace_profile(self, user_id: str,
                        items_data: list[dict],
                        now: datetime = None) -> UserProfile:
        """全量替换用户画像。

        Args:
            user_id: 用户标识。
            items_data: 结构化画像条目列表,每项含 dimension/text/importance/evidence。
            now: 当前时间,用于时间戳。

        Returns:
            替换后的 UserProfile。
        """
        now = now or datetime.now()
        items = []
        for d in items_data:
            items.append(ProfileItem(
                text=d["text"],
                dimension=d["dimension"],
                importance=float(d.get("importance", 5)),
                created_at=d.get("created_at", now.isoformat()),
                last_seen=now.isoformat(),
                mention_count=int(d.get("mention_count", 1)),
                evidence=str(d.get("evidence", "")),
                locked=bool(d.get("locked", False)),
                source=str(d.get("source", "llm")),
            ))
        profile = UserProfile(user_id=user_id, items=items)
        self._save(profile)
        return profile

    def ingest(self, user_id: str, memory_texts: list, now: datetime = None) -> UserProfile:
        now = now or datetime.now()
        profile = self._load(user_id)
        for text in memory_texts:
            if not text or not text.strip():
                continue
            dim = classify(text)
            norm = self._normalize(text)
            match = None
            for item in profile.items:
                if item.dimension == dim and (
                    norm in self._normalize(item.text)
                    or self._normalize(item.text) in norm
                ):
                    match = item
                    break
            if match:
                match.mention_count += 1
                match.last_seen = now.isoformat()
                match.importance = weighted_importance(dim, match.mention_count)
            else:
                profile.items.append(ProfileItem(
                    text=text.strip(), dimension=dim,
                    importance=weighted_importance(dim, 1),
                    created_at=now.isoformat(), last_seen=now.isoformat(),
                    mention_count=1,
                ))
        self._save(profile)
        return profile

    def get_profile(self, user_id: str, now: datetime = None,
                    include_forgotten: bool = False) -> UserProfile:
        now = now or datetime.now()
        profile = self._load(user_id)
        for item in profile.items:
            surv = survival_score(item.importance, item.last_seen, item.dimension, now)
            item.forgotten = surv < FORGET_THRESHOLD
        profile.items.sort(
            key=lambda i: survival_score(i.importance, i.last_seen, i.dimension, now),
            reverse=True,
        )
        if not include_forgotten:
            profile.items = [i for i in profile.items if not i.forgotten]
        return profile


class ProjectStore(ProfileStore):
    """项目记录存储:复用 ProfileStore 的落盘与替换逻辑,但**不做遗忘**。

    项目的进度 / 待办 / 决策不应随时间自然消退(scoring 的 90 天默认半衰期
    对这些维度没有意义),故读取时按重要性排序,保留全部条目,不计算 survival。
    默认写到独立目录 ``./project_tracking``,与用户画像互不干扰。
    """

    def __init__(self, base_dir: str = "./project_tracking") -> None:
        super().__init__(base_dir=base_dir)

    def get_project(self, project_id: str) -> UserProfile:
        """读取项目全部条目,按重要性降序,不遗忘、不过滤。"""
        profile = self._load(project_id)
        for item in profile.items:
            item.forgotten = False
        profile.items.sort(key=lambda i: i.importance, reverse=True)
        return profile

    def set_locked(self, project_id: str, dimension: str, text: str,
                   locked: bool) -> bool:
        """按 (dimension, text) 定位条目并置锁定态;找不到返回 False。"""
        profile = self._load(project_id)
        hit = False
        for it in profile.items:
            if it.dimension == dimension and it.text == text:
                it.locked = bool(locked)
                hit = True
        if hit:
            self._save(profile)
        return hit

    def add_item(self, project_id: str, dimension: str, text: str,
                 importance: float = 6):
        """手动添加一条 (source=manual, locked=True) 的条目并写盘。

        text 去空白后为空则忽略;同 (dimension, text) 已存在则就地标记为
        manual+locked(不重复追加)。返回写盘后的 UserProfile。
        """
        clean = (text or "").strip()
        profile = self._load(project_id)
        if not clean:
            return profile
        for it in profile.items:
            if it.dimension == dimension and it.text == clean:
                it.source = "manual"
                it.locked = True
                self._save(profile)
                return profile
        now = datetime.now().isoformat()
        profile.items.append(ProfileItem(
            text=clean, dimension=dimension, importance=float(importance),
            created_at=now, last_seen=now, mention_count=1,
            source="manual", locked=True))
        self._save(profile)
        return profile
