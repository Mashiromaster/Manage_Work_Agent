# 结构化用户画像 + 重要性评估 + 时间衰减遗忘 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 mem0 向量记忆层之上叠加结构化用户画像层,支持四维度画像、规则+频次重要性评分、重要性×时间衰减遗忘,每轮对话实时增量更新。

**Architecture:** 新增三个纯 Python 模块(profile/scoring/profile_store),画像是向量记忆的衍生视图存为 JSON。打分与衰减为纯函数可单测。chat.reply 每轮存记忆后增量更新画像;Gradio 右侧展示分维度画像。

**Tech Stack:** Python 3.11 (conda env `mem0`),现有 mem0ai/memory_framework,pytest。无新依赖。

## Global Constraints

- Python 环境:conda env **`mem0`**,命令用 `conda run -n mem0` 或先 `conda activate mem0`。
- 工作目录 `/Users/mashiro/Mem0`,入口脚本运行需 `PYTHONPATH=.`。
- 时间统一用**注入的 `now: datetime` 参数**(默认 `datetime.now()`),便于测试;纯函数不在内部隐式取当前时间。
- 四维度常量名:`interest` / `personality` / `event` / `taboo`。
- 不额外调 LLM,不新增依赖,不改动现有 MemoryStore/config 的公有接口。
- 每个 Task 结束提交一次 git commit。

---

### Task 1: 画像数据结构 `profile.py`

**Files:**
- Create: `memory_framework/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Produces:
  - 维度常量 `INTEREST="interest"`, `PERSONALITY="personality"`, `EVENT="event"`, `TABOO="taboo"`,`DIMENSIONS=[...4个...]`,`DIMENSION_LABELS={dim: 中文名}`。
  - `@dataclass ProfileItem`:`text:str, dimension:str, importance:float, created_at:str(ISO), last_seen:str(ISO), mention_count:int=1, forgotten:bool=False`;方法 `to_dict()->dict` / `from_dict(d)->ProfileItem`(classmethod)。
  - `@dataclass UserProfile`:`user_id:str, items:list[ProfileItem]`;`to_dict()` / `from_dict(d)`;`by_dimension()->dict[str,list[ProfileItem]]`。

- [ ] **Step 1: 写失败测试 `tests/test_profile.py`**

```python
from memory_framework.profile import (
    ProfileItem, UserProfile, DIMENSIONS, DIMENSION_LABELS, TABOO,
)


def test_dimensions_has_four():
    assert len(DIMENSIONS) == 4
    assert TABOO in DIMENSIONS
    assert all(d in DIMENSION_LABELS for d in DIMENSIONS)


def test_profile_item_roundtrip():
    item = ProfileItem(text="不吃香菜", dimension=TABOO, importance=9.0,
                       created_at="2026-07-04T10:00:00", last_seen="2026-07-04T10:00:00")
    d = item.to_dict()
    back = ProfileItem.from_dict(d)
    assert back.text == "不吃香菜"
    assert back.dimension == TABOO
    assert back.importance == 9.0
    assert back.mention_count == 1
    assert back.forgotten is False


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
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n mem0 python -m pytest tests/test_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: memory_framework.profile`

- [ ] **Step 3: 写实现 `memory_framework/profile.py`**

```python
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

    def to_dict(self) -> dict:
        return {
            "text": self.text, "dimension": self.dimension,
            "importance": self.importance, "created_at": self.created_at,
            "last_seen": self.last_seen, "mention_count": self.mention_count,
            "forgotten": self.forgotten,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileItem":
        return cls(
            text=d["text"], dimension=d["dimension"], importance=d["importance"],
            created_at=d["created_at"], last_seen=d["last_seen"],
            mention_count=d.get("mention_count", 1), forgotten=d.get("forgotten", False),
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
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n mem0 python -m pytest tests/test_profile.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add memory_framework/profile.py tests/test_profile.py
git commit -m "feat: 画像数据结构 ProfileItem/UserProfile + 四维度"
```

---

### Task 2: 打分与衰减纯函数 `scoring.py`

**Files:**
- Create: `memory_framework/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `memory_framework.profile`(维度常量)。
- Produces:
  - `classify(text: str) -> str`:返回四维度之一。
  - `base_score(dimension: str) -> float`:taboo=9, personality=7, interest=6, event=5。
  - `weighted_importance(dimension: str, mention_count: int) -> float`:`min(10, base + log2(mention_count))`。
  - `HALFLIFE_DAYS: dict[str,float]`:taboo=365, personality=180, interest=90, event=30。
  - `survival_score(importance: float, last_seen: str, dimension: str, now: datetime) -> float`:`importance * 0.5 ** (Δdays / halflife)`。
  - `FORGET_THRESHOLD: float = 1.0`。

- [ ] **Step 1: 写失败测试 `tests/test_scoring.py`**

```python
from datetime import datetime, timedelta

from memory_framework.profile import TABOO, EVENT, INTEREST, PERSONALITY
from memory_framework.scoring import (
    classify, base_score, weighted_importance, survival_score,
    HALFLIFE_DAYS, FORGET_THRESHOLD,
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
    assert fresh > old_event  # 事件衰减快


def test_taboo_decays_slower_than_event():
    now = datetime(2026, 7, 4)
    dt = (now - timedelta(days=60)).isoformat()
    assert survival_score(9.0, dt, TABOO, now) > survival_score(9.0, dt, EVENT, now)
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n mem0 python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: memory_framework.scoring`

- [ ] **Step 3: 写实现 `memory_framework/scoring.py`**

```python
"""重要性打分与时间衰减(纯函数,不调 LLM,不隐式取当前时间)。"""

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
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n mem0 python -m pytest tests/test_scoring.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add memory_framework/scoring.py tests/test_scoring.py
git commit -m "feat: 规则分类+频次重要性+时间衰减纯函数"
```

---

### Task 3: 画像存储与增量更新 `profile_store.py`

**Files:**
- Create: `memory_framework/profile_store.py`
- Test: `tests/test_profile_store.py`

**Interfaces:**
- Consumes: `profile`(ProfileItem/UserProfile),`scoring`(classify/weighted_importance/survival_score/FORGET_THRESHOLD)。
- Produces:
  - `ProfileStore(base_dir="./profiles")`:构造时确保目录存在。
  - `ingest(user_id: str, memory_texts: list[str], now: datetime=None) -> UserProfile`:对每条文本分类→查重(归一化包含判定)→命中则 mention_count+1 且刷新 last_seen、否则新建 ProfileItem→重算 importance→存盘→返回更新后 profile。
  - `get_profile(user_id: str, now: datetime=None, include_forgotten: bool=False) -> UserProfile`:读盘→按 survival 重算 forgotten 标记→按 survival 降序排序 items→按 include_forgotten 过滤。
  - `_path(user_id)` / 内部 `_normalize(text)`。

- [ ] **Step 1: 写失败测试 `tests/test_profile_store.py`**

```python
from datetime import datetime, timedelta

import pytest

from memory_framework.profile import TABOO, EVENT
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
    assert jazz[0].mention_count == 2  # 去重+频次


def test_get_profile_marks_forgotten(store):
    now = datetime(2026, 7, 4)
    store.ingest("u1", ["昨天我随便逛了逛"], now=now - timedelta(days=120))
    p = store.get_profile("u1", now=now, include_forgotten=True)
    # 低基础分(event)+ 120天 → 存活分应低于阈值被标记遗忘
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
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n mem0 python -m pytest tests/test_profile_store.py -v`
Expected: FAIL — `ModuleNotFoundError: memory_framework.profile_store`

- [ ] **Step 3: 写实现 `memory_framework/profile_store.py`**

```python
"""画像持久化与增量更新。

画像存 JSON(profiles/<user_id>.json)。ingest 对新记忆分类、查重加频次、
打分、写盘;get_profile 读盘并按 survival 重算遗忘标记与排序。
"""

import json
import os
from datetime import datetime

from memory_framework.profile import DIMENSIONS, ProfileItem, UserProfile
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
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n mem0 python -m pytest tests/test_profile_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add memory_framework/profile_store.py tests/test_profile_store.py
git commit -m "feat: ProfileStore 增量更新+去重频次+遗忘标记"
```

---

### Task 4: 集成到 `chat.py`(每轮实时增量更新画像)

**Files:**
- Modify: `memory_framework/chat.py`
- Test: `tests/test_chat_profile_integration.py`

**Interfaces:**
- Consumes: `ProfileStore.ingest`,`MemoryStore.get_all`。
- Produces:
  - `update_profile_from_memories(profile_store, store, user_id, now=None) -> UserProfile`:从 MemoryStore 拉该用户全部记忆文本,喂给 ProfileStore.ingest。供 reply 与 UI 复用。

- [ ] **Step 1: 写失败测试 `tests/test_chat_profile_integration.py`**

```python
from datetime import datetime

from memory_framework.chat import update_profile_from_memories


class FakeStore:
    def get_all(self, user_id):
        return [{"memory": "用户不吃香菜"}, {"memory": "用户喜欢爵士乐"}]


def test_update_profile_from_memories(tmp_path):
    from memory_framework.profile_store import ProfileStore
    ps = ProfileStore(base_dir=str(tmp_path / "profiles"))
    prof = update_profile_from_memories(ps, FakeStore(), "u1", now=datetime(2026, 7, 4))
    texts = [i.text for i in prof.items]
    assert any("香菜" in t for t in texts)
    assert any("爵士" in t for t in texts)
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n mem0 python -m pytest tests/test_chat_profile_integration.py -v`
Expected: FAIL — `ImportError: cannot import name 'update_profile_from_memories'`

- [ ] **Step 3: 修改 `memory_framework/chat.py`**

在文件顶部 import 区加入:
```python
from memory_framework.profile_store import ProfileStore
```

在 `reply` 函数**之前**新增:
```python
def update_profile_from_memories(profile_store, store, user_id, now=None):
    """从 MemoryStore 拉取该用户全部记忆文本,增量更新画像。返回更新后 UserProfile。"""
    mems = store.get_all(user_id=user_id)
    texts = [m.get("memory", m) if isinstance(m, dict) else m for m in mems]
    return profile_store.ingest(user_id, [t for t in texts if isinstance(t, str)], now=now)
```

在 `reply` 函数末尾 `return answer` **之前**加入(使聊天每轮更新画像):
```python
    update_profile_from_memories(ProfileStore(), store, user_id)
```
即 reply 结尾变为:
```python
    update_profile_from_memories(ProfileStore(), store, user_id)
    return answer
```

- [ ] **Step 4: 运行确认通过 + 回归**

Run: `conda run -n mem0 python -m pytest tests/test_chat_profile_integration.py tests/test_profile.py tests/test_scoring.py tests/test_profile_store.py -v`
Expected: 全部 passed(集成 1 + profile 4 + scoring 8 + store 5 = 18)

- [ ] **Step 5: Commit**

```bash
git add memory_framework/chat.py tests/test_chat_profile_integration.py
git commit -m "feat: chat 每轮实时增量更新用户画像"
```

---

### Task 5: Gradio 展示结构化画像

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `ProfileStore.get_profile`,`profile.DIMENSION_LABELS`,`chat.update_profile_from_memories`。
- Produces: UI 无新导出接口,只新增一个画像 Markdown 面板与渲染函数 `_render_profile(user_id)`。

- [ ] **Step 1: 修改 `app.py` —— 顶部 import 与单例**

在 import 区加入:
```python
from memory_framework.profile import DIMENSION_LABELS
from memory_framework.profile_store import ProfileStore
from memory_framework.chat import update_profile_from_memories
```
在 `_STORE = None` 附近加:
```python
_PROFILE = None


def get_profile_store():
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = ProfileStore()
    return _PROFILE
```

- [ ] **Step 2: 新增画像渲染函数**(放在 `_render_memories` 之后)

```python
def _render_profile(user_id: str) -> str:
    if not user_id.strip():
        return "_(请先输入 user_id)_"
    prof = get_profile_store().get_profile(user_id.strip())
    grouped = prof.by_dimension()
    lines = [f"## 👤 {user_id} 的结构化画像\n"]
    for dim, label in DIMENSION_LABELS.items():
        items = grouped.get(dim, [])
        lines.append(f"**{label}**")
        if not items:
            lines.append("_(暂无)_")
        for it in items:
            lines.append(f"- {it.text}  `重要性 {it.importance:.1f} · 提及 {it.mention_count}`")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 3: 在 `on_send` 里更新画像并返回**

把 `on_send` 改为在存储后刷新画像,并多返回一个画像面板值。将其签名/返回改为:
```python
def on_send(message, history, user_id):
    if not message.strip() or not user_id.strip():
        return history, _render_memories(user_id), _render_profile(user_id), ""
    answer = reply(get_store(), user_id.strip(), message.strip(), MODEL)
    update_profile_from_memories(get_profile_store(), get_store(), user_id.strip())
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, _render_memories(user_id), _render_profile(user_id), ""
```

- [ ] **Step 4: 在 `on_load_testset` 里也构建画像**

在 `ingest(store, conv)` 之后、`return` 之前加入:
```python
    update_profile_from_memories(get_profile_store(), store, uid)
```
并把返回值改为多返回画像:
```python
    return _render_memories(uid), _render_profile(uid), uid, greeting
```

- [ ] **Step 5: 在 `build_ui` 里加画像面板并接线**

在右侧 Column 内 `mem_view` 之后加:
```python
                prof_view = gr.Markdown(_render_profile("demo_user"))
```
更新事件绑定的 outputs(把 `prof_view` 加入):
```python
        msg.submit(on_send, [msg, chatbot, user_id],
                   [chatbot, mem_view, prof_view, msg])
        load_btn.click(on_load_testset, [testset, user_id],
                       [mem_view, prof_view, user_id, chatbot])
        refresh.click(lambda uid: (_render_memories(uid), _render_profile(uid)),
                      user_id, [mem_view, prof_view])
        user_id.change(lambda uid: (_render_memories(uid), _render_profile(uid)),
                       user_id, [mem_view, prof_view])
```

- [ ] **Step 6: 启动验证**

Run: `cd /Users/mashiro/Mem0 && PYTHONPATH=. conda run -n mem0 python app.py`(后台),`curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860` → 期望 200,日志无 Traceback。

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: Gradio 展示四维度结构化画像(重要性+提及次数)"
```

---

## 并行开发映射

- Task 1(数据结构)是所有的前置。
- Task 2(scoring)只依赖 Task 1 的维度常量,可与 Task 1 尾部并行起草。
- Task 3(profile_store)依赖 1+2。
- Task 4(chat 集成)、Task 5(UI)依赖 1-3;4 与 5 可并行(改不同文件:chat.py vs app.py)。

推荐:Task1 → Task2 → Task3 → (Task4 ∥ Task5)。
