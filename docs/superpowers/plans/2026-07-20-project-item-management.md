# 项目条目管理(聚焦粒度 + 增删/锁定/手动添加)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让项目进度四维聚焦到功能级/关键决策(约 12–20 条),并给每条加锁定/删除、每维加手动添加,锁定与手动条目在重导入/全量重跑时不被 LLM 覆盖。

**Architecture:** 存储层给 `ProfileItem` 加 `locked`/`source` 两个向后兼容字段并新增 `set_locked`/`add_item`;编排层 `track_from_dialog` 在增量与全量两条路径都保留锁定+手动条目(`preserved + fresh`);提示词收敛粒度;UI 用 `gr.State` + `@gr.render` 把纯 HTML 面板重构成可交互的四维组件。

**Tech Stack:** Python 3.11(conda env `mem0`)、Gradio 6.20(`@gr.render` 动态渲染)、litellm、pytest。

## Global Constraints

- 运行测试:`LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/ -q --ignore=tests/test_memory_store.py`
- 运行 app:`LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. conda run --no-capture-output -n mem0 python -u app.py`
- git commit **绝不**加 `Co-Authored-By` trailer(用户明确要求)。
- Gradio 6:`theme`/`css`/`js` 传给 `launch()` 不传 `Blocks()`;不传 `js=`(会破坏事件绑定)。
- 条目逻辑主键 = `(dimension, text)`;不引入随机 id、不做迁移脚本(靠字段默认值向后兼容)。
- `ProfileItem` 新字段:`locked: bool = False`、`source: str = "llm"`(`"llm"` | `"manual"`)。
- Qdrant 本地文件单进程独占;`track_from_dialog` 不碰 Qdrant,子进程测试安全,但跑 app 与跑测试脚本不要同时占用。
- 多语句 `python -c` 在 conda 下易失败;需要脚本时写到 `$CLAUDE_JOB_DIR/tmp/*.py` 再运行。

---

### Task 1: ProfileItem 加 locked / source 字段

**Files:**
- Modify: `memory_framework/profile.py:22-48`(ProfileItem dataclass + to_dict/from_dict)
- Test: `tests/test_profile.py`(新建)

**Interfaces:**
- Produces: `ProfileItem(text, dimension, importance, created_at, last_seen, mention_count=1, forgotten=False, evidence="", locked=False, source="llm")`;`to_dict()` 含 `locked`/`source` 两个 key;`from_dict(d)` 用 `d.get("locked", False)` / `d.get("source", "llm")` 读,旧数据缺字段回落默认。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_profile.py`:

```python
from memory_framework.profile import ProfileItem


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
```

- [ ] **Step 2: 运行确认失败**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile.py -q`
Expected: FAIL(`TypeError: unexpected keyword argument 'locked'` 或 AttributeError)

- [ ] **Step 3: 实现**

`memory_framework/profile.py` 的 `ProfileItem` 加两字段并更新序列化:

```python
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
    locked: bool = False   # 用户锁定:重导入不覆盖
    source: str = "llm"    # "llm" 提炼产出 / "manual" 手动添加

    def to_dict(self) -> dict:
        return {
            "text": self.text, "dimension": self.dimension,
            "importance": self.importance, "created_at": self.created_at,
            "last_seen": self.last_seen, "mention_count": self.mention_count,
            "forgotten": self.forgotten, "evidence": self.evidence,
            "locked": self.locked, "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileItem":
        return cls(
            text=d["text"], dimension=d["dimension"], importance=d["importance"],
            created_at=d["created_at"], last_seen=d["last_seen"],
            mention_count=d.get("mention_count", 1), forgotten=d.get("forgotten", False),
            evidence=d.get("evidence", ""),
            locked=d.get("locked", False), source=d.get("source", "llm"),
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add memory_framework/profile.py tests/test_profile.py
git commit -m "feat(profile): ProfileItem 加 locked/source 字段(向后兼容)"
```

---

### Task 2: replace_profile 透传 locked/source

**Files:**
- Modify: `memory_framework/profile_store.py:57-84`(`ProfileStore.replace_profile`)
- Test: `tests/test_profile_store.py`(追加)

**Interfaces:**
- Consumes: Task 1 的 `ProfileItem(..., locked=, source=)`。
- Produces: `replace_profile(user_id, items_data, now=None)` 构造条目时读 `d.get("locked", False)` / `d.get("source", "llm")`,使写盘保留锁定态与来源。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_profile_store.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile_store.py::test_replace_profile_preserves_locked_and_source -q`
Expected: FAIL(`locked` 恒为默认 False,断言 True 失败)

- [ ] **Step 3: 实现**

`profile_store.py` 的 `replace_profile` 循环里加两字段(在 `evidence=` 后):

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile_store.py -q`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add memory_framework/profile_store.py tests/test_profile_store.py
git commit -m "feat(store): replace_profile 透传 locked/source"
```

---

### Task 3: ProjectStore 加 set_locked / add_item

**Files:**
- Modify: `memory_framework/profile_store.py`(在 `ProjectStore` 类里加两方法,`delete_item` 继承自基类已可用)
- Test: `tests/test_profile_store.py`(追加)

**Interfaces:**
- Consumes: Task 2 的 `replace_profile` 保留字段;基类已有 `delete_item(user_id, dimension, text) -> bool`、`_load`/`_save`。
- Produces:
  - `ProjectStore.set_locked(project_id, dimension, text, locked) -> bool`:按 `(dimension, text)` 找到条目置 `locked`,写盘;找不到返回 False。
  - `ProjectStore.add_item(project_id, dimension, text, importance=6) -> UserProfile`:追加 `source="manual", locked=True` 条目;text 去空白为空则忽略(返回现状);同 `(dimension,text)` 已存在则把它更新为 manual+locked(不重复追加)。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_profile_store.py`(顶部已 `from memory_framework.profile_store import ProfileStore`;补 import):

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile_store.py -q -k "add_item or set_locked or delete_item"`
Expected: FAIL(`AttributeError: 'ProjectStore' object has no attribute 'set_locked'`)

- [ ] **Step 3: 实现**

在 `profile_store.py` 的 `ProjectStore` 类里(`get_project` 之后)加。文件顶部已 `from datetime import datetime`:

```python
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
```

确认 `profile_store.py` 顶部已 `from memory_framework.profile import ProfileItem, UserProfile`(已有)。

- [ ] **Step 4: 运行确认通过**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile_store.py -q`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add memory_framework/profile_store.py tests/test_profile_store.py
git commit -m "feat(store): ProjectStore 加 set_locked/add_item"
```

---

### Task 4: track_from_dialog 保护锁定/手动条目(增量+全量)

**Files:**
- Modify: `memory_framework/project_tracker.py:187-224`(existing_items 读取 + 合并逻辑)
- Test: `tests/test_project_tracker.py`(追加)

**Interfaces:**
- Consumes: `ProfileItem.to_dict()` 含 `locked`/`source`;`extract_dims_from_messages(...)` 返回 fresh(list[dict],无 locked/source)。
- Produces: `track_from_dialog` 在增量与全量下 `items_data` 都 = `preserved + <未锁定部分>`,其中 `preserved` = 旧条目里 `locked or source=="manual"` 的;fresh 里与 preserved 同 `(dim, _norm(text))` 的被丢弃。`llm_empty` 语义不变(fresh 空时不动盘)。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_project_tracker.py`(文件已有 `_patches`、`FakeDialogStore`、`_Item`、`_norm` 场景):

```python
@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([{"role": "user", "content": "hi"}], {"s.jsonl": ["k"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_locked_item_survives_incremental(mock_ext, mock_collect):
    from memory_framework.project_tracker import track_from_dialog
    mock_ext.return_value = [{"dimension": "progress", "text": "新提炼", "importance": 7}]
    ps = FakeDialogStore(existing=[
        {"dimension": "todo", "text": "锁定待办", "importance": 5, "locked": True},
        {"dimension": "progress", "text": "旧未锁", "importance": 4, "locked": False},
    ])
    with _patches():
        track_from_dialog(ps, "X", "p", model="test/m", incremental=True)
    texts = {i["text"] for i in ps.replaced}
    assert "锁定待办" in texts       # 锁定保留
    assert "新提炼" in texts         # 新提炼进入
    assert "旧未锁" not in texts     # 旧未锁被替换掉


@patch("memory_framework.project_tracker.parse_cc_session",
       return_value=[{"role": "user", "content": "全文"}])
@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([], {"s.jsonl": ["k1"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_manual_and_locked_survive_full_rerun(mock_ext, mock_collect, mock_parse):
    from memory_framework.project_tracker import track_from_dialog
    mock_ext.return_value = [{"dimension": "progress", "text": "重提", "importance": 7}]
    ps = FakeDialogStore(existing=[
        {"dimension": "todo", "text": "手动项", "importance": 6, "source": "manual"},
        {"dimension": "progress", "text": "锁定项", "importance": 8, "locked": True},
        {"dimension": "blocker", "text": "旧未锁", "importance": 3},
    ])
    with _patches():
        track_from_dialog(ps, "X", "p", model="test/m", incremental=False)
    texts = {i["text"] for i in ps.replaced}
    assert "手动项" in texts and "锁定项" in texts   # 全量重跑也保留
    assert "重提" in texts                          # 新提炼进入
    assert "旧未锁" not in texts                     # 未锁定的旧 llm 被清


@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([{"role": "user", "content": "hi"}], {"s.jsonl": ["k"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_fresh_dropped_when_collides_with_locked(mock_ext, mock_collect):
    from memory_framework.project_tracker import track_from_dialog
    # LLM 又产出了和锁定条目同文本的项:应以锁定版本为准,不重复
    mock_ext.return_value = [{"dimension": "todo", "text": "锁定待办", "importance": 9}]
    ps = FakeDialogStore(existing=[
        {"dimension": "todo", "text": "锁定待办", "importance": 5, "locked": True}])
    with _patches():
        track_from_dialog(ps, "X", "p", model="test/m", incremental=True)
    todos = [i for i in ps.replaced if i["text"] == "锁定待办"]
    assert len(todos) == 1 and todos[0]["locked"] is True
    assert todos[0]["importance"] == 5   # 保留锁定版本(未被 fresh 的 9 覆盖)
```

- [ ] **Step 2: 运行确认失败**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_project_tracker.py -q -k "locked or manual or collides"`
Expected: FAIL(全量重跑 `items_data = fresh` 丢弃了锁定/手动项;增量未按 locked 保护)

- [ ] **Step 3: 实现**

替换 `project_tracker.py` 中 `track_from_dialog` 的 187-208 段(从 `existing = ...` 到 `items_data = fresh`):

```python
    existing = project_store.get_project(project_id)
    existing_items = [it.to_dict() for it in existing.items]

    # 提炼上下文:增量把全量旧条目喂给 LLM(让它见过历史);全量不喂。
    extract_existing = existing_items if incremental else None
    fresh = extract_dims_from_messages(
        msgs, dim_prompt, existing_items=extract_existing, model=model,
        progress_cb=progress_cb)

    if not fresh:
        # 提炼空(限流/解析失败):不覆盖旧记录,如实回报。
        return {"project_id": project_id, "new_messages": len(msgs),
                "status": "llm_empty",
                "profile": project_store.get_project(project_id)}

    # 锁定条目 + 手动条目:两条路径都原样保留,LLM 结果不得覆盖。
    preserved = [it for it in existing_items
                 if it.get("locked") or it.get("source") == "manual"]
    preserved_keys = {(it["dimension"], _norm(it["text"])) for it in preserved}
    fresh = [it for it in fresh
             if (it.get("dimension"), _norm(it.get("text", ""))) not in preserved_keys]

    if incremental:
        # 按维度替换:fresh 触及的维度里旧的未保留条目被本轮提炼替换掉;
        # fresh 未触及的维度保留旧条目(避免丢历史);preserved 始终保留。
        # (LLM 增量已吃过全量历史,touched 维度的 fresh 即该维度当前权威态,
        #  保留同维旧条目会导致重复漂移——正是要治的“太细散”。)
        fresh_dims = {it.get("dimension") for it in fresh}
        kept_old = {(it["dimension"], _norm(it["text"])): it
                    for it in existing_items
                    if (it["dimension"], _norm(it["text"])) not in preserved_keys
                    and it["dimension"] not in fresh_dims}
        for it in fresh:
            kept_old[(it.get("dimension"), _norm(it.get("text", "")))] = it
        items_data = preserved + list(kept_old.values())
    else:
        # 全量重跑:丢弃旧的未保留条目,但 preserved 保留。
        items_data = preserved + fresh
```

注意:`_norm` 已在文件末尾定义(`project_tracker.py:227`),`contextlib` 已在函数内 import;写盘段(`with lock_ctx:` ...)保持不变。

- [ ] **Step 4: 运行确认通过**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_project_tracker.py -q`
Expected: PASS(含既有的 incremental/full_rerun/write_lock/progress_cb 测试全绿)

- [ ] **Step 5: 提交**

```bash
git add memory_framework/project_tracker.py tests/test_project_tracker.py
git commit -m "feat(tracker): 锁定/手动条目在增量与全量重跑下不被覆盖"
```

---

### Task 5: 提示词聚焦到功能级/关键决策

**Files:**
- Modify: `memory_framework/persona_store.py:64-113`(`DEFAULT_DIM_PROMPT` + `DIM_MERGE_PROMPT`)
- Test: `tests/test_persona_store.py`(追加一条约束词断言)

**Interfaces:**
- Produces: `load_dim_prompt()` 默认文本包含聚焦约束(“功能”“不记琐碎”“关键”),四维合计目标 12–20 条。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_persona_store.py`:

```python
def test_default_dim_prompt_focuses_on_features():
    from memory_framework.persona_store import DEFAULT_DIM_PROMPT, DIM_MERGE_PROMPT
    # 聚焦到功能级/关键,明确不记琐碎修复
    assert "琐碎" in DEFAULT_DIM_PROMPT
    assert "功能" in DEFAULT_DIM_PROMPT
    assert "12" in DEFAULT_DIM_PROMPT and "20" in DEFAULT_DIM_PROMPT
    assert "12" in DIM_MERGE_PROMPT and "20" in DIM_MERGE_PROMPT
```

- [ ] **Step 2: 运行确认失败**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_persona_store.py::test_default_dim_prompt_focuses_on_features -q`
Expected: FAIL(当前是“10–25 条”,不含“12/20”组合)

- [ ] **Step 3: 实现**

在 `persona_store.py`。`DEFAULT_DIM_PROMPT` 的“核心原则”块把最后两条替换/强化,并把数量段改为 12–20:

在“核心原则”列表(第 70-73 行那几条)后追加一条,并改数量段。具体:把 `## 数量与粒度(重要)` 段的第一条改成:

```
- 追求**信息密度**而非条数。四个维度合计控制在**约 12–20 条**;若对话很短可更少。
```

并在“核心原则”块末尾(`- **进度要"有效"**` 那条之后)追加:

```
- **只记项目开发进度管理层面的内容**:功能/模块级进展、里程碑、架构与技术选型决策、当前仍未解决且**影响开发推进**的阻塞。
- **不记琐碎修复**:单个 bug 的修复、变量改名、加日志、格式调整、临时验证、措辞微调——除非该 bug 属于**关键阻塞**(阻断发布 / 影响全局),否则不单列;可并入相关功能的 progress 用一句话带过。
```

`DIM_MERGE_PROMPT` 的“控制总量”条把“约 **10–25 条**”改成“约 **12–20 条**”,并在合并规则里追加一条:

```
- **过滤琐碎**:合并阶段进一步删掉琐碎 bug 修复、改名、加日志、临时验证等非项目管理层面的条目;只保留功能级进展、里程碑、关键决策、仍未解决的关键阻塞。
```

(仅改文本,函数逻辑不动。)

- [ ] **Step 4: 运行确认通过**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_persona_store.py -q`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add memory_framework/persona_store.py tests/test_persona_store.py
git commit -m "feat(prompt): 四维提取/合并聚焦功能级,过滤琐碎修复(约12-20条)"
```

---

### Task 6: UI — proj_view 由 HTML 改为 gr.State + @gr.render 可交互面板

**Files:**
- Modify: `app.py`(`_render_project` 用途改变、新增 `_project_items(pid)` 辅助、UI 区块 1080-1092、切项目/加载/导入接线 808-823 / 670-755 / 1133-1142)
- 无独立单测(Gradio UI 层),靠 app 启动 + 手动走查验证;辅助函数 `_project_items` 可加轻量单测。
- Test: `tests/test_app_helpers.py`(新建,仅测纯函数 `_project_items`)

**Interfaces:**
- Consumes: Task 3 的 `get_project_store().set_locked/add_item/delete_item`;`get_project(pid).items`(带 locked/source)。
- Produces:
  - `_project_items(pid) -> list[dict]`:读 store 返回条目 dict 列表(含 dimension/text/importance/evidence/locked/source),空 pid 返回 `[]`。
  - `proj_items_state = gr.State([])`;`@gr.render(inputs=[g_project, proj_items_state])` 画四维交互面板。
  - `on_switch_project` / `_load_on_open` / `on_import_project` 的 `proj_view` 输出位改为输出 `proj_items_state`(list[dict])。

- [ ] **Step 1: 写失败测试(纯函数)**

新建 `tests/test_app_helpers.py`:

```python
import os
import app


def test_project_items_empty_pid():
    assert app._project_items("") == []
    assert app._project_items(None) == []


def test_project_items_reads_store(tmp_path, monkeypatch):
    from memory_framework.profile_store import ProjectStore
    ps = ProjectStore(base_dir=str(tmp_path / "pt"))
    ps.replace_profile("P", [
        {"dimension": "progress", "text": "做了X", "importance": 7,
         "locked": True, "source": "manual"}])
    monkeypatch.setattr(app, "get_project_store", lambda: ps)
    items = app._project_items("P")
    assert len(items) == 1
    it = items[0]
    assert it["text"] == "做了X" and it["locked"] is True
    assert it["source"] == "manual" and it["dimension"] == "progress"
```

- [ ] **Step 2: 运行确认失败**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_app_helpers.py -q`
Expected: FAIL(`AttributeError: module 'app' has no attribute '_project_items'`)

- [ ] **Step 3: 实现 `_project_items` 辅助**

在 `app.py`(`_render_project` 附近)加:

```python
def _project_items(project_id: str) -> list[dict]:
    """读某项目全部四维条目为 dict 列表(供 @gr.render 与 state 用)。"""
    if not project_id or not project_id.strip():
        return []
    prof = get_project_store().get_project(project_id.strip())
    return [it.to_dict() for it in prof.items]
```

- [ ] **Step 4: 运行确认 `_project_items` 通过**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_app_helpers.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 重构进度追踪 Tab 的 UI 区块**

把 `app.py:1080-1092` 的 `proj_view = gr.HTML(...)` + `_refresh_project` + 两个 `.click` 替换为 state + `@gr.render`。`_DIM_COLORS`/`_DIM_BADGE`/`PROJECT_DIMENSION_LABELS` 已存在,复用:

```python
                proj_status = gr.HTML(_status_bar("idle"))
                proj_items_state = gr.State([])

                @gr.render(inputs=[g_project, proj_items_state])
                def _render_proj_panel(pid, items):
                    if not pid or not str(pid).strip():
                        gr.HTML("<div class='panel-card'><p class='dim-empty'>"
                                "在顶部选择项目后点「导入并分析当前项目」</p></div>")
                        return
                    pid = str(pid).strip()
                    grouped = {}
                    for it in items:
                        grouped.setdefault(it["dimension"], []).append(it)
                    gr.Markdown(f"### {pid} · 进度追踪")
                    with gr.Row():
                        for dim, label in PROJECT_DIMENSION_LABELS.items():
                            rows = grouped.get(dim, [])
                            with gr.Column():
                                gr.Markdown(f"**{_DIM_BADGE[dim]} {label}** · {len(rows)}")
                                for it in rows:
                                    text, locked = it["text"], it.get("locked")
                                    src = it.get("source") == "manual"
                                    prefix = ("🔒 " if locked else "") + ("✎ " if src else "")
                                    with gr.Row():
                                        gr.Markdown(f"{prefix}{text}  ·  {it['importance']:.0f}")
                                        lock_btn = gr.Button(
                                            "解锁" if locked else "锁定", size="sm", scale=0)
                                        del_btn = gr.Button("删除", size="sm", scale=0)

                                        def _toggle_lock(pid=pid, dim=dim, text=text,
                                                         locked=locked):
                                            get_project_store().set_locked(
                                                pid, dim, text, not locked)
                                            return _project_items(pid)

                                        def _delete(pid=pid, dim=dim, text=text):
                                            get_project_store().delete_item(pid, dim, text)
                                            return _project_items(pid)

                                        lock_btn.click(_toggle_lock, None, proj_items_state,
                                                       show_progress="hidden")
                                        del_btn.click(_delete, None, proj_items_state,
                                                      show_progress="hidden")
                                with gr.Row():
                                    add_tb = gr.Textbox(show_label=False, scale=3,
                                                        placeholder=f"手动添加到「{label}」…")
                                    add_btn = gr.Button("+ 添加", size="sm", scale=1)

                                    def _add(text, pid=pid, dim=dim):
                                        get_project_store().add_item(pid, dim, text)
                                        return _project_items(pid), ""

                                    add_btn.click(_add, add_tb,
                                                  [proj_items_state, add_tb],
                                                  show_progress="hidden")

                import_btn.click(on_import_project, [g_project, proj_full_rerun],
                                 [proj_status, proj_items_state], show_progress="hidden")
                proj_refresh.click(lambda pid: _project_items(pid), g_project,
                                   proj_items_state, show_progress="hidden")
```

（删除原 `proj_view = gr.HTML(...)` 与 `_refresh_project` 定义。注意默认参数绑定 `pid=pid, dim=dim, text=text` 是必须的——闭包在循环里捕获当前值。）

- [ ] **Step 6: 改 `on_import_project` 的输出**

`app.py:670-755` 的 `on_import_project`：把所有 `yield <status_bar>, <panel_html>` 的第二个输出由 HTML 改为条目 list。首帧与中间轮询帧输出 `gr.update()`(不动 state），末帧输出 `_project_items(pid)`：

- 首帧(670-686):`yield _status_bar("reading", ...), _project_items(pid)`（去掉 `with _WRITE_LOCK: panel = _render_project(pid)`，直接读）。
- 轮询帧(722-729)第二个输出保持 `gr.update()`（不变）。
- error 帧(732-735):`yield _status_bar("error", detail=...), _project_items(pid)`。
- 末帧(753-755):去掉 `with _WRITE_LOCK: panel = _render_project(pid, note=note)`；改为 `yield bar, _project_items(pid)`；`note` 文案挪到状态栏 detail(已有 bar detail，note 可省或并入)。

（`_render_project` 不再被 UI 调用；可保留函数或删除——保留不影响，YAGNI 建议删，但删要确认无其它引用。用 `grep -n "_render_project" app.py` 确认后删除定义。）

- [ ] **Step 7: 改切项目/加载接线**

`app.py:808-823` `on_switch_project` 返回值第 2 位由 `proj_html` 改为 `_project_items(name)`；空 pid 分支第 2 位改为 `[]`。对应 `_switch_outputs`(1133)把 `proj_view` 换成 `proj_items_state`。
`_load_on_open`(demo.load 的 fn，1140-1142 输出列表)把 `proj_view` 位置改成 `proj_items_state`，其 fn 返回对应位置改为 `_project_items(name)`；先 `grep -n "_load_on_open" app.py` 找到定义改返回值。

- [ ] **Step 8: 启动 app 探测 + 手动走查**

Run(后台):`LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. conda run --no-capture-output -n mem0 python -u app.py`
探测:`curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860` → Expected `200`。
检查启动日志无 `UserWarning`(Gradio 参数位置)、无 traceback。
走查(浏览器或 gradio_client)：选 Mem0 → 面板显示四维条目 + 每条锁定/删除按钮 + 每维底部添加框；点锁定→前缀出现 🔒；添加一条→出现且带 ✎；删除→消失。

- [ ] **Step 9: 跑全量测试**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/ -q --ignore=tests/test_memory_store.py`
Expected: PASS(全绿)

- [ ] **Step 10: 提交**

```bash
git add app.py tests/test_app_helpers.py
git commit -m "feat(ui): 四维面板重构为可交互组件(锁定/删除/手动添加)"
```

---

### Task 7: 端到端验证 + 收尾

**Files:** 无改动(验证任务)

- [ ] **Step 1: 全量测试绿**

Run: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/ -q --ignore=tests/test_memory_store.py`
Expected: PASS。

- [ ] **Step 2: 全量重跑保留锁定项(真实数据)**

先在 UI 里对 Mem0 项目锁定一条、手动添加一条，勾「全量重跑」导入，确认锁定项与手动项仍在、其余被重提替换、四维条数收敛到约 12–20 且无琐碎 bug 条目。（LLM 限流导致 `llm_empty` 属正常，重试即可，非代码 bug。）

- [ ] **Step 3: git 状态干净**

Run: `git status`
Expected: working tree clean（所有改动已在 Task1–6 分次提交）。

- [ ] **Step 4: 若需要,push**

征询用户后 `git push`（本任务默认不自动 push）。

---

## Self-Review

**Spec coverage:**
- 聚焦粒度 → Task 5 ✓
- locked/source 数据模型 → Task 1 ✓
- 存储 set_locked/add_item/delete_item → Task 3(delete 复用基类)✓
- 重导入不覆盖锁定/手动(增量+全量) → Task 4 ✓（含修复全量 `items_data=fresh` 丢失 bug）
- UI 每条锁定/删除 + 每维添加 → Task 6 ✓
- 接线迁移(switch/load/import 输出 state) → Task 6 Step 6-7 ✓
- MCP 只读兼容 → 新字段是 JSON 多 key，不破坏(Task 1 向后兼容 from_dict)✓

**Placeholder scan:** 无 TBD；每个代码步给了完整代码或精确行号+替换文本。Task 6 Step 6-7 用“精确行号 + 改法描述”而非整段重贴（因散落在长函数里），已给出每处的输入/输出契约。

**Type consistency:** `set_locked(pid,dim,text,locked)->bool`、`add_item(pid,dim,text,importance=6)->UserProfile`、`delete_item(pid,dim,text)->bool`、`_project_items(pid)->list[dict]`、`_norm` 复用现有——全篇一致。`proj_items_state` 名称在 Task 6 各处一致。
