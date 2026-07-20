"""project_tracker.refresh_project 状态回报测试。假 store,mock 提炼器。"""

from unittest.mock import patch

from memory_framework.project_tracker import refresh_project


class FakeMemStore:
    def __init__(self, mems):
        self._mems = mems

    def get_all(self, user_id):
        return self._mems


class FakeProfile:
    def __init__(self, items=None):
        self.items = items or []


class FakeProjectStore:
    def __init__(self):
        self.replaced = None
        self._profile = FakeProfile()

    def get_project(self, project_id):
        return self._profile

    def replace_profile(self, project_id, items_data, now=None):
        self.replaced = items_data
        self._profile = FakeProfile(items_data)
        return self._profile


def test_no_memories_status():
    store = FakeMemStore([])
    ps = FakeProjectStore()
    profile, status = refresh_project(store, ps, "X", model="test/m")
    assert status == "no_memories"
    assert ps.replaced is None  # 未写盘


@patch("memory_framework.project_tracker.synthesize_project_dims")
def test_llm_empty_keeps_old_and_reports(mock_synth):
    mock_synth.return_value = []  # 提炼返回空(限流/解析失败)
    store = FakeMemStore([{"memory": "做了点事"}])
    ps = FakeProjectStore()
    profile, status = refresh_project(store, ps, "X", model="test/m")
    assert status == "llm_empty"
    assert ps.replaced is None  # 关键:不覆盖旧记录


@patch("memory_framework.project_tracker.synthesize_project_dims")
def test_ok_writes_and_reports(mock_synth):
    mock_synth.return_value = [
        {"dimension": "progress", "text": "完成 X", "importance": 7}]
    store = FakeMemStore([{"memory": "完成了 X"}])
    ps = FakeProjectStore()
    profile, status = refresh_project(store, ps, "X", model="test/m")
    assert status == "ok"
    assert ps.replaced and ps.replaced[0]["text"] == "完成 X"


# ---- track_from_dialog(对话原文直接提炼,不经 mem0) ----

class _Item:
    def __init__(self, d):
        self._d = d
    def to_dict(self):
        return self._d


class FakeDialogStore:
    def __init__(self, existing=None, base_dir="/tmp/_pt_test"):
        self.base_dir = base_dir
        self.replaced = None
        self._items = [_Item(d) for d in (existing or [])]

    def get_project(self, pid):
        return FakeProfile(self._items)

    def replace_profile(self, pid, items_data, now=None):
        self.replaced = items_data
        self._items = [_Item(d) for d in items_data]
        return FakeProfile(self._items)


def _patches():
    return patch.multiple(
        "memory_framework.project_tracker",
        find_project=lambda pid: {"project_id": pid, "sessions": ["s.jsonl"]},
        load_cursor=lambda p: {},
        save_cursor=lambda p, c: None,
    )


@patch("memory_framework.project_tracker.collect_new_messages")
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_dialog_not_found(mock_ext, mock_collect):
    with patch("memory_framework.project_tracker.find_project", return_value=None):
        from memory_framework.project_tracker import track_from_dialog
        r = track_from_dialog(FakeDialogStore(), "X", "p", model="test/m")
    assert r["status"] == "not_found"


@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([], {}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_dialog_no_messages(mock_ext, mock_collect):
    from memory_framework.project_tracker import track_from_dialog
    with _patches():
        r = track_from_dialog(FakeDialogStore(), "X", "p", model="test/m")
    assert r["status"] == "no_messages"


@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([{"role": "user", "content": "hi"}], {"s.jsonl": ["k"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages",
       return_value=[])
def test_dialog_llm_empty_keeps_old(mock_ext, mock_collect):
    from memory_framework.project_tracker import track_from_dialog
    ps = FakeDialogStore(existing=[{"dimension": "todo", "text": "旧", "importance": 5}])
    with _patches():
        r = track_from_dialog(ps, "X", "p", model="test/m")
    assert r["status"] == "llm_empty"
    assert ps.replaced is None  # 空不覆盖


@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([{"role": "user", "content": "hi"}], {"s.jsonl": ["k"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_dialog_incremental_merges(mock_ext, mock_collect):
    mock_ext.return_value = [{"dimension": "progress", "text": "新进度", "importance": 8}]
    from memory_framework.project_tracker import track_from_dialog
    ps = FakeDialogStore(existing=[{"dimension": "todo", "text": "旧待办", "importance": 5}])
    with _patches():
        r = track_from_dialog(ps, "X", "p", model="test/m", incremental=True)
    assert r["status"] == "ok"
    texts = {i["text"] for i in ps.replaced}
    assert "新进度" in texts and "旧待办" in texts  # 合并保留旧的


@patch("memory_framework.project_tracker.parse_cc_session",
       return_value=[{"role": "user", "content": "全量原文"}])
@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([], {"s.jsonl": ["k1", "k2"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_dialog_full_rerun_replaces(mock_ext, mock_collect, mock_parse):
    mock_ext.return_value = [{"dimension": "progress", "text": "重提", "importance": 7}]
    from memory_framework.project_tracker import track_from_dialog
    ps = FakeDialogStore(existing=[{"dimension": "todo", "text": "旧", "importance": 5}])
    with _patches():
        r = track_from_dialog(ps, "X", "p", model="test/m", incremental=False)
    assert r["status"] == "ok"
    # 全量:直接替换,旧待办不保留
    texts = {i["text"] for i in ps.replaced}
    assert texts == {"重提"}


@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([{"role": "user", "content": "hi"}], {"s.jsonl": ["k"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_dialog_progress_cb_threaded_through(mock_ext, mock_collect):
    """track_from_dialog 应把 progress_cb 透传给 extract_dims_from_messages。"""
    mock_ext.return_value = [{"dimension": "progress", "text": "p", "importance": 5}]
    from memory_framework.project_tracker import track_from_dialog
    sentinel = object()
    with _patches():
        track_from_dialog(FakeDialogStore(), "X", "p", model="test/m",
                          progress_cb=sentinel)
    assert mock_ext.call_args.kwargs.get("progress_cb") is sentinel


@patch("memory_framework.project_tracker.collect_new_messages",
       return_value=([{"role": "user", "content": "hi"}], {"s.jsonl": ["k"]}))
@patch("memory_framework.project_tracker.extract_dims_from_messages")
def test_dialog_write_lock_held_only_around_write(mock_ext, mock_collect):
    """write_lock 只应在写盘(replace_profile)时持有,提炼(extract)时不持锁。"""
    mock_ext.return_value = [{"dimension": "progress", "text": "p", "importance": 5}]
    from memory_framework.project_tracker import track_from_dialog

    events = []

    class RecordingLock:
        def __enter__(self):
            events.append("lock")
            return self

        def __exit__(self, *a):
            events.append("unlock")
            return False

    # 提炼时记录是否持锁:extract 被调用时 events 里不应有未配对的 "lock"。
    def _ext(*a, **k):
        events.append("extract")
        return [{"dimension": "progress", "text": "p", "importance": 5}]
    mock_ext.side_effect = _ext

    ps = FakeDialogStore()
    orig_replace = ps.replace_profile

    def _rec_replace(*a, **k):
        events.append("write")
        return orig_replace(*a, **k)
    ps.replace_profile = _rec_replace

    with _patches():
        track_from_dialog(ps, "X", "p", model="test/m", write_lock=RecordingLock())

    # 提炼在加锁之前;加锁包住写盘。
    assert events.index("extract") < events.index("lock")
    assert events.index("lock") < events.index("write") < events.index("unlock")


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



def test_read_messages_max_mb_picks_recent(tmp_path):
    """max_mb>0 时只读最近的会话文件累计到上限;至少读入最近一个。"""
    import os
    from memory_framework import project_tracker as pt

    old = tmp_path / "old.jsonl"
    mid = tmp_path / "mid.jsonl"
    new = tmp_path / "new.jsonl"
    for f in (old, mid, new):
        f.write_bytes(b"x" * (600 * 1024))  # 各 ~0.6MB
    # 设 mtime:old < mid < new
    os.utime(old, (1000, 1000))
    os.utime(mid, (2000, 2000))
    os.utime(new, (3000, 3000))

    seen = []

    def fake_parse(path):
        seen.append(os.path.basename(path))
        return [{"role": "user", "content": "hi"}]

    project = {"sessions": [str(old), str(mid), str(new)]}
    with patch.object(pt, "parse_cc_session", side_effect=fake_parse):
        pt._read_project_messages(project, max_mb=1)  # ~1MB → 只容得下最近一个多一点

    # 最近的 new 必读;old 最旧应被排除
    assert "new.jsonl" in seen
    assert "old.jsonl" not in seen


def test_read_messages_zero_mb_reads_all(tmp_path):
    """max_mb=0(默认)读全部会话。"""
    import os
    from memory_framework import project_tracker as pt

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for f in (a, b):
        f.write_bytes(b"x" * 1024)
    seen = []
    with patch.object(pt, "parse_cc_session",
                      side_effect=lambda p: seen.append(os.path.basename(p)) or []):
        pt._read_project_messages({"sessions": [str(a), str(b)]}, max_mb=0)
    assert set(seen) == {"a.jsonl", "b.jsonl"}
