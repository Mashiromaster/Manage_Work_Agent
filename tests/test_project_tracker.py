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
