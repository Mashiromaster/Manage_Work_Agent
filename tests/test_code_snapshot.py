"""代码变更快照测试。tmp_path 造文件,改动后 diff。"""

import pytest

import memory_framework.code_snapshot as cs
from memory_framework.code_analyzer import collect_source_files
from memory_framework.code_snapshot import (
    compute_hashes,
    diff_snapshot,
    load_snapshot,
    save_snapshot,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    return str(tmp_path)


def _hashes(root):
    return compute_hashes(collect_source_files(root))


class TestSnapshot:
    def test_first_run_all_added(self, repo, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "SNAPSHOT_DIR", str(tmp_path / "snap"))
        d = diff_snapshot("Demo", _hashes(repo))
        assert set(d["added"]) == {"a.py", "b.py"}
        assert d["changed"] == [] and d["removed"] == []

    def test_roundtrip_and_no_change(self, repo, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "SNAPSHOT_DIR", str(tmp_path / "snap"))
        h = _hashes(repo)
        save_snapshot("Demo", h)
        assert load_snapshot("Demo") == h
        d = diff_snapshot("Demo", _hashes(repo))
        assert d == {"added": [], "changed": [], "removed": []}

    def test_detect_changed(self, repo, tmp_path, monkeypatch):
        import pathlib
        monkeypatch.setattr(cs, "SNAPSHOT_DIR", str(tmp_path / "snap"))
        save_snapshot("Demo", _hashes(repo))
        # 改 a.py 的内容与大小
        (pathlib.Path(repo) / "a.py").write_text(
            "x = 1\nx = 999999\n", encoding="utf-8")
        d = diff_snapshot("Demo", _hashes(repo))
        assert d["changed"] == ["a.py"]
        assert d["added"] == [] and d["removed"] == []

    def test_detect_added_removed(self, repo, tmp_path, monkeypatch):
        import pathlib
        monkeypatch.setattr(cs, "SNAPSHOT_DIR", str(tmp_path / "snap"))
        save_snapshot("Demo", _hashes(repo))
        (pathlib.Path(repo) / "b.py").unlink()          # 删 b
        (pathlib.Path(repo) / "c.py").write_text("z=3\n", encoding="utf-8")  # 增 c
        d = diff_snapshot("Demo", _hashes(repo))
        assert d["added"] == ["c.py"]
        assert d["removed"] == ["b.py"]

    def test_load_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "SNAPSHOT_DIR", str(tmp_path / "snap"))
        assert load_snapshot("Nope") == {}
