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
