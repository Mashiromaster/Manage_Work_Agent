"""Skill 导出测试:断言 SKILL.md frontmatter + reference 文件生成,内容来自盘上产物。"""

import json
import os

import pytest

import mcp_server.server as srv
from skill_export.exporter import _skill_slug, export_skill


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    tracking = tmp_path / "project_tracking"
    code = tmp_path / "code_analysis"
    structure = tmp_path / "project_analysis"
    for d in (tracking, code, structure):
        d.mkdir()
    monkeypatch.setattr(srv, "PROJECT_TRACKING_DIR", str(tracking))
    monkeypatch.setattr(srv, "STRUCTURE_ANALYSIS_DIR", str(structure))
    import memory_framework.code_analyzer as ca
    monkeypatch.setattr(ca, "CODE_ANALYSIS_DIR", str(code))
    monkeypatch.setattr(srv, "CODE_ANALYSIS_DIR", str(code))

    (tracking / "Mem0.json").write_text(json.dumps({
        "user_id": "Mem0",
        "items": [{"text": "完成 MCP", "dimension": "progress", "importance": 8,
                   "created_at": "2026-01-01T00:00:00",
                   "last_seen": "2026-01-01T00:00:00"}],
    }), encoding="utf-8")
    (code / "Mem0.json").write_text(json.dumps({
        "project_name": "Mem0",
        "summaries": [{"file": "app.py", "role": "UI", "summary": "入口"}],
    }), encoding="utf-8")
    (code / "Mem0.md").write_text("## 项目概述\ndemo", encoding="utf-8")
    (structure / "Mem0.md").write_text("## 结构\n树", encoding="utf-8")
    return tmp_path


def test_slug():
    assert _skill_slug("Mem0") == "mem0"
    assert _skill_slug("My Cool/Proj") == "my-cool-proj"


def test_export_creates_skill(seeded, tmp_path):
    target = str(tmp_path / "skills")
    path = export_skill("Mem0", target=target)
    assert path.endswith(os.path.join("Mem0", "SKILL.md"))
    assert os.path.isfile(path)

    with open(path, encoding="utf-8") as f:
        skill = f.read()
    # frontmatter
    assert skill.startswith("---\n")
    assert "name: mem0" in skill
    assert "description:" in skill

    ref = os.path.join(target, "Mem0", "reference")
    assert os.path.isfile(os.path.join(ref, "progress.md"))
    assert os.path.isfile(os.path.join(ref, "structure.md"))
    assert os.path.isfile(os.path.join(ref, "code_analysis.md"))

    with open(os.path.join(ref, "progress.md"), encoding="utf-8") as f:
        assert "完成 MCP" in f.read()
    with open(os.path.join(ref, "code_analysis.md"), encoding="utf-8") as f:
        assert "项目概述" in f.read()


def test_export_missing_products(seeded, tmp_path):
    # 没有任何产物的项目也应生成骨架(占位内容),不报错
    target = str(tmp_path / "skills")
    path = export_skill("Ghost", target=target)
    assert os.path.isfile(path)
    with open(os.path.join(target, "Ghost", "reference", "progress.md"),
              encoding="utf-8") as f:
        assert "暂无进度" in f.read()
