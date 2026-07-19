"""项目结构分析(LLM 归纳 + md 存取)测试。mock litellm,不触网。"""

from datetime import datetime
from unittest.mock import patch

import memory_framework.repo_analyzer as ra
from memory_framework.repo_analyzer import (
    analyze_project_structure,
    list_analyzed_projects,
    load_analysis,
    save_analysis,
)


def _snap():
    return {
        "root": "/x/Demo", "project_name": "Demo",
        "tree": "Demo/\n  app.py",
        "manifests": {"requirements.txt": "gradio\n"},
        "readme": "# Demo", "lang_stats": {".py": 1}, "entry_files": ["app.py"],
    }


class TestAnalyze:
    def test_empty_snapshot_returns_empty(self):
        assert analyze_project_structure({}, "Demo") == ""
        assert analyze_project_structure({"tree": ""}, "Demo") == ""

    @patch("memory_framework.repo_analyzer.litellm.completion")
    def test_returns_md(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "## 项目概述\nDemo 是示例项目"}}]
        }
        md = analyze_project_structure(_snap(), "Demo", model="test/model")
        assert "项目概述" in md
        # prompt 应带上目录树与清单
        user_msg = mock_completion.call_args[1]["messages"][1]["content"]
        assert "app.py" in user_msg
        assert "requirements.txt" in user_msg

    @patch("memory_framework.repo_analyzer.litellm.completion")
    def test_llm_exception_returns_empty(self, mock_completion):
        mock_completion.side_effect = RuntimeError("API 不可用")
        assert analyze_project_structure(_snap(), "Demo", model="test/model") == ""


class TestStorage:
    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "ANALYSIS_DIR", str(tmp_path))
        p = save_analysis("Demo", "## 项目概述\n内容", source_path="/x/Demo",
                          now=datetime(2026, 7, 19, 10, 0, 0))
        assert p.endswith("Demo.md")
        md = load_analysis("Demo")
        assert "项目概述" in md
        assert "/x/Demo" in md          # 源路径写进头部
        assert "2026-07-19" in md        # 时间戳写进头部

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "ANALYSIS_DIR", str(tmp_path))
        assert load_analysis("Nope") is None

    def test_list_analyzed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "ANALYSIS_DIR", str(tmp_path))
        save_analysis("Alpha", "a", now=datetime(2026, 1, 1))
        save_analysis("Beta", "b", now=datetime(2026, 1, 1))
        assert list_analyzed_projects() == ["Alpha", "Beta"]

    def test_safe_name_slash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "ANALYSIS_DIR", str(tmp_path))
        p = save_analysis("a/b", "x", now=datetime(2026, 1, 1))
        assert "a_b.md" in p
