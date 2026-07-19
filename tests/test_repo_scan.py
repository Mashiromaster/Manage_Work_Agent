"""项目目录扫描测试。用临时目录构造 fixture,不触真实仓库。"""

import os

import pytest

from memory_framework.repo_scan import scan_repo


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n这是一个示例项目。", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("gradio\nmem0ai\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hi')", encoding="utf-8")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "core.py").write_text("x = 1", encoding="utf-8")
    (pkg / "util.py").write_text("y = 2", encoding="utf-8")
    # 应被忽略的目录与文件
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "junk.js").write_text("noise", encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("noise", encoding="utf-8")
    return str(tmp_path)


class TestScanRepo:
    def test_missing_dir_returns_empty(self):
        s = scan_repo("/nonexistent/path/xyz")
        assert s["tree"] == ""
        assert s["manifests"] == {}
        assert s["entry_files"] == []

    def test_tree_excludes_ignored(self, sample_repo):
        s = scan_repo(sample_repo)
        assert "node_modules" not in s["tree"]
        assert "junk.js" not in s["tree"]
        assert ".DS_Store" not in s["tree"]

    def test_manifests_read(self, sample_repo):
        s = scan_repo(sample_repo)
        assert "requirements.txt" in s["manifests"]
        assert "gradio" in s["manifests"]["requirements.txt"]

    def test_readme_read(self, sample_repo):
        s = scan_repo(sample_repo)
        assert "示例项目" in s["readme"]

    def test_lang_stats(self, sample_repo):
        s = scan_repo(sample_repo)
        # app.py + pkg/core.py + pkg/util.py = 3 个 .py;node_modules 里的 .js 不计
        assert s["lang_stats"].get(".py") == 3
        assert ".js" not in s["lang_stats"]

    def test_entry_files_detected(self, sample_repo):
        s = scan_repo(sample_repo)
        assert "app.py" in s["entry_files"]

    def test_project_name(self, sample_repo):
        s = scan_repo(sample_repo)
        assert s["project_name"] == os.path.basename(sample_repo)

    def test_max_entries_truncates(self, tmp_path):
        for i in range(50):
            (tmp_path / f"f{i}.py").write_text("x", encoding="utf-8")
        s = scan_repo(str(tmp_path), max_entries=10)
        assert "已截断" in s["tree"]
