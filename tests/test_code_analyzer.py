"""代码深度分析测试。ast 用真实解析(不 mock),LLM 调用 mock,tmp_path 造 fixture。"""

import json
from unittest.mock import patch

import pytest

import memory_framework.code_analyzer as ca
from memory_framework.code_analyzer import (
    build_import_graph,
    collect_source_files,
    deep_analyze,
    list_code_analyzed,
    load_code_analysis,
    save_code_analysis,
    summarize_files,
)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """重试退避的 sleep 在测试里置空,避免真实等待拖慢用例。"""
    monkeypatch.setattr(ca.time, "sleep", lambda *_: None)


@pytest.fixture
def sample_repo(tmp_path):
    # a 依赖 b,b 依赖 c(内部);a 还 import os(外部)。
    (tmp_path / "a.py").write_text(
        "import os\nfrom b import helper\n\ndef main():\n    helper()\n",
        encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "import c\n\ndef helper():\n    return c.value\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("value = 42\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo", encoding="utf-8")  # 非源码
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "junk.js").write_text("noise", encoding="utf-8")  # 忽略目录
    return str(tmp_path)


class TestCollect:
    def test_missing_dir(self):
        assert collect_source_files("/nonexistent/xyz") == []

    def test_collects_source_only(self, sample_repo):
        files = collect_source_files(sample_repo)
        rels = {f["rel"] for f in files}
        assert rels == {"a.py", "b.py", "c.py"}  # README/node_modules 排除

    def test_changed_only_filter(self, sample_repo):
        files = collect_source_files(sample_repo, changed_only=["b.py"])
        assert [f["rel"] for f in files] == ["b.py"]


class TestImportGraph:
    def test_edges_via_real_ast(self, sample_repo):
        files = collect_source_files(sample_repo)
        graph = build_import_graph(files)
        assert set(graph["nodes"]) == {"a", "b", "c"}
        assert ["a", "b"] in graph["edges"]
        assert ["b", "c"] in graph["edges"]
        assert "os" in graph["external"]

    def test_unparsed_tracked(self, tmp_path):
        (tmp_path / "broken.py").write_text("def (:\n", encoding="utf-8")
        files = collect_source_files(str(tmp_path))
        graph = build_import_graph(files)
        assert "broken.py" in graph["unparsed"]


class TestSummarize:
    @patch("memory_framework.code_analyzer.litellm.completion")
    def test_batching_and_cap(self, mock_completion, sample_repo):
        # 每批返回覆盖该批的摘要 JSON。
        def _resp(*args, **kwargs):
            user = kwargs["messages"][1]["content"]
            files = [ln.split("文件:")[1] for ln in user.splitlines()
                     if ln.startswith("### 文件:")]
            arr = [{"file": f, "role": "r", "key_symbols": [], "summary": "s"}
                   for f in files]
            return {"choices": [{"message": {"content": json.dumps(arr)}}]}
        mock_completion.side_effect = _resp

        files = collect_source_files(sample_repo)  # 3 个
        summaries, capinfo = summarize_files(files, model="test/m", cap=2)
        # cap=2 → 只分析前 2 个,第 3 个进 skipped
        assert capinfo["total"] == 3
        assert capinfo["analyzed"] == 2
        assert len(capinfo["skipped"]) == 1
        # FILE_BATCH=4,2 个文件一批 → 1 次调用
        assert capinfo["llm_calls"] == 1
        assert {s["file"] for s in summaries} == {"a.py", "b.py"}

    @patch("memory_framework.code_analyzer.litellm.completion")
    def test_llm_failure_placeholder(self, mock_completion, sample_repo):
        mock_completion.side_effect = RuntimeError("429")
        files = collect_source_files(sample_repo)
        summaries, capinfo = summarize_files(files, model="test/m")
        # 失败也要给每个文件占位,保证覆盖
        assert len(summaries) == 3
        assert all("未获得摘要" in s["summary"] for s in summaries)
        # 失败文件记入 capinfo.failed_files,供上层重试补齐
        assert set(capinfo["failed_files"]) == {"a.py", "b.py", "c.py"}

    @patch("memory_framework.code_analyzer.litellm.completion")
    def test_retry_then_succeed(self, mock_completion, sample_repo):
        # 前两次抛 429,第三次成功 → 重试应把结果救回来
        calls = {"n": 0}

        def _resp(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 rate limited")
            files = [ln.split("文件:")[1] for ln in kwargs["messages"][1]["content"]
                     .splitlines() if ln.startswith("### 文件:")]
            arr = [{"file": f, "role": "r", "key_symbols": [], "summary": "ok"}
                   for f in files]
            return {"choices": [{"message": {"content": json.dumps(arr)}}]}
        mock_completion.side_effect = _resp

        files = collect_source_files(sample_repo)[:1]  # 单文件单批,聚焦重试
        summaries, capinfo = summarize_files(files, model="test/m")
        assert summaries[0]["summary"] == "ok"
        assert capinfo["failed_files"] == []
        assert calls["n"] == 3  # 两次失败 + 一次成功


class TestDeepAnalyzeAndStorage:
    @patch("memory_framework.code_analyzer.litellm.completion")
    def test_deep_analyze_end_to_end(self, mock_completion, sample_repo):
        def _resp(*args, **kwargs):
            sys = kwargs["messages"][0]["content"]
            if "markdown 代码分析报告" in sys:  # 终归纳调用
                return {"choices": [{"message": {"content": "## 项目概述\n一个 demo"}}]}
            user = kwargs["messages"][1]["content"]
            files = [ln.split("文件:")[1] for ln in user.splitlines()
                     if ln.startswith("### 文件:")]
            arr = [{"file": f, "role": "r", "key_symbols": [], "summary": "s"}
                   for f in files]
            return {"choices": [{"message": {"content": json.dumps(arr)}}]}
        mock_completion.side_effect = _resp

        result = deep_analyze(sample_repo, "Demo", model="test/m")
        assert "项目概述" in result["md"]
        assert result["structured"]["graph"]["edges"]
        assert len(result["structured"]["summaries"]) == 3

    @patch("memory_framework.code_analyzer.litellm.completion")
    def test_synthesis_failure_falls_back_not_discarded(self, mock_completion,
                                                        sample_repo):
        # 逐文件摘要成功,但终归纳调用一直失败 → 不能丢弃已得摘要,应给降级报告
        def _resp(*args, **kwargs):
            sys = kwargs["messages"][0]["content"]
            if "markdown 代码分析报告" in sys:  # 终归纳:一直 429
                raise RuntimeError("429")
            files = [ln.split("文件:")[1] for ln in kwargs["messages"][1]["content"]
                     .splitlines() if ln.startswith("### 文件:")]
            arr = [{"file": f, "role": "r", "key_symbols": [], "summary": "s"}
                   for f in files]
            return {"choices": [{"message": {"content": json.dumps(arr)}}]}
        mock_completion.side_effect = _resp

        result = deep_analyze(sample_repo, "Demo", model="test/m")
        # md 非空(降级报告),摘要与依赖图都在,capinfo 标记 degraded
        assert result["md"]
        assert result["capinfo"]["degraded"] is True
        assert len(result["structured"]["summaries"]) == 3
        assert "a.py" in result["md"]  # 逐文件摘要进了降级报告


    def test_deep_analyze_missing_dir(self):
        result = deep_analyze("/nope/xyz", "X")
        assert result["md"] == ""

    @patch("memory_framework.code_analyzer.litellm.completion")
    def test_incremental_merges_with_prior(self, mock_completion, sample_repo,
                                            tmp_path, monkeypatch):
        monkeypatch.setattr(ca, "CODE_ANALYSIS_DIR", str(tmp_path / "out"))

        def _resp(*args, **kwargs):
            sys = kwargs["messages"][0]["content"]
            if "markdown 代码分析报告" in sys:
                return {"choices": [{"message": {"content": "## 概述\nx"}}]}
            user = kwargs["messages"][1]["content"]
            files = [ln.split("文件:")[1] for ln in user.splitlines()
                     if ln.startswith("### 文件:")]
            arr = [{"file": f, "role": "r", "key_symbols": [], "summary": "v1:" + f}
                   for f in files]
            return {"choices": [{"message": {"content": json.dumps(arr)}}]}
        mock_completion.side_effect = _resp

        # 首次全量,存盘
        full = deep_analyze(sample_repo, "Demo", model="test/m")
        save_code_analysis("Demo", full["md"], full["structured"])
        assert len(full["structured"]["summaries"]) == 3

        # 增量:只重分析 a.py,其余复用旧摘要(合并后仍 3 个)
        inc = deep_analyze(sample_repo, "Demo", model="test/m", changed_only=["a.py"])
        summ = {s["file"]: s for s in inc["structured"]["summaries"]}
        assert set(summ) == {"a.py", "b.py", "c.py"}
        assert inc["capinfo"]["incremental"] is True
        assert inc["capinfo"]["reanalyzed"] == ["a.py"]

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ca, "CODE_ANALYSIS_DIR", str(tmp_path))
        structured = {"summaries": [{"file": "a.py"}], "graph": {}, "capinfo": {}}
        md_path, json_path = save_code_analysis(
            "Demo", "## 概述\n内容", structured, source_path="/x/Demo")
        assert md_path.endswith("Demo.md") and json_path.endswith("Demo.json")
        data = load_code_analysis("Demo")
        assert data["summaries"][0]["file"] == "a.py"
        assert "/x/Demo" in data["source_path"]
        assert "概述" in data["md"]

    def test_load_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ca, "CODE_ANALYSIS_DIR", str(tmp_path))
        assert load_code_analysis("Nope") is None

    def test_list_code_analyzed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ca, "CODE_ANALYSIS_DIR", str(tmp_path))
        save_code_analysis("Beta", "b", {"summaries": []})
        save_code_analysis("Alpha", "a", {"summaries": []})
        assert list_code_analyzed() == ["Alpha", "Beta"]
