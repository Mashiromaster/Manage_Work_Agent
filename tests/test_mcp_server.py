"""MCP server 测试:只读磁盘产物,断言绝不构造 MemoryStore / 打开 Qdrant。

关键:monkeypatch MemoryStore.__init__ 使其一旦被构造即抛错,若任一工具触碰
Qdrant 测试立即失败——这是整个设计的承重约束。
"""

import json
import os

import pytest

import mcp_server.server as srv


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    tracking = tmp_path / "project_tracking"
    code = tmp_path / "code_analysis"
    structure = tmp_path / "project_analysis"
    persona = tmp_path / "persona"
    profiles = tmp_path / "profiles"
    tracking.mkdir()
    code.mkdir()
    structure.mkdir()
    persona.mkdir()
    profiles.mkdir()

    monkeypatch.setattr(srv, "PROJECT_TRACKING_DIR", str(tracking))
    monkeypatch.setattr(srv, "CODE_ANALYSIS_DIR", str(code))
    monkeypatch.setattr(srv, "STRUCTURE_ANALYSIS_DIR", str(structure))
    monkeypatch.setattr(srv, "PROFILE_DIR", str(profiles))
    # load_code_analysis 读的是 code_analyzer.CODE_ANALYSIS_DIR,也要指过去
    import memory_framework.code_analyzer as ca
    monkeypatch.setattr(ca, "CODE_ANALYSIS_DIR", str(code))
    # persona_store 的 PERSONA_DIR 也要指过去(server 通过它读)
    import memory_framework.persona_store as pstore
    monkeypatch.setattr(pstore, "PERSONA_DIR", str(persona))

    # 用户人设 + 画像
    (persona / "demo_user.json").write_text(json.dumps(
        {"persona": "你是我的中文编程助手,回答简洁", "inject": True}),
        encoding="utf-8")
    (profiles / "demo_user.json").write_text(json.dumps({
        "user_id": "demo_user",
        "items": [{"text": "后端工程师", "dimension": "event",
                   "importance": 7, "created_at": "2026-01-01T00:00:00",
                   "last_seen": "2026-01-01T00:00:00"}],
    }), encoding="utf-8")

    # 进度四维
    (tracking / "Mem0.json").write_text(json.dumps({
        "user_id": "Mem0",
        "items": [
            {"text": "完成结构化画像", "dimension": "progress",
             "importance": 8, "created_at": "2026-01-01T00:00:00",
             "last_seen": "2026-01-01T00:00:00"},
            {"text": "接入 MCP", "dimension": "todo",
             "importance": 6, "created_at": "2026-01-01T00:00:00",
             "last_seen": "2026-01-01T00:00:00"},
        ],
    }), encoding="utf-8")

    # 代码深度分析
    (code / "Mem0.json").write_text(json.dumps({
        "project_name": "Mem0",
        "summaries": [
            {"file": "app.py", "role": "UI 入口", "key_symbols": ["build_ui"],
             "summary": "Gradio 应用入口"},
            {"file": "memory_framework/chat.py", "role": "聊天",
             "key_symbols": ["reply"], "summary": "长期记忆聊天"},
        ],
        "graph": {"edges": [["app", "memory_framework.chat"]]},
    }), encoding="utf-8")
    (code / "Mem0.md").write_text("## 项目概述\nMem0 是记忆项目", encoding="utf-8")

    # 结构分析
    (structure / "Mem0.md").write_text("## 结构\n目录树...", encoding="utf-8")

    # 只做代码分析、没进度的项目
    (code / "Solo.json").write_text(json.dumps(
        {"project_name": "Solo", "summaries": []}), encoding="utf-8")

    return tmp_path


@pytest.fixture(autouse=True)
def forbid_qdrant(monkeypatch):
    """任何 MemoryStore 构造都视为违反只读约束 → 抛错。"""
    import memory_framework.memory_store as ms

    def _boom(self, *a, **k):
        raise AssertionError("MCP server 触碰了 MemoryStore / Qdrant!")

    monkeypatch.setattr(ms.MemoryStore, "__init__", _boom)


def test_list_projects(seeded):
    assert srv.list_projects() == ["Mem0", "Solo"]


def test_get_progress(seeded):
    prog = srv.get_progress("Mem0")
    # 按 label 分组
    assert any("完成结构化画像" in v for v in prog.values() for v in [v])
    flat = [t for texts in prog.values() for t in texts]
    assert "完成结构化画像" in flat and "接入 MCP" in flat


def test_get_progress_missing(seeded):
    assert srv.get_progress("Nope") == {}


def test_get_code_analysis(seeded):
    md = srv.get_code_analysis("Mem0")
    assert "项目概述" in md


def test_get_code_analysis_missing(seeded):
    assert "尚无" in srv.get_code_analysis("Nope")


def test_search_code_analysis_hit(seeded):
    out = srv.search_code_analysis("Mem0", "聊天")
    assert "chat.py" in out and "app.py" not in out


def test_search_code_analysis_no_hit(seeded):
    assert "未找到" in srv.search_code_analysis("Mem0", "zzznotexist")


def test_get_structure_analysis(seeded):
    assert "目录树" in srv.get_structure_analysis("Mem0")


def test_get_project_memory_full(seeded):
    mem = srv.get_project_memory("Mem0")
    assert "开发进度" in mem
    assert "完成结构化画像" in mem
    assert "app.py" in mem
    assert "app → memory_framework.chat" in mem


def test_get_project_memory_empty(seeded):
    assert "暂无任何长期记忆" in srv.get_project_memory("Ghost")


def test_list_users(seeded):
    assert srv.list_users() == ["demo_user"]


def test_get_user_persona(seeded):
    out = srv.get_user_persona("demo_user")
    assert "你是我的中文编程助手,回答简洁" in out  # 人设
    assert "后端工程师" in out  # 画像


def test_get_user_persona_missing(seeded):
    assert "暂无人设" in srv.get_user_persona("nobody")


def test_get_user_persona_inject_off(seeded, tmp_path, monkeypatch):
    import memory_framework.persona_store as pstore
    (tmp_path / "persona" / "off_user.json").write_text(
        '{"persona": "简洁", "inject": false}', encoding="utf-8")
    out = srv.get_user_persona("off_user")
    assert "已在 Mem0 关闭" in out and "简洁" in out


def test_build_server_registers_tools(seeded):
    # 构造 FastMCP 不应触碰 Qdrant;能建起来即通过。
    mcp = srv.build_server()
    assert mcp is not None
