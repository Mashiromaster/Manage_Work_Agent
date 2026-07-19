"""Claude Code 日志解析测试。不依赖真实 CC 日志,用临时 JSONL 文件构造。"""

import json

from memory_framework.cc_ingest import (
    _extract_text,
    _is_system_text,
    collect_new_messages,
    derive_project_id,
    derive_project_path,
    parse_cc_session,
)


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "session.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


class TestExtractText:
    def test_str_content(self):
        assert _extract_text("你好") == "你好"

    def test_block_list_takes_text_blocks(self):
        content = [
            {"type": "text", "text": "第一段"},
            {"type": "tool_use", "name": "x"},
            {"type": "text", "text": "第二段"},
        ]
        assert _extract_text(content) == "第一段\n第二段"

    def test_other_types_empty(self):
        assert _extract_text(None) == ""
        assert _extract_text(123) == ""


class TestIsSystemText:
    def test_system_prefixes_flagged(self):
        assert _is_system_text("<system-reminder>foo")
        assert _is_system_text("<task-notification>\n<task-id>x</task-id>")
        assert _is_system_text("  <command-name>/x</command-name>")
        assert _is_system_text("Caveat: something")

    def test_normal_text_not_flagged(self):
        assert not _is_system_text("我想优化聊天延迟")
        assert not _is_system_text("这是一条正常消息")


class TestParseCcSession:
    def test_filters_metadata_sidechain_and_system(self, tmp_path):
        rows = [
            {"type": "user", "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
             "message": {"role": "user", "content": "真实用户消息"}},
            {"type": "assistant", "uuid": "a1", "timestamp": "2026-01-01T00:00:01Z",
             "message": {"role": "assistant", "content": [{"type": "text", "text": "真实助手回复"}]}},
            {"type": "ai-title", "message": {"role": "user", "content": "标题元数据"}},
            {"type": "user", "isSidechain": True,
             "message": {"role": "user", "content": "子代理侧链"}},
            {"type": "user", "uuid": "u2",
             "message": {"role": "user", "content": "<task-notification>\nignore"}},
            {"type": "user", "uuid": "u3",
             "message": {"role": "user", "content": "   "}},
        ]
        msgs = parse_cc_session(_write_jsonl(tmp_path, rows))
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "真实用户消息",
                           "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z"}
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "真实助手回复"

    def test_missing_file_returns_empty(self):
        assert parse_cc_session("/nonexistent/x.jsonl") == []

    def test_bad_json_lines_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps({"type": "user", "uuid": "u1",
                                "message": {"role": "user", "content": "ok"}}) + "\n")
        msgs = parse_cc_session(str(p))
        assert len(msgs) == 1
        assert msgs[0]["content"] == "ok"


class TestDeriveProjectId:
    def test_last_segment(self):
        assert derive_project_id("-Users-mashiro-Mem0") == "Mem0"
        assert derive_project_id("-Users-mashiro-Documents-FPGA-Agent") == "Agent"

    def test_fallback_on_empty(self):
        assert derive_project_id("") == ""


class TestDeriveProjectPath:
    def test_restores_existing_dir(self):
        # /Users/mashiro/Mem0 存在且各段无连字符 → 可靠往返还原。
        import os
        target = "/Users/mashiro/Mem0"
        if os.path.isdir(target):
            assert derive_project_path("-Users-mashiro-Mem0") == target

    def test_nonexistent_returns_empty(self):
        assert derive_project_path("-nope-does-not-exist-xyz") == ""

    def test_non_absolute_returns_empty(self):
        assert derive_project_path("relative-name") == ""
        assert derive_project_path("") == ""


class TestCollectNewMessages:
    def test_incremental_dedup(self, tmp_path):
        rows = [
            {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "一"}},
            {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": "二"}},
        ]
        sess = _write_jsonl(tmp_path, rows)
        project = {"sessions": [sess]}
        new1, cur1 = collect_new_messages(project, {})
        assert len(new1) == 2
        new2, cur2 = collect_new_messages(project, cur1)
        assert new2 == []

    def test_new_message_after_cursor(self, tmp_path):
        rows = [{"type": "user", "uuid": "u1", "message": {"role": "user", "content": "一"}}]
        sess = _write_jsonl(tmp_path, rows)
        project = {"sessions": [sess]}
        _, cur1 = collect_new_messages(project, {})
        # append a new message to the same file
        with open(sess, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "assistant", "uuid": "a2",
                                "message": {"role": "assistant", "content": "新"}}) + "\n")
        new2, _ = collect_new_messages(project, cur1)
        assert len(new2) == 1
        assert new2[0]["content"] == "新"
