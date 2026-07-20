"""对话原文四维提取测试。patch _complete(避免真实 LLM/sleep),验证拼接/分段/去重。"""

import json
from unittest.mock import patch

import memory_framework.dialog_extractor as de
from memory_framework.dialog_extractor import (
    _chunk_messages,
    _merge_dedup,
    extract_dims_from_messages,
)


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_empty_messages_returns_empty():
    assert extract_dims_from_messages([], "prompt", model="test/m") == []


def test_chunk_by_char_cap():
    # 每条 ~1000 字,cap=2500 → 每段最多 2 条
    msgs = _msgs(*[("user", "x" * 1000) for _ in range(5)])
    chunks = _chunk_messages(msgs, cap=2500)
    assert len(chunks) == 3  # 2+2+1
    assert sum(len(c) for c in chunks) == 5


def test_merge_dedup_keeps_higher_importance():
    a = [{"dimension": "progress", "text": "完成 X", "importance": 5}]
    b = [{"dimension": "progress", "text": "完成 X", "importance": 8},
         {"dimension": "todo", "text": "做 Y", "importance": 4}]
    merged = _merge_dedup([a, b])
    prog = [m for m in merged if m["dimension"] == "progress"]
    assert len(prog) == 1 and prog[0]["importance"] == 8  # 取高者
    assert any(m["dimension"] == "todo" for m in merged)


@patch("memory_framework.dialog_extractor._complete")
def test_single_chunk_extracts(mock_complete):
    mock_complete.return_value = json.dumps([
        {"dimension": "progress", "text": "启动了 UI", "importance": 7},
        {"dimension": "blocker", "text": "429 限流", "importance": 8},
    ])
    items = extract_dims_from_messages(
        _msgs(("user", "我启动了 UI"), ("assistant", "好的")),
        "我的四维提示词", model="test/m")
    assert len(items) == 2
    # system 用的是传入的 dim_prompt
    sys_prompt = mock_complete.call_args[0][0][0]["content"]
    assert sys_prompt == "我的四维提示词"
    # user 里含对话原文
    user_msg = mock_complete.call_args[0][0][1]["content"]
    assert "我启动了 UI" in user_msg


@patch("memory_framework.dialog_extractor._complete")
def test_multi_chunk_consolidates(mock_complete):
    # 3 段各返回一条 + 第 4 次是跨段汇总,汇总输出即最终结果。
    calls = {"n": 0}

    def _resp(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 3:
            return json.dumps([{"dimension": "progress", "text": f"第{calls['n']}段进度",
                                "importance": 6}])
        # 第 4 次:跨段汇总,收敛成一条
        return json.dumps([{"dimension": "progress", "text": "汇总后的总进度",
                            "importance": 8}])
    mock_complete.side_effect = _resp

    big = _msgs(*[("user", "y" * 1000) for _ in range(5)])
    import memory_framework.dialog_extractor as mod
    orig = mod.CHUNK_CHARS
    mod.CHUNK_CHARS = 2500  # 5 条 → 3 段
    try:
        items = extract_dims_from_messages(big, "p", model="test/m")
    finally:
        mod.CHUNK_CHARS = orig
    assert calls["n"] == 4  # 3 段提取 + 1 次跨段汇总
    # 返回的是汇总结果,不是拼接的原始段结果
    texts = {i["text"] for i in items}
    assert texts == {"汇总后的总进度"}
    # 汇总那次用的是 DIM_MERGE_PROMPT
    merge_sys = mock_complete.call_args_list[3][0][0][0]["content"]
    assert "合并规则" in merge_sys


@patch("memory_framework.dialog_extractor._complete")
def test_single_chunk_skips_consolidation(mock_complete):
    # 单段:不触发跨段汇总,只调 1 次。
    mock_complete.return_value = json.dumps(
        [{"dimension": "progress", "text": "唯一进度", "importance": 5}])
    items = extract_dims_from_messages(
        _msgs(("user", "短对话")), "p", model="test/m")
    assert mock_complete.call_count == 1
    assert items[0]["text"] == "唯一进度"


@patch("memory_framework.dialog_extractor._complete")
def test_consolidation_failure_falls_back(mock_complete):
    # 汇总步骤失败(返回空)时,退回未收敛的精确去重结果,不丢数据。
    calls = {"n": 0}

    def _resp(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 3:
            return json.dumps([{"dimension": "todo", "text": f"待办{calls['n']}",
                                "importance": 5}])
        return ""  # 汇总失败
    mock_complete.side_effect = _resp

    big = _msgs(*[("user", "y" * 1000) for _ in range(5)])
    import memory_framework.dialog_extractor as mod
    orig = mod.CHUNK_CHARS
    mod.CHUNK_CHARS = 2500
    try:
        items = extract_dims_from_messages(big, "p", model="test/m")
    finally:
        mod.CHUNK_CHARS = orig
    texts = {i["text"] for i in items}
    assert texts == {"待办1", "待办2", "待办3"}  # 退回段结果


@patch("memory_framework.dialog_extractor._complete")
def test_all_llm_fail_returns_empty(mock_complete):
    mock_complete.return_value = ""  # LLM 全失败
    items = extract_dims_from_messages(_msgs(("user", "hi")), "p", model="test/m")
    assert items == []


@patch("memory_framework.dialog_extractor._complete")
def test_existing_items_injected(mock_complete):
    mock_complete.return_value = "[]"
    extract_dims_from_messages(
        _msgs(("user", "hi")), "p",
        existing_items=[{"dimension": "blocker", "text": "旧障碍"}],
        model="test/m")
    user_msg = mock_complete.call_args[0][0][1]["content"]
    assert "旧障碍" in user_msg and "现有四维记录" in user_msg


@patch("memory_framework.dialog_extractor._complete")
def test_progress_cb_called_per_chunk(mock_complete):
    mock_complete.return_value = "[]"
    calls = []
    big = _msgs(*[("user", "z" * 1000) for _ in range(5)])
    import memory_framework.dialog_extractor as mod
    orig = mod.CHUNK_CHARS
    mod.CHUNK_CHARS = 2500  # 5 条 → 3 段
    try:
        extract_dims_from_messages(big, "p", model="test/m",
                                   progress_cb=lambda cur, total: calls.append((cur, total)))
    finally:
        mod.CHUNK_CHARS = orig
    assert calls == [(1, 3), (2, 3), (3, 3)]  # 每段调一次,current 从 1 起,total 恒定


@patch("memory_framework.dialog_extractor._complete")
def test_progress_cb_merge_phase_signaled(mock_complete):
    # 多段且有结果 → 汇总阶段用 phase="merge" 再回调一次。
    mock_complete.return_value = json.dumps(
        [{"dimension": "progress", "text": "p", "importance": 5}])
    phases = []
    big = _msgs(*[("user", "z" * 1000) for _ in range(5)])
    import memory_framework.dialog_extractor as mod
    orig = mod.CHUNK_CHARS
    mod.CHUNK_CHARS = 2500  # 5 条 → 3 段
    try:
        extract_dims_from_messages(
            big, "p", model="test/m",
            progress_cb=lambda cur, total, phase="chunk": phases.append(phase))
    finally:
        mod.CHUNK_CHARS = orig
    assert phases == ["chunk", "chunk", "chunk", "merge"]


@patch("memory_framework.dialog_extractor._complete")
def test_progress_cb_failure_ignored(mock_complete):
    """回调抛异常不影响提炼结果。"""
    mock_complete.return_value = json.dumps(
        [{"dimension": "progress", "text": "ok", "importance": 5}])

    def _boom(*_a):
        raise RuntimeError("cb boom")

    items = extract_dims_from_messages(
        _msgs(("user", "hi")), "p", model="test/m", progress_cb=_boom)
    assert items and items[0]["text"] == "ok"

