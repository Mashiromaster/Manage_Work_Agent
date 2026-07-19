"""用户人设存储测试。tmp 目录,纯文件读写。"""

import json

import pytest

import memory_framework.persona_store as ps
from memory_framework.persona_store import (
    list_persona_users,
    load_persona,
    save_persona,
)


@pytest.fixture(autouse=True)
def _tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PERSONA_DIR", str(tmp_path / "persona"))


def test_default_when_missing():
    d = load_persona("nobody")
    assert d == {"persona": "", "inject": True}


def test_save_load_roundtrip():
    path = save_persona("alice", "你是我的中文编程助手,回答简洁", inject=False)
    assert path.endswith("alice.json")
    d = load_persona("alice")
    assert d["persona"] == "你是我的中文编程助手,回答简洁"
    assert d["inject"] is False


def test_inject_defaults_true_on_partial_json(tmp_path):
    # 手写一个只有 persona、无 inject 的文件 → inject 缺省 True
    p = tmp_path / "persona"
    p.mkdir()
    (p / "bob.json").write_text(json.dumps({"persona": "x"}), encoding="utf-8")
    d = load_persona("bob")
    assert d["persona"] == "x" and d["inject"] is True


def test_list_users():
    save_persona("beta", "b")
    save_persona("alpha", "a")
    assert list_persona_users() == ["alpha", "beta"]


def test_safe_name_slash():
    save_persona("a/b", "x")
    assert load_persona("a/b")["persona"] == "x"  # 归一化后可回读
