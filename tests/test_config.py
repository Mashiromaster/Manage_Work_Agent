import os

import pytest
from memory_framework.config import (
    build_config,
    apply_litellm_env,
    build_config_and_apply_env,
    MissingConfigError,
)


def test_build_config_has_three_sections(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    monkeypatch.setenv("MEM0_LLM_MODEL", "anthropic/Claude-Sonnet-4.7-hq")
    cfg = build_config()
    assert "llm" in cfg and "embedder" in cfg and "vector_store" in cfg


def test_build_config_embedder_is_local_bge(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    cfg = build_config()
    assert cfg["embedder"]["provider"] == "huggingface"
    assert "bge-small-zh" in cfg["embedder"]["config"]["model"]


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x")
    with pytest.raises(MissingConfigError):
        build_config()


def test_build_config_has_no_side_effects_on_environ(monkeypatch):
    """build_config 必须是纯函数:调用后 os.environ 不应被修改。"""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    before = dict(os.environ)
    build_config()
    after = dict(os.environ)

    assert after == before
    assert "ANTHROPIC_API_BASE" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_apply_litellm_env_sets_bridge_vars(monkeypatch):
    """apply_litellm_env 应把基址/密钥桥接到 litellm 识别的环境变量。"""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-bridge")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    apply_litellm_env()

    assert os.environ["ANTHROPIC_API_BASE"] == "https://relay.example"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-bridge"


def test_apply_litellm_env_missing_base_url_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    with pytest.raises(MissingConfigError):
        apply_litellm_env()


def test_build_config_and_apply_env_does_both(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-both")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = build_config_and_apply_env()

    assert "llm" in cfg and "embedder" in cfg and "vector_store" in cfg
    assert os.environ["ANTHROPIC_API_BASE"] == "https://relay.example"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-both"
