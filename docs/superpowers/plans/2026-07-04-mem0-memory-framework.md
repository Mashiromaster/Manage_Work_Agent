# Mem0 长期记忆框架 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Mem0 搭建一个可用的长期记忆框架,支持从对话自动提取记忆、语义检索、按 user_id 隔离。

**Architecture:** 用 `MemoryStore` 薄封装隔离 Mem0,所有配置集中在 `config.py`。LLM 用 litellm 转 Claude,embedding 用本地 HF bge-small-zh,向量库用 Qdrant 本地模式。四条开发线(数据/框架/测试/研究)按依赖并行。

**Tech Stack:** Python 3.11 (conda env `sft`), mem0ai, litellm, sentence-transformers, qdrant-client, pytest。

## Global Constraints

- Python 环境:conda env **`sft`**(Python 3.11.15),所有 pip/pytest 命令前先 `conda activate sft`。
- 工作目录:`/Users/mashiro/Mem0`,所有路径相对于此。
- LLM 密钥来自环境变量 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`,**不写死进代码**。
- Embedding 模型:`BAAI/bge-small-zh-v1.5`(384 维),纯本地。
- 向量库:Qdrant 本地文件模式,数据存 `./qdrant_data/`。
- **配置键校验**:Mem0/litellm/embedder 的确切配置键,必须在依赖安装后对照已装版本核实(见 Task 2 Step 2),不得凭记忆写死。
- 每个任务结束提交一次 git commit。

---

### Task 1: 项目骨架与依赖

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `memory_framework/__init__.py`

**Interfaces:**
- Consumes: 无
- Produces: 可安装的依赖环境;`memory_framework` 包可 import。

- [ ] **Step 1: 写 `requirements.txt`**

```
mem0ai>=0.1.0
litellm>=1.40.0
sentence-transformers>=2.2.0
qdrant-client>=1.7.0
pytest>=7.0.0
```

- [ ] **Step 2: 写 `.env.example`**

```
# Claude via litellm — 复用中转站
ANTHROPIC_BASE_URL=https://your-proxy-endpoint
ANTHROPIC_AUTH_TOKEN=sk-xxxx
# 记忆提取用的模型名(litellm 格式,如 anthropic/claude-3-5-sonnet-20241022)
MEM0_LLM_MODEL=anthropic/claude-3-5-sonnet-20241022
```

- [ ] **Step 3: 写 `.gitignore`**

```
__pycache__/
*.pyc
.env
qdrant_data/
.pytest_cache/
models/
```

- [ ] **Step 4: 建空包文件** — `memory_framework/__init__.py` 内容:`"""Mem0 长期记忆框架。"""`

- [ ] **Step 5: 安装依赖并验证**

Run:
```bash
conda activate sft && cd /Users/mashiro/Mem0 && pip install -r requirements.txt
python -c "import mem0, litellm, sentence_transformers, qdrant_client; print('all imported')"
```
Expected: 打印 `all imported`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore memory_framework/__init__.py
git commit -m "chore: project skeleton and dependencies"
```

---

### Task 2: 配置模块 `config.py`

**Files:**
- Create: `memory_framework/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 环境变量 `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `MEM0_LLM_MODEL`
- Produces: `build_config() -> dict`(Mem0 的 config dict,含 llm/embedder/vector_store 三块);`MissingConfigError`(异常类)。

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
import os
import pytest
from memory_framework.config import build_config, MissingConfigError


def test_build_config_has_three_sections(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    monkeypatch.setenv("MEM0_LLM_MODEL", "anthropic/claude-3-5-sonnet-20241022")
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
```

- [ ] **Step 2: 核实确切配置键(关键)**

Run:
```bash
conda activate sft && python -c "from mem0.configs.llms.base import BaseLlmConfig; import inspect; print(inspect.signature(BaseLlmConfig.__init__))"
python -c "import mem0; help(mem0.Memory.from_config)" 2>&1 | head -20
```
用输出确认 llm/embedder/vector_store 三块的确切键名与嵌套结构。若与下面 Step 3 的假设不符,以实际输出为准修正 Step 3。

- [ ] **Step 3: 写实现 `memory_framework/config.py`**

```python
import os

DEFAULT_LLM_MODEL = "anthropic/claude-3-5-sonnet-20241022"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIMS = 384


class MissingConfigError(RuntimeError):
    pass


def build_config() -> dict:
    token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if not token:
        raise MissingConfigError("缺少环境变量 ANTHROPIC_AUTH_TOKEN")
    if not base_url:
        raise MissingConfigError("缺少环境变量 ANTHROPIC_BASE_URL")
    model = os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)

    return {
        "llm": {
            "provider": "litellm",
            "config": {
                "model": model,
                "api_key": token,
                "api_base": base_url,
                "temperature": 0.1,
                "max_tokens": 1024,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": EMBED_MODEL, "embedding_dims": EMBED_DIMS},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {"path": "./qdrant_data", "embedding_model_dims": EMBED_DIMS},
        },
    }
```

- [ ] **Step 4: 跑测试**

Run: `conda activate sft && cd /Users/mashiro/Mem0 && pytest tests/test_config.py -v`
Expected: 3 个测试 PASS

- [ ] **Step 5: Commit**

```bash
git add memory_framework/config.py tests/test_config.py
git commit -m "feat: config module with litellm+bge+qdrant"
```

---

### Task 3: 核心封装 `memory_store.py`

**Files:**
- Create: `memory_framework/memory_store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Consumes: `build_config()` from `config.py`
- Produces: `MemoryStore` 类,方法签名:
  - `add(messages: list[dict], user_id: str) -> dict`
  - `search(query: str, user_id: str, limit: int = 5) -> list`
  - `get_all(user_id: str) -> list`
  - `update(memory_id: str, data: str) -> dict`
  - `delete(memory_id: str) -> None`
  - `delete_all(user_id: str) -> None`

- [ ] **Step 1: 写真端到端失败测试 `tests/test_memory_store.py`**

```python
import pytest
from memory_framework.memory_store import MemoryStore


@pytest.fixture(scope="module")
def store():
    s = MemoryStore()
    s.delete_all(user_id="pytest_user")
    yield s
    s.delete_all(user_id="pytest_user")


def test_add_then_search_recalls(store):
    store.add(
        messages=[{"role": "user", "content": "我不吃香菜,对海鲜过敏"}],
        user_id="pytest_user",
    )
    results = store.search(query="这个人的饮食禁忌是什么", user_id="pytest_user", limit=5)
    joined = " ".join(str(r) for r in results)
    assert "香菜" in joined or "海鲜" in joined


def test_get_all_returns_list(store):
    store.add(messages=[{"role": "user", "content": "我住在上海"}], user_id="pytest_user")
    allm = store.get_all(user_id="pytest_user")
    assert isinstance(allm, list) and len(allm) >= 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda activate sft && pytest tests/test_memory_store.py -v`
Expected: FAIL — `ModuleNotFoundError` / `MemoryStore` 未定义

- [ ] **Step 3: 写实现 `memory_framework/memory_store.py`**

```python
from mem0 import Memory
from memory_framework.config import build_config


class MemoryStore:
    def __init__(self):
        self._memory = Memory.from_config(build_config())

    def add(self, messages: list[dict], user_id: str) -> dict:
        return self._memory.add(messages, user_id=user_id)

    def search(self, query: str, user_id: str, limit: int = 5) -> list:
        res = self._memory.search(query, user_id=user_id, limit=limit)
        return res.get("results", res) if isinstance(res, dict) else res

    def get_all(self, user_id: str) -> list:
        res = self._memory.get_all(user_id=user_id)
        return res.get("results", res) if isinstance(res, dict) else res

    def update(self, memory_id: str, data: str) -> dict:
        return self._memory.update(memory_id=memory_id, data=data)

    def delete(self, memory_id: str) -> None:
        self._memory.delete(memory_id=memory_id)

    def delete_all(self, user_id: str) -> None:
        self._memory.delete_all(user_id=user_id)
```

- [ ] **Step 4: 跑测试(真实调用 Claude+embedding,首次会下载 bge 模型)**

Run: `conda activate sft && cd /Users/mashiro/Mem0 && pytest tests/test_memory_store.py -v -s`
Expected: 2 个测试 PASS(首次运行较慢,含模型下载)。若 `search`/`get_all` 返回结构与假设不符,依据实际返回修正 Step 3 的解包逻辑。

- [ ] **Step 5: Commit**

```bash
git add memory_framework/memory_store.py tests/test_memory_store.py
git commit -m "feat: MemoryStore wrapper over mem0"
```

---

### Task 4: 对话数据准备 `conversation.py` + 演示数据

**Files:**
- Create: `data/conversations/alice.json`
- Create: `data/conversations/bob.json`
- Create: `memory_framework/conversation.py`
- Test: `tests/test_conversation.py`

**Interfaces:**
- Consumes: `MemoryStore` from `memory_store.py`
- Produces:
  - `load_conversation(path: str) -> dict`(返回 `{"user_id": str, "messages": list}`,校验结构)
  - `ingest(store, conv: dict) -> dict`(调 `store.add`)
  - `InvalidConversationError`(异常类)

- [ ] **Step 1: 写演示数据 `data/conversations/alice.json`**

```json
{
  "user_id": "alice",
  "messages": [
    {"role": "user", "content": "我最近在学做菜,尤其想学川菜"},
    {"role": "assistant", "content": "好的,记住你在学川菜"},
    {"role": "user", "content": "对了我不吃香菜,而且对海鲜过敏"},
    {"role": "user", "content": "我周末一般在家,喜欢一边做饭一边听爵士乐"}
  ]
}
```

- [ ] **Step 2: 写演示数据 `data/conversations/bob.json`**

```json
{
  "user_id": "bob",
  "messages": [
    {"role": "user", "content": "我是一名后端工程师,主力语言是 Go"},
    {"role": "assistant", "content": "了解,你用 Go 做后端"},
    {"role": "user", "content": "最近在读 React,前端对我是新领域"},
    {"role": "user", "content": "我习惯早上 6 点起床跑步 5 公里"}
  ]
}
```

- [ ] **Step 3: 写失败测试 `tests/test_conversation.py`**

```python
import json
import pytest
from memory_framework.conversation import (
    load_conversation, InvalidConversationError,
)


def test_load_valid(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"user_id": "u1", "messages": [
        {"role": "user", "content": "hi"}]}), encoding="utf-8")
    conv = load_conversation(str(p))
    assert conv["user_id"] == "u1"
    assert conv["messages"][0]["content"] == "hi"


def test_load_missing_user_id_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"messages": []}), encoding="utf-8")
    with pytest.raises(InvalidConversationError):
        load_conversation(str(p))


def test_load_messages_not_list_raises(tmp_path):
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps({"user_id": "u", "messages": "nope"}), encoding="utf-8")
    with pytest.raises(InvalidConversationError):
        load_conversation(str(p))
```

- [ ] **Step 4: 跑测试确认失败**

Run: `conda activate sft && pytest tests/test_conversation.py -v`
Expected: FAIL — `conversation` 模块/函数未定义

- [ ] **Step 5: 写实现 `memory_framework/conversation.py`**

```python
import json


class InvalidConversationError(ValueError):
    pass


def load_conversation(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "user_id" not in data or not isinstance(data["user_id"], str):
        raise InvalidConversationError(f"{path}: 缺少或非法的 user_id")
    if "messages" not in data or not isinstance(data["messages"], list):
        raise InvalidConversationError(f"{path}: messages 必须是列表")
    for m in data["messages"]:
        if "role" not in m or "content" not in m:
            raise InvalidConversationError(f"{path}: message 缺 role/content")
    return data


def ingest(store, conv: dict) -> dict:
    return store.add(messages=conv["messages"], user_id=conv["user_id"])
```

- [ ] **Step 6: 跑测试**

Run: `conda activate sft && cd /Users/mashiro/Mem0 && pytest tests/test_conversation.py -v`
Expected: 3 个测试 PASS

- [ ] **Step 7: Commit**

```bash
git add data/conversations/ memory_framework/conversation.py tests/test_conversation.py
git commit -m "feat: conversation loader and demo datasets"
```

---

### Task 5: 端到端演示 `demo.py`

**Files:**
- Create: `examples/demo.py`

**Interfaces:**
- Consumes: `MemoryStore`, `load_conversation`, `ingest`
- Produces: 可运行的演示脚本

- [ ] **Step 1: 写 `examples/demo.py`**

```python
"""端到端演示:灌入对话 → 提问 → 语义召回。"""
import glob
from memory_framework.memory_store import MemoryStore
from memory_framework.conversation import load_conversation, ingest


def main():
    store = MemoryStore()
    for path in glob.glob("data/conversations/*.json"):
        conv = load_conversation(path)
        print(f"灌入 {conv['user_id']} 的对话...")
        ingest(store, conv)

    print("\n--- 检索演示 ---")
    q1 = store.search("这个人有什么饮食禁忌", user_id="alice")
    print("alice 饮食禁忌:", q1)
    q2 = store.search("他的技术背景是什么", user_id="bob")
    print("bob 技术背景:", q2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行演示验证**

Run: `conda activate sft && cd /Users/mashiro/Mem0 && python examples/demo.py`
Expected: 打印灌入日志,并在 alice 结果里出现"香菜/海鲜"、bob 结果里出现"Go/后端"相关记忆。

- [ ] **Step 3: Commit**

```bash
git add examples/demo.py
git commit -m "feat: end-to-end demo script"
```

---

### Task 6: 研究指导文档 `README.md`

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 全部已完成模块的实际接口
- Produces: 使用文档

- [ ] **Step 1: 写 `README.md`**

内容需覆盖:
1. 项目简介(长期记忆框架,基于 Mem0)
2. 架构图(config → MemoryStore → conversation/demo 的依赖关系)
3. 安装步骤(`conda activate sft` → `pip install -r requirements.txt` → 复制 `.env.example` 为 `.env` 并填 `ANTHROPIC_*`)
4. 快速上手(3 行代码:`MemoryStore()` → `add` → `search`,附实际输出示例)
5. Mem0 配置要点与避坑:
   - litellm 转 Claude 的 provider/model 写法(Task 2 实测确认的确切键)
   - bge-small-zh 首次运行会下载模型(约 100MB),可用 `HF_ENDPOINT=https://hf-mirror.com` 加速
   - Qdrant 本地模式数据存 `./qdrant_data/`,删掉即清空
   - `user_id` 是记忆隔离的关键,务必传
6. 运行测试的命令
7. Mem0 官方文档链接:https://docs.mem0.ai

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: usage guide and Mem0 integration notes"
```

---

## 并行开发映射(Workflow)

实现阶段可用 workflow 多 Agent 并行,依赖关系:

- **Task 1**(骨架/依赖)必须先跑 —— 所有任务的前置。
- **Task 2**(config)是核心前置,Task 3 依赖它。
- **Task 4 数据部分**(Step 1-2 写 JSON)、**Task 6**(README 初稿)无代码依赖,可与 Task 2/3 并行起草。
- **Task 3**(MemoryStore)完成后,**Task 4 的 ingest 测试**、**Task 5 demo**、**Task 6 的接口示例**才能跑通验证。

推荐执行顺序:Task 1 → (Task 2 ∥ Task4-数据 ∥ Task6-初稿) → Task 3 → (Task 4-验证 ∥ Task 5) → Task 6-定稿。
