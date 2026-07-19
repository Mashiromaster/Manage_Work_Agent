# Mem0 长期记忆框架 — 设计文档

日期:2026-07-04
状态:待用户审阅

## 1. 目标与范围

基于 [Mem0](https://github.com/mem0ai/mem0) 搭建一个**长期记忆框架**,让 AI 应用能跨会话记住用户的事实、偏好与历史,并在需要时语义召回。

### 核心能力
- **写入记忆** — 从对话中由 Claude 自动提取关键事实并存储
- **检索记忆** — 语义向量检索,给定新问题召回相关历史记忆
- **记忆管理** — 增/删/改/查,按 `user_id` 隔离不同用户

### 技术栈
| 组件 | 选型 | 理由 |
|------|------|------|
| LLM(记忆提取) | Claude,经 **litellm** 转接 | 提取质量高;复用环境已有的 `ANTHROPIC_BASE_URL`/`TOKEN`;Mem0 官方无 Anthropic 一等支持,litellm 是最稳的转接方式 |
| Embedding | HuggingFace **bge-small-zh-v1.5** | 中文检索友好;纯本地离线;约 100MB |
| 向量库 | **Qdrant 本地模式** | Mem0 默认;零额外服务,pip 装完即用 |

除 Claude API 外全部本地运行。

### 范围外(YAGNI)
Web UI、多模态记忆、生产级分布式部署、用户认证系统。

## 2. 架构与模块划分

目录结构(根目录 `/Users/mashiro/Mem0`):

```
Mem0/
├── memory_framework/          # 核心包
│   ├── __init__.py
│   ├── config.py              # 统一配置:litellm→Claude、bge embedder、Qdrant
│   ├── memory_store.py        # 对 Mem0 Memory 的薄封装
│   └── conversation.py        # 对话数据载入与批量灌入
├── data/conversations/        # 演示对话数据(JSON)
├── tests/
│   ├── test_config.py
│   ├── test_memory_store.py
│   └── test_conversation.py
├── examples/demo.py           # 端到端演示
├── docs/superpowers/specs/    # 本设计文档
├── requirements.txt
├── .env.example
└── README.md
```

### 模块职责与依赖
| 模块 | 职责 | 依赖 |
|------|------|------|
| `config.py` | 构造 Mem0 配置 dict(LLM/embedder/vector_store 三块),从环境变量读密钥。**唯一配置来源** | 环境变量 |
| `memory_store.py` | `MemoryStore` 类,封装 `mem0.Memory`,暴露干净接口,隐藏 Mem0 内部细节 | `config.py`, `mem0` |
| `conversation.py` | 读对话 JSON,规整成 Mem0 messages 格式,批量调 `MemoryStore.add` | `memory_store.py` |
| `demo.py` | 串起全流程演示 | 上述三者 |

### 关键取舍
用 `MemoryStore` 薄封装隔离 Mem0:未来换向量库或 LLM 只改 `config.py`,业务代码与测试不动。

## 3. 数据流与接口

### 写入流程
```
对话JSON → conversation.load() → [{role,content},...]
  → MemoryStore.add(messages, user_id)
  → Mem0:Claude 提取事实 → bge 向量化 → 存 Qdrant
```

### 检索流程
```
新问题 → MemoryStore.search(query, user_id)
  → bge 向量化 → Qdrant 相似度检索 → 返回 top-k 记忆
  → (可选)拼进 prompt 给 Claude 回答
```

### MemoryStore 对外接口(冻结)
```python
store = MemoryStore()                              # 内部读 config
store.add(messages, user_id)          -> dict      # 存,返回提取出的记忆
store.search(query, user_id, limit=5) -> list      # 语义检索
store.get_all(user_id)                -> list      # 某用户全部记忆
store.update(memory_id, data)         -> dict      # 改
store.delete(memory_id)               -> None      # 删
store.delete_all(user_id)             -> None      # 清空某用户
```

### 对话数据格式(`data/conversations/*.json`)
```json
{
  "user_id": "alice",
  "messages": [
    {"role": "user", "content": "我最近在学做菜,尤其想学川菜"},
    {"role": "assistant", "content": "好的,记住你在学川菜"},
    {"role": "user", "content": "对了我不吃香菜"}
  ]
}
```

### 错误处理
只在系统边界校验:(1) `config.py` 启动时检查必需环境变量,缺失抛清晰错误;(2) `conversation.py` 校验 JSON 结构。内部模块间信任,不做冗余校验。

### 测试策略
- `config.py` / `conversation.py`:纯逻辑,单测直接验证。
- `memory_store.py`:小规模**真端到端**测试(存 2-3 条 → 检索 → 断言召回),**不 mock** 记忆提取——mock 掉就等于没测到核心价值。

## 4. 并行开发编排(Workflow)

实现阶段用 workflow 多 Agent 并行,对应用户提出的四条线。依赖关系决定并行度:

- **Agent 1 — 对话数据准备**:写 `data/conversations/*.json` 演示数据。**无依赖,可立即并行**。
- **Agent 2 — 框架搭建**:`config.py` + `memory_store.py` + `conversation.py` + `requirements.txt` + `.env.example`。**核心,其他依赖它的接口**。
- **Agent 3 — 测试搭建**:`tests/*`。依赖 Agent 2 的接口定义(接口已在本 spec 冻结,故可基于 spec 并行起草,Agent 2 完成后跑通)。
- **Agent 4 — Mem0 研究指导**:产出 `README.md` + 使用指南 + Mem0 配置要点/避坑。**无依赖,可立即并行**(基于 Mem0 官方文档研究)。

实际编排:Agent 1、2、4 首轮并行;Agent 3 的测试代码可并行起草,但"跑通验证"这一步排在 Agent 2 之后。具体计划由 writing-plans 阶段细化。
