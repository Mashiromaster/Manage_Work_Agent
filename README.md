# Manage_Work_Agent · 项目开发管理助手

基于 [Mem0](https://docs.mem0.ai) 的中文**项目开发管理助手**:为每个项目维护一份长期记忆,提供**结构分析**与**代码深度分析**两种模式,追踪开发进度,并能把项目记忆通过 **MCP + Skill** 暴露给 Claude Code,让 AI 对项目上下文了如指掌。

- **LLM**:Claude,经 litellm 走中转站。
- **Embedding**:本地 `BAAI/bge-small-zh-v1.5`(512 维),无需联网。
- **向量库**:Qdrant 本地文件模式,数据存 `./qdrant_data/`。
- **界面**:Gradio 可视化(`app.py`),默认 http://127.0.0.1:7860。

## 功能全景

四个 Tab,顶部一个**全局项目选择器**统一驱动其中三个:

1. **💬 记忆聊天 · 个性定制**
   - 全局**人设 / 系统提示词**文本框 + **注入开关**(决定对话是否带入个性化上下文)。
   - 逐条长期记忆**删除 / 编辑**,按 `user_id` 隔离。
2. **📊 项目进度追踪**
   - 从 Claude Code 对话日志(`~/.claude/projects`)提炼**进度 / 问题 / 待办 / 决策**四维度,支持增量重跑。
3. **🗂 项目结构分析**
   - 扫描项目目录,LLM 归纳**项目概述 / 技术栈 / 模块划分 / 目录结构**。
4. **🛠 项目工作台**
   - **代码深度分析**:逐文件职责摘要 + `ast` import 依赖图 + 关键路径,产出完整报告(全量 / 增量)。
   - **项目问答**:自动注入进度四维 + 代码结构 + 历史记忆作答。
   - **导出 Skill**:一键生成 `.claude/skills/<项目>/`,供 Claude Code 加载。

## Claude Code 集成(MCP + Skill)

**MCP server**(`mcp_server/server.py`)以只读方式暴露项目 / 用户长期记忆,共 8 个工具:
`list_projects` / `get_project_memory` / `get_progress` / `get_code_analysis` /
`search_code_analysis` / `get_structure_analysis` / `list_users` / `get_user_persona`。

> ⚠️ **设计承重点**:MCP server **只读磁盘产物、绝不打开 Qdrant**(本地文件模式单进程独占,否则与 Gradio 争锁)。故 MCP 侧暴露的是进度四维、代码分析、人设 + 画像等**已落盘**内容,不含 Qdrant 里的逐条原始记忆。

注册到 Claude Code(仓库内已带 `.mcp.json`,进项目目录批准即可;或手动注册):

```bash
claude mcp add mem0-project -- \
  /opt/homebrew/Caskroom/miniforge/base/envs/mem0/bin/python -m mcp_server.server
```

> 注意用 **conda 环境的 python 直连**,而非 `conda run`——后者会缓冲 stdio,破坏 MCP 的 JSON-RPC 握手。

## 安装

```bash
conda activate mem0                    # Python 3.11 环境
pip install -r requirements.txt        # 含 mem0ai / litellm / qdrant-client / mcp
cp .env.example .env                   # 然后填入真实 token
```

`.env`(已 gitignore,勿提交):

```
ANTHROPIC_BASE_URL=https://modelservice.jdcloud.com/anthropic
ANTHROPIC_AUTH_TOKEN=pk-xxxx           # 你的中转站 key
MEM0_LLM_MODEL=anthropic/Claude-Opus-4.8-hq
```

## 运行

```bash
# Gradio 界面(推荐);离线标志避免 litellm/HF 联网卡顿
LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. python app.py
# → 浏览器打开 http://127.0.0.1:7860

# CLI 记忆聊天(/mem 看记忆,/quit 退出)
PYTHONPATH=. python -m memory_framework.chat --user alice

# MCP server(stdio)
PYTHONPATH=. python -m mcp_server.server
```

## 运行测试

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. \
  conda run -n mem0 python -m pytest tests/ -q
```

> `test_memory_store.py` 需独占 Qdrant——app 运行时请先停掉再跑,或加 `--ignore=tests/test_memory_store.py`。

## 目录结构

```
app.py                       # Gradio 界面:四 Tab + 全局项目选择器
memory_framework/
  config.py                  # 配置 + litellm 环境桥接
  memory_store.py            # MemoryStore:mem0 薄封装(增删改查)
  chat.py                    # 带记忆的聊天(人设 + 注入开关)
  persona_store.py           # 用户人设 / 注入开关(persona/<uid>.json)
  profile_store.py           # 结构化画像存储
  conversation.py            # 对话 JSON 加载
  cc_ingest.py               # Claude Code 日志解析(含路径还原)
  project_tracker.py         # 进度四维追踪编排
  project_extractor.py       # 进度四维 LLM 提炼(带重试)
  repo_scan.py / repo_analyzer.py  # 目录扫描 + 结构分析
  code_analyzer.py           # 代码深度分析(逐文件摘要 + ast 依赖图 + 重试/降级)
  code_snapshot.py           # 文件哈希快照,驱动增量分析
  project_chat.py            # 项目问答(proj::<id> 命名空间)
mcp_server/server.py         # MCP:8 个只读工具,绝不开 Qdrant
skill_export/exporter.py     # 导出 Claude Code Skill
tests/                       # pytest 测试
```

## Mem0 集成避坑要点(实测记录)

1. **embedding 维度必须匹配模型真实输出。** `bge-small-zh-v1.5` 是 **512 维**;`config.py` 的 `EMBED_DIMS` 与 Qdrant 建集合维度须一致,换模型要清空 `qdrant_data/`。
2. **litellm 不接受 `api_base`。** 中转站基址桥接到环境变量 `ANTHROPIC_API_BASE`(见 `config.apply_litellm_env()`)。
3. **Opus 4.x 只接受 `temperature=1`。** `apply_litellm_env()` 设 `litellm.drop_params = True` 自动丢弃不支持的采样参数。
4. **默认会把中文记忆抽成英文。** 通过 `custom_instructions` 注入中文指令强制中文存储。
5. **`user_id` 是隔离键**;mem0 2.0.11 用 `filters={"user_id": ...}` + `top_k`(非顶层 `user_id=` / `limit`),差异已在 `MemoryStore` 抹平。
6. **Qdrant 本地文件模式单进程独占**:Gradio 与 MCP 不能同时开它——MCP 因此设计为纯文件只读。
7. **LLM 调用带重试与降级**:`code_analyzer` / `project_extractor` 对限流(429)做指数退避;终归纳失败时给降级报告而非丢弃已得结果。

参考:[Mem0 官方文档](https://docs.mem0.ai)
