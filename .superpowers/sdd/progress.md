# Mem0 记忆框架 — 进度账本

计划:docs/superpowers/plans/2026-07-04-mem0-memory-framework.md
执行方式:subagent-driven-development → 后续用户手动大幅推进
Conda 环境:**mem0**(Python 3.11.15)

## 真实进度(以 git log + 代码为准,2026-07-08 重新对齐)

原始 6 任务记忆框架 + 结构化画像子系统均已完成并提交。git 历史:
- b990af5 设计spec / c1b9859 计划
- 1562e9c 骨架依赖 / 538691a config / 647a802 config重构
- ea260c2 修复4个真实bug + MemoryStore落地
- 5e96eaf 对话加载/demo/CLI聊天/README (Task4-6)
- d408b79 画像设计spec / 2a55b3d 画像计划
- 9b23e46 画像核心(数据结构+规则打分+时间衰减遗忘)
- 5c596a1 chat每轮增量更新画像
- f276276 Gradio四维度画像展示

## 当前功能全景
- 记忆层:config(litellm→京东云中转站Opus / bge-small-zh 512维 / qdrant本地)、MemoryStore封装、conversation加载
- 画像层:profile数据结构(4维+evidence)、profile_store(replace/ingest/遗忘)、scoring(打分+时间衰减)、profile_extractor(LLM提炼)
- 应用层:chat CLI、app.py Gradio界面
- 数据:data/conversations、data/testsets、profiles/

## 已验证(2026-07-08)
- 48 个非API测试全绿
- profile_extractor 真实调中转站验证通过:Claude 正确从事件推断性格(跑马→坚毅自律 / 学三门语言→求知欲强),禁忌高权重(海鲜过敏10/香菜9)

## 关键约定
- config.py 已由用户实测调优(勿覆盖):Opus模型 / EMBED_DIMS=512 / temperature=1 / custom_instructions中文 / litellm.drop_params=True
- 创建 Memory 前必须先 apply_litellm_env() 或用 build_config_and_apply_env()

## 项目开发管理助手(2026-07-19 完成,6 层)
计划:.claude/plans/reactive-percolating-crown.md
把 Mem0 扩成「项目开发管理助手」:结构分析 + 代码深度分析双模式 + 项目长期记忆 + Claude Code 集成(MCP + Skill)。

- Layer 1 `code_analyzer.py`:逐文件 LLM 摘要 + 纯 ast import 依赖图 + 归纳报告。成本护栏 MAX_FILES=120/FILE_BATCH=4/增量。产物 code_analysis/<id>.{md,json}。
- Layer 2 `code_snapshot.py`:sha256(mtime+size+前8k)哈希快照 → diff_snapshot 出 added/changed/removed,驱动增量重分析。
- Layer 3 `project_chat.py`:项目 Q&A,记忆命名空间 user_id=`proj::<id>`,注入进度四维+代码摘要;sediment_analysis/sediment_change 沉淀分析与变更。
- Layer 4 `mcp_server/server.py`:FastMCP 6 只读工具(list_projects/get_project_memory/get_progress/get_code_analysis/search_code_analysis/get_structure_analysis)。**承重约束:只读磁盘产物,绝不开 Qdrant**(单进程独占,与 Gradio 冲突)。.mcp.json 已生成。
- Layer 5 `skill_export/exporter.py`:export_skill 生成 .claude/skills/<id>/SKILL.md + reference/*.md。
- Layer 6 app.py「🛠 项目工作台」Tab:深度分析(全量/增量)+ 项目 Q&A 聊天 + 导出 Skill 按钮;repo_scan._IGNORE_DIRS 加 code_analysis/.claude。

### 已验证(2026-07-19)
- 122 个非 API 测试全绿(含新增 40:code_analyzer 13 / code_snapshot 5 / project_chat 8 / mcp_server 11 / skill_export 3)。
- 真实 LLM 深度分析 demo 项目:import 图 main→pkg.util 正确、3 摘要、1 次批调用、md 报告生成、save/load 回环 OK。
- MCP 6 工具注册成功;FastMCP server 可构造。
- Gradio 重启探测 7860 → 200,工作台 Tab 生效。
- 注册 MCP:`claude mcp add mem0-project -- conda run -n mem0 python -m mcp_server.server`。

### 关键约定(新增)
- MCP server 永不构造 MemoryStore(test_mcp_server 用 monkeypatch 强制断言)。
- 本 Gradio 版本 gr.Chatbot 不接受 type= 参数(默认已兼容 message-dict)。
- mcp>=1.2.0 已加入 requirements 并 pip 装入 mem0 环境(实装 1.28.1)。

## 项目条目管理(2026-07-20 开始,SDD)
计划:docs/superpowers/plans/2026-07-20-project-item-management.md
分支:main;起始 HEAD: deb2d95
7 任务:ProfileItem 加 locked/source → replace_profile 透传 → ProjectStore set_locked/add_item → track_from_dialog 保护锁定 → 提示词聚焦 → UI @gr.render 重构 → E2E。
Task 1: complete (commits deb2d95..5000370, review clean)
Task 2: complete (commits 5000370..3ae1b26, review clean)
Task 3: complete (commits 3ae1b26..86082f6, review clean; minors: add_item return-annotation cosmetic, set_locked loops-all intentional)
Task 4: complete (commits 86082f6..1d144e4, review clean; deviation: 按维度替换替代 exact-text keying, adjudicated correct. minor: bare subscript style)
Task 5: complete (commit 8be53e6, review clean; minor: 12/20 substring assertion loose)
Task 6: complete (commits 8be53e6..3d3e6e6, review Approved; agent cut off mid-task, controller finished: demo.load state wiring + removed _render_project + fixed pre-existing _render_analysis NameError; minor _WRITE_LOCK on mutations fixed in 3d3e6e6)
Task 7: complete (183 tests pass; E2E lock-survival PASS via mock-LLM full-rerun on temp store; git clean). Push deferred pending user consent. In-browser visual walkthrough left to user.
Task 7: + final-review Minor#1 fixed (note→status-bar detail) in 3c65c26
