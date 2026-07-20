# 项目进度追踪:聚焦粒度 + 条目级增删/锁定/手动添加

## Context(为什么做)

现状(截图 #28/#29/#30):四维面板内容**太细散**——把每次小 bug 修复、变量改名、临时验证都单列成条(progress 78 条、blocker 27、todo 35、decision 55)。用户要的是**项目开发进度管理**视图,不是流水账。

两个诉求:
1. **聚焦粒度**:提取只保留功能级进展与重要决策,允许少量关键 bug(阻断发布/全局影响),不记琐碎修复;四维合计约 **12–20 条**。
2. **条目可管控**:每条支持 🔒锁定 / 🗑删除,每维支持手动 **+ 添加**;锁定与手动条目在重新导入/全量重跑时**不被 LLM 覆盖**。

**已确认决策**:
- 聚焦粒度 = 功能级 + 关键 bug(约 12–20 条)。
- 锁定语义 = 不被覆盖:重导入时锁定条目 + 手动条目原样保留,LLM 结果与它们合并去重;只有**未锁定的 LLM 条目**会被新一轮提炼替换。
- UI = 重构为 Gradio 原生组件(每条一行带按钮 + 每维底部添加框),替换现有纯 HTML 面板。

## 已探明(复用点与约束)

- `ProfileItem`(`profile.py`)**无独立 id**,历史按 `(dimension, text)` 定位;`ProjectStore.delete_item(pid, dim, text)` 已存在(全量按 `(dim,text)` 删)。
- `ProjectStore.replace_profile(pid, items_data)` 全量替换、不遗忘;`get_project` 按 importance 降序、不过滤。
- `_render_project`(`app.py:420`)输出**纯 HTML 字符串**,无法挂 Gradio 事件——这是重构的核心。
- `track_from_dialog`(`project_tracker.py:141`):增量路径已做 `(dim, norm(text))` 合并;**全量路径 `items_data = fresh` 会清空一切**——锁定/手动条目会在全量重跑时丢失,必须修。
- `DEFAULT_DIM_PROMPT` / `DIM_MERGE_PROMPT`(`persona_store.py`)是提取/合并的系统提示词,聚焦粒度改这里。
- `_norm`(`project_tracker.py:227`)= 去空白 + 去尾标点,用于去重键。

## 架构

### 1) 数据模型:给 ProfileItem 加 `locked` 与 `source`(`profile.py`)

`ProfileItem` 新增两字段(都带默认值,向后兼容旧 JSON):
- `locked: bool = False` —— 用户锁定,重导入不覆盖。
- `source: str = "llm"` —— `"llm"`(提炼产出)/ `"manual"`(手动添加)。手动条目视同锁定语义(永不被 LLM 覆盖),但用 source 区分展示与来源标注。

`to_dict`/`from_dict` 带上这两个字段;`from_dict` 用 `.get(..., 默认)` 读,旧文件缺字段自动回落。`replace_profile`(`profile_store.py`)在构造 `ProfileItem` 时透传 `d.get("locked", False)` 与 `d.get("source", "llm")`,使合并写盘保留锁定态。

**条目身份**:仍用 `(dimension, text)` 作为逻辑主键(delete/lock 的定位键)。不引入随机 id——避免 schema 迁移,且四维条目文本天然唯一(同维度重复文本本就该去重)。删除/锁定按 `(dimension, text)` 精确匹配。

### 2) 存储层:锁定/添加/删除(`profile_store.py`,加在 ProjectStore 或基类)

- `delete_item(pid, dim, text)` —— 已有,直接用。
- `set_locked(pid, dim, text, locked: bool) -> bool` —— 读盘、按 `(dim,text)` 找到条目、置 `locked`、写盘;找不到返回 False。
- `add_item(pid, dim, text, importance=6) -> UserProfile` —— 追加一条 `source="manual", locked=True` 的条目(手动加的默认锁定,天然不被覆盖),写盘返回新 profile。文本去空白后为空则忽略。同 `(dim,text)` 已存在则不重复添加(更新为 manual+locked)。

这三个都是纯 JSON 读写,可单测,MCP 只读不受影响。

### 3) 提取编排:锁定/手动条目永不被覆盖(`project_tracker.py`)

`track_from_dialog` 改动(增量与全量都要保护锁定/手动条目):

```
existing_items = [it.to_dict() for it in existing.items]   # 两种模式都读
preserved = [it for it in existing_items if it.get("locked") or it.get("source")=="manual"]
preserved_keys = {(it["dimension"], _norm(it["text"])) for it in preserved}

# fresh = LLM 提炼结果;丢掉与 preserved 同键的(锁定条目优先,不被 LLM 版本替换)
fresh = [it for it in fresh if (it.get("dimension"), _norm(it.get("text",""))) not in preserved_keys]

if incremental:
    # 旧的“未锁定 llm 条目”允许被 fresh 覆盖;preserved 始终保留
    merged = {(k): it for it in unlocked_old}      # 旧未锁定
    for it in fresh: merged[key(it)] = it            # 新提炼覆盖
    items_data = preserved + list(merged.values())
else:  # 全量重跑:丢弃旧的未锁定 llm 条目,但保留 preserved
    items_data = preserved + fresh
```

要点:全量重跑不再是 `items_data = fresh`,而是 `preserved + fresh`。`llm_empty`(提炼空)时仍旧不覆盖(维持现语义)。给提取喂的 `existing_items` 增量时仍传全量(让 LLM 见过历史),但**锁定条目从提炼上下文里不排除**(排除只在“不被覆盖”层面做,避免 LLM 重复产出锁定项——由 preserved_keys 去重兜底)。

### 4) 提示词聚焦(`persona_store.py`:`DEFAULT_DIM_PROMPT` + `DIM_MERGE_PROMPT`)

在两段提示词的“核心原则/合并规则”里强化:
- **只记项目开发进度管理层面的内容**:功能/模块级进展、里程碑、架构与技术选型决策、当前仍未解决且**影响开发推进**的阻塞。
- **不记琐碎修复**:单个 bug 的修复、变量改名、加日志、格式调整、临时验证、措辞微调——除非该 bug 属于**关键阻塞**(阻断发布、影响全局),否则不单列;可并入相关功能的 progress 作一句带过。
- **数量**:四维合计约 **12–20 条**(比现在的 10–25 更收敛),追求信息密度。importance 指南不变。

`test_persona_store.py` 现以符号引用 `DEFAULT_DIM_PROMPT`,文本改动不破测试;补一条断言新提示词含关键约束词(如“不记琐碎”“功能级”)。

### 5) UI 重构:HTML 面板 → Gradio 原生四维(`app.py`)

替换 `_render_project` 的纯 HTML 渲染为**动态组件**。Gradio 里“数量随数据变化的按钮列表”用 `gr.State + @gr.render` 实现(Gradio 4+/6 支持 `@gr.render` 动态渲染)。

结构:
- 一个 `proj_items_state = gr.State([])` 存当前项目的条目列表(list[dict])。
- `@gr.render(inputs=[g_project, proj_items_state])` 装饰的函数按四维分组渲染:
  - 每维一个卡片(标题 + 计数),维持现有配色。
  - 每条 = `gr.Row`:文本(`gr.Markdown`,锁定条目前缀 🔒、手动条目标 ✎)+ importance 徽标 + `🔒`按钮(toggle)+ `🗑`按钮。按钮 `.click` 调 `set_locked`/`delete_item` 后刷新 state → 重渲染。
  - 每维底部一个 `gr.Row`:`gr.Textbox`(占位“手动添加…”)+ `gr.Button("+ 添加")`,click 调 `add_item(pid, dim, text)` 刷新 state。
- **数据流**:所有增删/锁定/添加回调 = 改盘(store)→ 重新 `get_project(pid)` → 更新 `proj_items_state` → `@gr.render` 自动重画。导入完成后也把结果写入 `proj_items_state`(替代现在返回 HTML)。
- 保留顶部状态栏(`proj_status`)与导入/刷新/全量重跑按钮,行为不变;`on_import_project` 最终输出改为更新 `proj_items_state`(而非 HTML)。

**兼容成本**:`_render_project` 被 `@gr.render` 函数取代;凡是把 `proj_view`(HTML)当输出的接线(切项目 `_switch_project`、`demo.load`、刷新)都改成输出 `proj_items_state`。`_render_code_analysis` 等其他 Tab 不动。

### 涉及文件
- 改:`memory_framework/profile.py`(+locked/+source)、`memory_framework/profile_store.py`(+set_locked/+add_item,replace_profile 透传字段)、`memory_framework/project_tracker.py`(preserved 保护)、`memory_framework/persona_store.py`(两段提示词聚焦)、`app.py`(四维 UI 重构 + 接线)。
- 测试补:`tests/test_profile_store.py`(set_locked/add_item/replace 保留字段)、`tests/test_project_tracker.py`(锁定/手动条目在增量+全量下都保留、未锁定 llm 被覆盖)、`tests/test_persona_store.py`(提示词含聚焦约束词)。

## 边界 / 兼容
- 旧 project_tracking/*.json 无 locked/source 字段 → `from_dict` 默认 False/"llm",照常读。
- 锁定条目 + 手动条目在**增量与全量**两条路径都不被覆盖(全量重跑修掉现有 `items_data=fresh` 的丢失 bug)。
- `llm_empty`(提炼返回空)时不覆盖任何旧记录(维持现语义)。
- 删除/锁定按 `(dimension, text)` 定位;若极端情况下同维度文本重复,delete_item 会删掉所有同文本项(可接受,重复本就该合并)。
- MCP server 只读磁盘产物,不受影响;新字段是 JSON 里多两个 key。
- 不引入随机 id、不做 schema 迁移脚本(默认值向后兼容即可)。

## 验证(端到端)
1. 单测(mock `_complete`),`pytest tests/ -q --ignore=tests/test_memory_store.py` 全绿:
   - profile_store:set_locked 置位/清位、add_item 追加 manual+locked、replace_profile 保留 locked/source。
   - project_tracker:锁定条目在 incremental=True 合并后仍在;手动条目在 incremental=False 全量重跑后仍在;未锁定 llm 条目被 fresh 覆盖;llm_empty 不动盘。
   - persona_store:load_dim_prompt 默认含聚焦约束词。
2. 重启 app 探测 7860→200。
3. UI 走查:导入 Mem0 → 四维条数明显收敛(约 12–20)且无琐碎 bug 条;点某条 🔒 → 图标变锁定;某维 + 添加一条 → 出现且标手动;点「全量重跑」→ 锁定/手动条目仍在、其余被重提替换;🗑 删除某条 → 消失且写盘。

## 不做
- 不改记忆聊天 / mem0 链路 / 结构分析 Tab。
- 不引入条目随机 id / 迁移脚本。
- 不做拖拽排序、批量选择等超出范围的交互。
