# Mem0 长期记忆框架 — 开发文档

> 本文档记录项目当前结构、已实现功能、可优化点与下一步方案,供后续开发参考。
> 最后更新:2026-07-08

## 一、项目概述

基于 [mem0ai](https://github.com/mem0ai/mem0) 2.0.11 搭建的**中文长期记忆框架**:让 AI 助手跨会话记住用户,并把零散记忆提炼成结构化的四维用户画像。

**技术栈**
- LLM:Claude(`Claude-Opus-4.8-hq`),经 **litellm** 转 **京东云中转站**(`modelservice.jdcloud.com`)
- Embedding:本地 **bge-small-zh-v1.5**(512 维,中文友好)
- 向量库:**Qdrant** 本地文件模式(`./qdrant_data`)
- 界面:**Gradio**
- 环境:conda 环境 **`mem0`**(Python 3.11.15)

**运行入口**
```bash
conda activate mem0
export HF_ENDPOINT=https://hf-mirror.com   # 首次下载 embedding 模型加速
# CLI 聊天
python -m memory_framework.chat --user alice
# Gradio 界面
PYTHONPATH=. python app.py    # → http://127.0.0.1:7860
# 检索召回评测
PYTHONPATH=. python tests/evaluate.py            # 默认抽样前 2 个测试集
PYTHONPATH=. python tests/evaluate.py --all      # 全部 10 个(慢)
```

## 二、当前项目结构

```
Mem0/
├── memory_framework/          # 核心包
│   ├── config.py              # 配置:litellm→Claude / bge / qdrant。已实测调优(见约定)
│   ├── memory_store.py        # MemoryStore:mem0.Memory 薄封装(add/search/get_all/update/delete/delete_all)
│   ├── conversation.py        # 对话 JSON 加载/校验/灌入
│   ├── profile.py             # 画像数据结构:ProfileItem / UserProfile,4 维度
│   ├── scoring.py             # 打分+时间衰减遗忘(纯函数)。classify() 已 deprecated
│   ├── profile_extractor.py   # LLM 驱动的画像提炼(synthesize_profile)
│   ├── profile_store.py       # 画像持久化(profiles/<uid>.json):replace_profile / ingest / get_profile
│   └── chat.py                # CLI 聊天 + 每轮更新画像
├── app.py                     # Gradio 界面(聊天 + 记忆面板 + 画像面板 + 加载测试集)
├── data/
│   ├── conversations/         # 少量演示对话
│   ├── testsets/              # 10 个人设 × ~70 轮合成对话 + .facts.json(评测真值)
│   ├── gen_testsets.py        # 生成上面的测试集
│   └── eval_report.json       # 最近一次评测结果
├── tests/                     # 单元测试(48 个)+ evaluate.py(检索评测脚本)
├── profiles/                  # 运行时生成的用户画像(未纳入 git)
├── qdrant_data/               # 向量库数据(gitignore)
├── docs/superpowers/          # 设计 spec + 实现计划
├── requirements.txt / .env.example / README.md
```

## 三、核心数据流

**写入(灌记忆 + 建画像)**
```
对话 messages → MemoryStore.add → mem0(Claude 抽取事实 + bge 向量化 + 存 Qdrant)
             → chat.update_profile_from_memories
             → get_all 取全部记忆 → synthesize_profile(Claude 提炼四维画像)
             → ProfileStore.replace_profile 全量替换落盘
```

**检索(对话时召回)**
```
用户输入 → MemoryStore.search(top_k=5) → 注入 system prompt → Claude 生成回复
```

**画像四维度**:`personality`(性格)/ `interest`(兴趣)/ `event`(事件)/ `taboo`(禁忌)。
每条含 `text`(推断结论)、`importance`(1-10)、`evidence`(原始记忆溯源)、`mention_count`、时间戳、`forgotten`。

## 四、关键设计约定(勿破坏)

1. **litellm 环境桥接**:mem0 2.0.11 的 litellm provider 用 `BaseLlmConfig`,**不接受 `api_base` 键**。中转站基址/密钥通过环境变量 `ANTHROPIC_API_BASE`/`ANTHROPIC_API_KEY` 传给 litellm。**创建 `Memory` 前必须先调 `apply_litellm_env()`**(或用 `build_config_and_apply_env()`)。
2. **config.py 已实测调优**:`temperature=1`(Opus 推理模型只接受 1)、`litellm.drop_params=True`(自动丢弃不支持的采样参数)、`custom_instructions` 强制中文存记忆、`EMBED_DIMS=512`。
3. **mem0 2.0.11 真实 API 差异**(已在 memory_store 封装):`search`/`get_all` 用 `filters={"user_id":...}` 而非顶层 `user_id=`;`search` 数量参数是 `top_k` 非 `limit`;返回 `{"results":[...]}` 需解包。
4. **画像更新是全量替换**(`replace_profile`),不是增量 append;LLM 提炼失败时**回退旧画像**不丢数据。

## 五、当前状态

- ✅ 48 个单元测试全绿(config/conversation/memory_store/profile/scoring/profile_extractor/chat 集成)
- ✅ 画像提炼真实验证:Claude 正确从事件推断性格(跑全马→"坚毅自律"、学三门语言→"求知欲强")
- ✅ 检索评测:抽样 2 个测试集(chef_lin / dev_zhao)召回率 **6/6 = 100%**
- ⚠️ 全量 10 个测试集的评测尚未跑过(省 API)

## 六、可优化的地方

### P0 — 影响正确性/体验
1. **画像条目重叠冗余**:同一次提炼会产出语义重叠的多条,例如 personality 同时出现"坚毅自律,有目标感""生活规律、自律"。`replace_profile` 全量替换也未做语义去重。
   - 现象来源:`synthesize_profile` 每次独立提炼,无跨条目合并。
2. **每轮对话都全量重新提炼画像**:`chat.reply` 每轮都 `get_all` 全部记忆 + 调 Claude 重新提炼整个画像。轮次多时**又慢又贵**(70 轮 = 70 次全量画像提炼)。
   - 应改为:增量提炼(只处理新增记忆)或降低提炼频率(每 N 轮/按需触发)。

### P1 — 质量/工程
3. **mention_count 语义在 LLM 路径下失真**:`replace_profile` 直接用 LLM 返回的 `mention_count`(默认 1),而 LLM 并不真正统计提及次数。时间衰减遗忘依赖它,当前基本恒为 1,削弱了"高频记忆更难遗忘"的设计。
4. **画像与遗忘机制脱节**:`scoring.survival_score` 的时间衰减遗忘只在 `get_profile` 读取时计算,但 `replace_profile` 每次全量覆盖 `created_at`/`last_seen`,导致衰减历史被重置——遗忘几乎不会触发。
5. **evaluate.py 放在 tests/ 但不是 pytest 测试**:它是脚本(需真实 API),混在单元测试目录里,`pytest tests/` 若不小心会触发真实调用。建议移到 `scripts/` 或 `data/`。
6. **无检索质量的量化基线沉淀**:评测只跑过 2 个集,缺全量基线,后续优化无法对比"改好了还是改坏了"。

### P2 — 可选增强
7. 画像提炼的 `existing_items` 已传给 LLM 做"增量合并",但 prompt 未强约束"合并同类、不要新增重叠条目",合并效果依赖模型自觉。
8. 无并发/多用户压测;Qdrant 本地文件模式在并发写下的行为未验证。
9. `config.py` 里模型名等硬编码默认值,可考虑集中到 `.env`。

## 七、下一步优化方案(建议顺序)

**第一步:治理画像质量(P0-1 + P1-3/4)**
- 在 `profile_store` 增加语义去重/合并:提炼后按 (dimension + 文本相似度) 归并重叠条目,合并时累加 `mention_count`、保留最早 `created_at`。
- `replace_profile` 改为 `merge_profile`:以现有画像为基,新提炼结果与旧条目对齐合并,保留时间戳与提及计数,让遗忘机制真正生效。
- 验证:构造"同一特征多轮重复提及"的对话,断言最终画像该特征只有一条且 `mention_count>1`。

**第二步:降低画像提炼成本(P0-2)**
- 把"每轮全量提炼"改为"按需/增量":
  - 方案 A(简单):每 N 轮或用户输入 `/profile` 时才提炼。
  - 方案 B(更优):只把**本轮新增记忆**喂给 `synthesize_profile`,与现有画像合并(依赖第一步的 merge)。
- 验证:对比改造前后 70 轮对话的 API 调用次数。

**第三步:建立评测基线(P1-6)**
- 跑一次 `evaluate.py --all`,把 10 个集的召回率存为基线到 `data/eval_baseline.json`。
- 之后每次优化后重跑,对比召回率与画像质量,防止回归。

**第四步:工程整理(P1-5)**
- `evaluate.py` / `gen_testsets.py` 归到 `scripts/`;确保 `pytest tests/` 只跑纯单元测试不触发真实 API。

## 八、给后续开发者的提示

- 跑任何真实功能前:`conda activate mem0`,并确保 `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_BASE_URL` 在环境里(会话已继承,无需 .env)。
- 改配置先读 `config.py` 顶部的长 docstring——里面记录了所有踩过的坑。
- 改画像逻辑注意三个文件的协作:`profile_extractor`(提炼)→ `profile_store`(存/合并/遗忘)→ `scoring`(打分衰减)。
- 清空某用户重来:删 `profiles/<uid>.json` + `MemoryStore().delete_all(user_id=...)`。
