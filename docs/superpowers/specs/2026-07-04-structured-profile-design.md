# 结构化用户画像 + 重要性评估 + 时间衰减遗忘 设计

日期:2026-07-04
状态:已确认,待实现

## 目标

在现有 `MemoryStore`(mem0 向量记忆层)之上叠加一个**结构化用户画像层**,能:
1. 从对话历史持续构建四维度用户画像(兴趣爱好 / 性格特征 / 事件经历 / 禁忌与反感);
2. 对每条记忆做**重要性评估**(规则打分 + 频次加权,零额外 LLM 成本);
3. 引入**时间权重**,按"重要性 × 时间衰减"计算存活分,低于阈值的记忆被标记遗忘(降权,不删除)。

## 已确认的关键决策

| 决策点 | 选择 |
|--------|------|
| 画像存储形态 | JSON 画像层,叠加在向量记忆之上(不改现有代码) |
| 画像维度 | 兴趣爱好、性格特征、事件经历、禁忌与反感(4 维) |
| 重要性评分 | 规则基础分 + 频次加权(不额外调 LLM) |
| 遗忘策略 | 存活分 = 重要性 × 时间衰减(指数),低于阈值标记遗忘、不删除 |
| 更新时机 | 每轮对话存入新记忆后,实时增量更新画像 |

## 架构

画像是向量记忆的**衍生视图**;记忆仍是唯一真相源,画像可随时从记忆重建。

```
一轮对话 → MemoryStore.add → mem0 抽取记忆(向量库)
                                │ 返回新增记忆
                                ▼
              ProfileEngine.ingest(新记忆)
                ① 分类:每条归到 4 维之一
                ② 打分:规则基础分 + 频次加权
                ③ 时间戳:created_at / last_seen
                                ▼
              ProfileStore(profiles/<user_id>.json)
                                │
       读取时:survival = importance × exp(-λ·Δt_days)
       survival < 阈值 → forgotten=True(降权,保留)
```

## 模块划分(全部在 memory_framework/ 下,职责单一)

- **profile.py** — 数据结构。`ProfileItem`(text, dimension, importance, created_at, last_seen, mention_count, forgotten),`UserProfile`(四维度容器 + 序列化)。
- **scoring.py** — 纯函数。`classify(text) -> dimension`、`base_score(dimension, text) -> float`、`survival_score(importance, last_seen, now) -> float`。可完全单测,不依赖 LLM/网络。
- **profile_store.py** — 持久化与增量更新。`ProfileStore`:`ingest(user_id, memories)`、`get_profile(user_id, now=None)`(读取时计算存活分并排序)、`forget_sweep(user_id)`(标记遗忘)。

## 第 2 节:四维度分类 + 规则打分

**分类**(关键词规则 + 兜底):
- 禁忌与反感:含"不吃/过敏/讨厌/不喜欢/忌/反感/受不了"等 → `taboo`
- 事件经历:含时间词或动作完成态("报名了/去了/遇到/昨天/上周/计划/打算"+日期) → `event`
- 性格特征:含"性格/习惯/喜欢一个人/内向/外向/自律/急性子"等描述特质 → `personality`
- 兴趣爱好:兜底默认 → `interest`(大部分"喜欢/在学/爱好"归此)

**基础分**(0-10,反映维度固有重要性):
- 禁忌与反感 = 9(高优先,几乎不忘)
- 事件经历 = 5(中,会随时间过时)
- 性格特征 = 7(较稳定)
- 兴趣爱好 = 6

**频次加权**:同一维度内语义近似的记忆重复出现,`mention_count += 1`,`importance = min(10, base + log2(mention_count))`。重复被提及 → 更重要。去重判定:文本归一化后包含关系或高重叠(简单规则,不调 embedding)。

## 第 3 节:时间衰减与遗忘

**存活分公式**:
```
survival = importance × exp(-λ × Δt_days)
Δt_days = (now - last_seen) 的天数
λ(半衰期换算):禁忌 λ 极小(半衰期 ~1年),性格中(~180天),兴趣(~90天),事件大(~30天)
```
即:每条记忆按其维度有不同衰减速度,禁忌几乎不衰减,事件衰减最快。`last_seen` 每次被重新提及会刷新(重新"记起")。

**遗忘阈值**:`survival < FORGET_THRESHOLD`(默认 1.0)→ `forgotten=True`。遗忘的条目**不删除**,只在 `get_profile` 默认视图里降到末尾/隐藏,可回溯。

## 第 4 节:集成与测试

**集成**:
- `chat.reply()` 存入记忆后调用 `ProfileStore.ingest`(增量),使画像每轮实时更新。
- Gradio `app.py` 右侧新增"结构化画像"视图:按四维度分组展示,每条显示存活分,遗忘的灰显。
- 不破坏现有 `MemoryStore` / `demo` / 测试。

**测试**(分层,大部分纯逻辑不调 LLM):
- `test_scoring.py` — 分类、基础分、频次加权、衰减公式(纯函数,快,全覆盖)。
- `test_profile_store.py` — ingest 增量、去重加频次、遗忘标记、JSON 往返(用假记忆,不调 LLM)。
- 时间用注入的 `now` 参数(不用 `Date.now`),便于测试不同时间点的衰减/遗忘。
- 一个可选的端到端:灌入测试集 → 构建画像 → 断言禁忌类存活、闲聊类被遗忘。

## 非目标(YAGNI)

- 不做图数据库 / 实体关系。
- 不额外调 LLM 做重要性打分(用规则+频次)。
- 不做画像的向量检索(画像是结构化视图,直接读)。
- 不做多用户并发写(沿用 Qdrant 单进程限制)。
