"""LLM 驱动的项目进度提炼模块。

仿 :mod:`memory_framework.profile_extractor`,但面向项目追踪:从项目对话记忆里
提炼进度 / 问题 / 待办 / 决策四类信息。复用同一套 JSON 解析与校验逻辑
(:func:`profile_extractor._parse_json_response`,传入 PROJECT_DIMENSIONS)。

用法::

    from memory_framework.project_extractor import synthesize_project_dims

    items = synthesize_project_dims(
        memories=["把聊天延迟从 14s 降到 4s", "还没换更快的模型"],
        existing_items=[],
        model="anthropic/Claude-Opus-4.8-hq",
    )
    # → [{dimension: "progress", text: "聊天延迟优化到 4s", ...}, ...]
"""

import os
import time

import litellm

from memory_framework.profile_extractor import _parse_json_response
from memory_framework.project_dims import PROJECT_DIMENSIONS

# LLM 提炼失败(限流/网络/解析空)时的重试:指数退避,与 code_analyzer 一致。
_LLM_RETRIES = 3
_LLM_RETRY_BASE = 2.0

SYNTHESIS_PROMPT = """你是一个项目进度分析专家。根据以下项目对话记忆,提炼结构化的项目进展。

## 核心原则

对话记忆来自开发者与 AI 助手关于某个项目的历次对话。请从中归纳出**项目当前的真实状态**,
而不是逐句复述对话。同一件事的多次提及应合并为一条,并反映其最新状态。

## 四个维度

1. **进度 (progress)**: 已经做到哪了、已完成的功能 / 里程碑。例如:
   - "已把聊天延迟从 14s 降到约 4s" ✅
   - "结构化用户画像功能已上线" ✅

2. **问题 / 障碍 (blocker)**: 尚未解决的 bug、报错、卡点、技术困难。例如:
   - "Qdrant 本地模式只允许单进程访问,多线程需加锁" ✅
   - "litellm 启动联网拉 cost map 导致卡顿(已用离线标志规避)" —— 若已解决则不必列为 blocker

3. **待办 / 待优化 (todo)**: 计划要做、待优化、"以后再说"的事项。例如:
   - "考虑换更快的模型进一步降延迟" ✅
   - "项目维度提炼需补单元测试" ✅

4. **决策记录 (decision)**: 关键技术选择及其理由。例如:
   - "选用 conda 虚拟环境而非系统 python(避免污染 homebrew,规避 PEP 668)" ✅
   - "回复先返回、记忆/画像后台异步落盘(降低感知延迟)" ✅

## 重要度评分 (1-10)

- 阻塞进展的严重 blocker: 8-10
- 关键决策 / 核心进度: 7-9
- 一般待办 / 次要进度: 4-7
- 边角事项: 1-4

## 输出格式

严格输出 JSON 数组(不要 markdown 代码块,不要其他文字):

[{"dimension": "progress", "text": "聊天延迟优化到约 4s", "importance": 8, "evidence": "把聊天延迟从 14s 降到 4s"}]

每条包含:
- dimension: progress / blocker / todo / decision
- text: 归纳后的状态描述(结论,不是原文复述)
- importance: 1-10
- evidence: 支撑该条目的原始记忆(简短引用即可)
"""


def synthesize_project_dims(
    memories: list[str],
    existing_items: list[dict] | None = None,
    model: str = "anthropic/Claude-Opus-4.8-hq",
) -> list[dict]:
    """用 LLM 从项目记忆中提炼进度 / 问题 / 待办 / 决策四类条目。

    Args:
        memories: 该项目全部记忆文本列表。
        existing_items: 现有的项目条目(用于增量合并),可选。
        model: litellm 格式的模型名。

    Returns:
        条目列表,每项含 dimension/text/importance/evidence;解析失败返回空列表。
    """
    if not memories:
        return []

    memory_lines = "\n".join(f"- {m}" for m in memories)
    user_msg = f"## 项目对话记忆\n\n{memory_lines}"

    if existing_items:
        existing_lines = "\n".join(
            f"- [{it.get('dimension', '?')}] {it.get('text', '')}"
            for it in existing_items
        )
        user_msg += f"\n\n## 现有项目记录(请在此基础上合并更新)\n\n{existing_lines}"

    user_msg += "\n\n请提炼项目进展,直接输出 JSON 数组。"

    messages = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # 单次调用撞 429 / 网络抖动 / 返回空,历史上会让整份提炼静默返回 []、
    # 导致 refresh_project 不写盘。这里做指数退避重试,把瞬时失败挡在外面。
    last_exc = None
    for attempt in range(_LLM_RETRIES + 1):
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=2048,
            )
            raw = resp["choices"][0]["message"]["content"]
            items = _parse_json_response(raw, allowed_dims=PROJECT_DIMENSIONS)
            if items:
                return items
            # 解析为空(限流截断/格式异常):除最后一次外都重试。
            if attempt < _LLM_RETRIES:
                time.sleep(_LLM_RETRY_BASE * (2 ** attempt))
                continue
            return []
        except Exception as exc:  # noqa: BLE001 —— 限流/网络统一退避重试
            last_exc = exc
            if attempt < _LLM_RETRIES:
                time.sleep(_LLM_RETRY_BASE * (2 ** attempt))
    if last_exc is not None:
        print(f"[project_extractor] 提炼重试 {_LLM_RETRIES} 次后仍失败:{last_exc}")
    return []
