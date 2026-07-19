"""LLM 驱动的用户画像提炼模块。

用 Claude 从全部记忆中推断结构化画像:从事件提取性格特征,而非单纯复述原文。

用法::

    from memory_framework.profile_extractor import synthesize_profile

    items = synthesize_profile(
        memories=["用户不吃香菜,对海鲜过敏", "用户坚持长跑训练半年并完赛全马"],
        existing_items=[],
        model="anthropic/Claude-Opus-4.8-hq",
    )
    # → [{dimension: "personality", text: "坚毅自律", evidence: "坚持长跑训练半年并完赛全马", ...}, ...]
"""

import json
import re

import litellm

from memory_framework.profile import DIMENSIONS

SYNTHESIS_PROMPT = """你是一个用户画像分析专家。根据以下用户记忆,提炼结构化的用户画像。

## 核心原则

**从事件和行为中推断性格特征,不要单纯复述事件本身。** 画像应该回答"这是个什么样的人",而不是"这个人做过什么"。

## 四个维度

1. **性格特征 (personality)**: 从行为推断的稳定性格特质。例如:
   - "坚持长跑训练半年并完赛全马" → "坚毅、自律、有目标感" ✅ (不是"跑过马拉松" ❌)
   - "同时学三门编程语言" → "求知欲强、有野心" ✅ (不是"在学编程" ❌)
   - "周末喜欢一个人待着" → "内向、享受独处" ✅

2. **兴趣爱好 (interest)**: 持续投入的领域,提炼兴趣标签。例如:
   - "每天练琴两小时,听了十年爵士乐" → "热爱音乐,尤其爵士乐,有长期练习习惯"
   - "买了单反后每周出门拍照" → "摄影爱好者"

3. **事件经历 (event)**: 重要的人生事件,需同时提炼该事件反映的特质。例如:
   - "刚从字节跳动离职创业" → 事件:"离开大厂创业" + 反映:"敢于冒险、追求自主"
   - "去年在 ICU 住了两周后康复" → 事件:"经历重病康复" + 反映:"有韧性"

4. **禁忌与反感 (taboo)**: 绝对不能触碰的底线(过敏、禁忌、强烈厌恶)。这类可较直接:
   - "不吃香菜,对海鲜过敏" → "不吃香菜,海鲜过敏"
   - "特别讨厌开会,觉得浪费时间" → "厌恶低效会议"

## 重要度评分 (1-10)

- 禁忌: 9-10 (关乎健康安全)
- 性格核心特质: 7-9
- 持续兴趣: 5-7
- 一般事件: 3-6

## 输出格式

严格输出 JSON 数组(不要 markdown 代码块,不要其他文字):

[{"dimension": "personality", "text": "坚毅自律", "importance": 8, "evidence": "坚持长跑训练半年并完赛全马"}]

每条包含:
- dimension: personality / interest / event / taboo
- text: 提炼后的特征描述(推断后的结论,不是原文复述)
- importance: 1-10
- evidence: 支撑该推断的原始记忆(简短引用即可)
"""


def synthesize_profile(
    memories: list[str],
    existing_items: list[dict] | None = None,
    model: str = "anthropic/Claude-Opus-4.8-hq",
) -> list[dict]:
    """用 LLM 从全部记忆中提炼结构化画像条目。

    Args:
        memories: 用户全部记忆文本列表。
        existing_items: 现有的画像条目(用于增量合并),可选。
        model: litellm 格式的模型名。

    Returns:
        结构化画像条目列表,每项含 dimension/text/importance/evidence。
        解析失败时返回空列表。
    """
    if not memories:
        return []

    # 组装用户消息
    memory_lines = "\n".join(f"- {m}" for m in memories)
    user_msg = f"## 用户记忆\n\n{memory_lines}"

    if existing_items:
        existing_lines = "\n".join(
            f"- [{it.get('dimension', '?')}] {it.get('text', '')}"
            for it in existing_items
        )
        user_msg += (
            f"\n\n## 现有画像(请在此基础上合并更新)\n\n{existing_lines}"
        )

    user_msg += "\n\n请提炼用户画像,直接输出 JSON 数组。"

    messages = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = litellm.completion(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=2048,
        )
        raw = resp["choices"][0]["message"]["content"]
        return _parse_json_response(raw)
    except Exception:
        return []


def _parse_json_response(raw: str, allowed_dims: list | None = None) -> list[dict]:
    """从 LLM 原始回复中提取 JSON 数组,解析并校验字段。

    ``allowed_dims`` 为合法维度集合,默认用户画像 DIMENSIONS;项目维度提炼时
    传入 PROJECT_DIMENSIONS 以复用同一套解析逻辑而不放行错误维度。
    """
    # 尝试直接解析
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return _validate_items(items, allowed_dims)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list):
                return _validate_items(items, allowed_dims)
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 JSON 数组
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(0))
            if isinstance(items, list):
                return _validate_items(items, allowed_dims)
        except json.JSONDecodeError:
            pass

    return []


def _validate_items(items: list, allowed_dims: list | None = None) -> list[dict]:
    """校验并清洗 LLM 输出的条目。``allowed_dims`` 默认用户画像 DIMENSIONS。"""
    dims = allowed_dims if allowed_dims is not None else DIMENSIONS
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dim = item.get("dimension", "")
        text = item.get("text", "")
        if dim not in dims:
            continue
        if not text or not text.strip():
            continue
        valid.append({
            "dimension": dim,
            "text": text.strip(),
            "importance": float(item.get("importance", 5)),
            "evidence": str(item.get("evidence", "")),
        })
    return valid
