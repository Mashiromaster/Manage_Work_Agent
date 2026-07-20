"""用户个性定制存储:全局人设文本 + 记忆注入开关。

与逐条记忆(存 Qdrant)、结构化画像(profiles/*.json)互补——这里存用户手写的
**全局人设/系统提示词**与「是否注入」开关,按 user_id 隔离,落 persona/<uid>.json。

纯 JSON 读写,不碰 Qdrant/LLM,可直接单测;MCP server 也读这里(文件后端,进程安全)。
"""

import json
import os

PERSONA_DIR = "./persona"


def _safe_name(user_id: str) -> str:
    return (user_id or "").replace("/", "_").replace("\\", "_").strip() or "default_user"


def _path(user_id: str) -> str:
    return os.path.join(PERSONA_DIR, f"{_safe_name(user_id)}.json")


def load_persona(user_id: str) -> dict:
    """读某用户的人设配置;缺省 ``{"persona": "", "inject": True}``。

    inject 缺省 True,保持与旧行为一致(默认注入记忆)。
    """
    p = _path(user_id)
    if not os.path.isfile(p):
        return {"persona": "", "inject": True}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"persona": "", "inject": True}
    return {
        "persona": str(data.get("persona", "")),
        "inject": bool(data.get("inject", True)),
    }


def save_persona(user_id: str, persona: str, inject: bool = True) -> str:
    """写某用户的人设配置,返回文件路径。"""
    os.makedirs(PERSONA_DIR, exist_ok=True)
    p = _path(user_id)
    payload = {"persona": persona or "", "inject": bool(inject)}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def list_persona_users() -> list:
    """列出已保存人设的 user_id(按名排序)。"""
    if not os.path.isdir(PERSONA_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PERSONA_DIR)
                  if f.endswith(".json") and not f.startswith("_"))


# ---- 四维提取系统提示词(进度追踪用,可在个性定制页编辑) ----

_DIM_PROMPT_FILE = "_dim_prompt.json"

DEFAULT_DIM_PROMPT = """你是资深项目进度分析师。下面给出开发者与 AI 助手关于某项目的**完整对话原文**。
请通读全文,归纳出项目**当前**的真实状态,按四个维度结构化输出。

## 核心原则:收敛、系统、反映现状
你的目标不是流水账,而是让人 30 秒看懂「这个项目做成了什么、现在卡在哪、接下来做什么、为什么这么做」。

- **按功能/模块聚合,不按对话轮次罗列**:把围绕同一功能、同一模块、同一目标的多次改动**合并成一条**,描述其**最终形态**,而不是把每次小修小补各列一条。宁可一条写全,不要拆成五条零散项。
- **只反映最新状态**:同一件事反复出现,只留最新结论;中间过程、试错、被推翻的做法都不要单列。
- **已解决 = 进度,不是障碍**:凡是在对话里最终**被修复/被解决**的 bug、报错、卡点,一律**不进 blocker**;若值得记,归并进对应的 progress(作为"已修复 X")。blocker **只放到现在仍未解决**的问题。
- **进度要"有效"**:一条 progress 应对应一个用户可感知的功能、一个完成的模块、或一个有意义的里程碑;琐碎的中间步骤(改了个变量名、加了行日志、临时验证)不单列。

## 四个维度
1. **进度 (progress)**:已完成并**当前仍成立**的功能 / 模块 / 里程碑 / 已修复项。按功能聚合。
2. **问题 / 障碍 (blocker)**:**截至对话结束仍未解决**的 bug、报错、卡点、限制。已解决的不要放这里。
3. **待办 / 待优化 (todo)**:明确计划要做、待优化、"以后再说"、已知但尚未动手的事项。
4. **决策记录 (decision)**:关键技术选择及其**理由**(为什么选 A 不选 B)。一次性琐碎选择不记。

## 数量与粒度(重要)
- 追求**信息密度**而非条数。四个维度合计控制在**约 10–25 条**;若对话很短可更少。
- 每个维度内先按重要度排序,同类合并后**只保留真正有价值的条目**;宁缺毋滥。
- 每条 text 一句话说清一件事(可含关键文件/模块名),避免又臭又长或过度拆分。

## 重要度 (1-10)
严重阻塞 / 项目级核心决策 8-10;关键功能进度 / 重要决策 6-8;一般待办 / 次要进度 4-6;边角 1-3。

## 输出格式
严格输出 JSON 数组(不要 markdown 代码块、不要多余文字):

[{"dimension": "progress", "text": "归纳后的结论", "importance": 8, "evidence": "对话里的简短依据"}]

每条含:dimension(progress/blocker/todo/decision)、text(结论)、importance(1-10)、evidence(对话里的简短依据)。"""


# 跨段汇总提示词:长对话切多段、各段分别提取后,把所有段的条目合并再收敛一遍。
# 内部固定(不暴露给用户编辑),目标是消除段间重复/零散、反映全局最新状态。
DIM_MERGE_PROMPT = """你是资深项目进度分析师。下面是对**同一个项目**的长对话分多段提取出的四维条目(progress/blocker/todo/decision),因为分段提取,条目之间存在**重复、零散、甚至互相矛盾**(后段可能推翻或解决了前段的内容)。

请把它们合并成一份**全局、收敛、反映最新状态**的四维清单。

## 合并规则
- **同一件事跨段重复**:合并成一条,保留信息最全的表述,importance 取最高。
- **按功能/模块聚合**:围绕同一功能、同一模块的多条零散 progress,合并成一条描述其最终形态。
- **已解决的问题移出 blocker**:若某条 blocker 在其他条目里显示已被修复/解决,则从 blocker 删除(可作为"已修复 X"并入 progress)。blocker 只留**全局仍未解决**的。
- **矛盾取最新**:前后不一致时,以更晚、更"完成态"的表述为准。
- **控制总量**:四维合计约 **10–25 条**,追求信息密度;删掉琐碎、边角、中间过程条目。每个维度内按重要度排序。

## 输出格式
严格输出 JSON 数组(不要 markdown 代码块、不要多余文字),字段与输入相同:

[{"dimension": "progress", "text": "...", "importance": 8, "evidence": "..."}]"""


def _dim_prompt_path() -> str:
    return os.path.join(PERSONA_DIR, _DIM_PROMPT_FILE)


def load_dim_prompt() -> str:
    """读四维提取系统提示词;未配置则返回内置默认。"""
    p = _dim_prompt_path()
    if not os.path.isfile(p):
        return DEFAULT_DIM_PROMPT
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        prompt = str(data.get("dim_prompt", "")).strip()
        return prompt or DEFAULT_DIM_PROMPT
    except (OSError, json.JSONDecodeError):
        return DEFAULT_DIM_PROMPT


def save_dim_prompt(text: str) -> str:
    """写四维提取系统提示词;空文本视为恢复默认(删除自定义文件)。返回路径或默认标记。"""
    os.makedirs(PERSONA_DIR, exist_ok=True)
    p = _dim_prompt_path()
    if not (text or "").strip():
        if os.path.isfile(p):
            os.remove(p)
        return "(默认)"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"dim_prompt": text}, f, ensure_ascii=False, indent=2)
    return p
