"""项目级长期记忆问答:把「进度四维 + 代码分析摘录 + 项目 Q&A 记忆」注入上下文。

镜像 :mod:`chat` 的 reply/persist_turn 范式,但面向**单个项目**而非单个用户:

- 记忆命名空间用 ``user_id=f"proj::{project_id}"``,与用户画像的记忆彻底隔离;
- system prompt 额外注入该项目的进度四维(ProjectStore)与代码深度分析摘录
  (code_analysis/<id>.json),让 Claude 回答时对项目上下文了如指掌;
- 分析历史沉淀:每次深度分析后调 :func:`sediment_analysis`,把一行摘要作为
  memory 存入 ``proj::<id>``,供后续 Q&A 语义召回。

纯编排,LLM 与 store 均可注入/mock,便于单测。
"""

import os

import litellm

from memory_framework.code_analyzer import load_code_analysis
from memory_framework.config import (
    DEFAULT_LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from memory_framework.project_dims import PROJECT_DIMENSION_LABELS

SYSTEM_BASE = (
    "你是一个熟悉本项目的中文开发助手。你能看到该项目的开发进度、代码结构分析与"
    "历史问答记忆。请结合这些上下文,准确、简洁地回答开发者关于本项目的问题;"
    "信息不足时如实说明,不要臆造。"
)


def project_namespace(project_id: str) -> str:
    """项目记忆命名空间键,与用户画像的 user_id 隔离。"""
    return f"proj::{project_id}"


def _progress_block(project_store, project_id: str) -> str:
    """把项目四维进度渲染成 system prompt 片段;无则空串。"""
    if project_store is None:
        return ""
    try:
        profile = project_store.get_project(project_id)
    except Exception:
        return ""
    items = getattr(profile, "items", None) or []
    if not items:
        return ""
    by_dim = {}
    for it in items:
        by_dim.setdefault(it.dimension, []).append(it.text)
    lines = []
    for dim, label in PROJECT_DIMENSION_LABELS.items():
        texts = by_dim.get(dim)
        if not texts:
            continue
        lines.append(f"【{label}】")
        lines.extend(f"  - {t}" for t in texts[:8])
    return "## 开发进度\n" + "\n".join(lines) if lines else ""


def _code_block(project_id: str, limit: int = 20) -> str:
    """把代码深度分析的逐文件摘要渲染成 system prompt 片段;无则空串。"""
    data = load_code_analysis(project_id)
    if not data:
        return ""
    summaries = data.get("summaries") or []
    if not summaries:
        return ""
    lines = []
    for s in summaries[:limit]:
        f = s.get("file", "?")
        role = s.get("role", "")
        summ = s.get("summary", "")
        lines.append(f"  - {f} — {role} :: {summ}")
    extra = ""
    if len(summaries) > limit:
        extra = f"\n  (另有 {len(summaries) - limit} 个文件未列出)"
    return "## 代码结构(逐文件摘要摘录)\n" + "\n".join(lines) + extra


def _memory_block(memories: list) -> str:
    if not memories:
        return ""
    lines = [str(m.get("memory", m)) if isinstance(m, dict) else str(m)
             for m in memories]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return ""
    return "## 相关历史记忆\n" + "\n".join(f"  - {ln}" for ln in lines)


def _build_system_prompt(project_id: str, project_store, memories: list) -> str:
    parts = [SYSTEM_BASE, f"当前项目:{project_id}"]
    for block in (
        _progress_block(project_store, project_id),
        _code_block(project_id),
        _memory_block(memories),
    ):
        if block:
            parts.append(block)
    return "\n\n".join(parts)


def project_reply(store, project_store, project_id: str, user_input: str,
                  model: str = None) -> str:
    """检索项目记忆 → 注入进度/代码/记忆上下文 → 调 Claude 回复(不落盘)。

    仅做用户等待的关键路径:检索 + 生成。落盘(记忆抽取)较慢,拆到
    :func:`persist_project_turn`,由调用方在回复返回后执行。
    """
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    ns = project_namespace(project_id)
    memories = store.search(query=user_input, user_id=ns, limit=5)
    messages = [
        {"role": "system",
         "content": _build_system_prompt(project_id, project_store, memories)},
        {"role": "user", "content": user_input},
    ]
    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp["choices"][0]["message"]["content"]


def persist_project_turn(store, project_id: str, user_msg: str,
                         answer: str) -> None:
    """把本轮项目问答交给 mem0 抽取并存入 ``proj::<id>`` 命名空间。"""
    store.add(
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ],
        user_id=project_namespace(project_id),
    )


def sediment_analysis(store, project_id: str, capinfo: dict = None,
                      summaries: list = None) -> None:
    """把一次深度分析的要点沉淀成一条项目记忆,供后续 Q&A 语义召回。

    存的是「本次分析覆盖了哪些关键模块、分析了多少文件」这类元信息,
    不重复整份报告——报告本身已在 code_analysis/<id>.md。
    """
    capinfo = capinfo or {}
    summaries = summaries or []
    key_files = [s.get("file") for s in summaries[:8] if s.get("file")]
    analyzed = capinfo.get("analyzed", len(summaries))
    kind = "增量" if capinfo.get("incremental") else "全量"
    note = (f"完成一次{kind}代码深度分析,共分析 {analyzed} 个源文件。"
            f"关键文件:{', '.join(key_files)}。" if key_files
            else f"完成一次{kind}代码深度分析,共分析 {analyzed} 个源文件。")
    store.add(
        messages=[{"role": "user", "content": note}],
        user_id=project_namespace(project_id),
    )


def sediment_change(store, project_id: str, diff: dict) -> None:
    """把代码变更 diff 沉淀成一条项目记忆(自上次分析哪些文件变了)。"""
    diff = diff or {}
    added = diff.get("added") or []
    changed = diff.get("changed") or []
    removed = diff.get("removed") or []
    if not (added or changed or removed):
        return
    parts = []
    if added:
        parts.append(f"新增 {', '.join(added[:8])}")
    if changed:
        parts.append(f"修改 {', '.join(changed[:8])}")
    if removed:
        parts.append(f"删除 {', '.join(removed[:8])}")
    note = "自上次分析以来代码变更:" + ";".join(parts) + "。"
    store.add(
        messages=[{"role": "user", "content": note}],
        user_id=project_namespace(project_id),
    )
