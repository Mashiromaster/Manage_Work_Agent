"""从**对话原文**直接提炼项目四维(进度/问题/待办/决策)——不经 mem0。

与 :mod:`project_extractor`(从 mem0 记忆碎片提炼)的关键区别:那条链路先让 mem0
把对话有损压成零散记忆再提炼,信息在第一步就丢了、结果不全;这里让 LLM 直接读
**完整对话原文**一次性提炼,无损、可控、更准。

超长对话按字符上限切多段,各段分别提炼后合并去重(同维度同文本取重要度高者)。
系统提示词由调用方传入(用户可在个性定制页编辑),四维定义与输出格式写在提示词里。

复用 :func:`code_analyzer._complete`(带退避重试的 LLM 调用)与
:func:`profile_extractor._parse_json_response`(三级 JSON 解析 + 维度校验)。
纯逻辑,LLM 可 mock。
"""

import os

from memory_framework.code_analyzer import _complete
from memory_framework.config import DEFAULT_LLM_MODEL
from memory_framework.persona_store import DIM_MERGE_PROMPT
from memory_framework.profile_extractor import _parse_json_response
from memory_framework.project_dims import PROJECT_DIMENSIONS

CHUNK_CHARS = 24000  # 每段喂 LLM 的对话原文字符上限(留足上下文余量)。


def _messages_to_text(messages: list) -> str:
    """把消息列表拼成「role: content」纯文本对话。"""
    lines = []
    for m in messages:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        if content and content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n\n".join(lines)


def _chunk_messages(messages: list, cap: int = CHUNK_CHARS) -> list:
    """按累计字符上限把消息切成多段(每段是消息子列表),按整条消息边界切。"""
    chunks = []
    cur, cur_len = [], 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        mlen = len(content) + 16  # 加上 role 前缀等开销
        if cur and cur_len + mlen > cap:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(m)
        cur_len += mlen
    if cur:
        chunks.append(cur)
    return chunks or [[]]


def _norm(text: str) -> str:
    return "".join((text or "").split()).strip("。,.!?;:")


def _merge_dedup(items_lists: list) -> list:
    """合并多段结果:按 (dimension, 归一化 text) 去重,重要度取高者。"""
    merged = {}
    for items in items_lists:
        for it in items:
            if not isinstance(it, dict):
                continue
            key = (it.get("dimension"), _norm(it.get("text", "")))
            if not key[0] or not key[1]:
                continue
            prev = merged.get(key)
            if prev is None or float(it.get("importance", 0)) > float(
                    prev.get("importance", 0)):
                merged[key] = it
    return list(merged.values())


def extract_dims_from_messages(messages: list, dim_prompt: str,
                               existing_items: list = None,
                               model: str = None,
                               progress_cb=None) -> list:
    """从对话原文提炼四维条目。

    Args:
        messages: ``[{"role","content"}, ...]`` 对话原文(cc_ingest.parse_cc_session 输出)。
        dim_prompt: 四维提取系统提示词(用户可编辑;persona_store.load_dim_prompt())。
        existing_items: 现有四维条目(增量合并用,传给 LLM 做参考),可选。
        model: litellm 模型名。
        progress_cb: 可选回调 ``fn(current, total)``,每段 LLM 调用**前**调用一次
            (current 从 1 起、total 为总段数),供上层做分段进度显示。

    Returns:
        ``[{"dimension","text","importance","evidence"}, ...]``;无消息或全失败返回 []。
    """
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    messages = [m for m in messages
                if isinstance(m, dict) and (m.get("content") or "").strip()]
    if not messages:
        return []

    existing_block = ""
    if existing_items:
        lines = "\n".join(
            f"- [{it.get('dimension', '?')}] {it.get('text', '')}"
            for it in existing_items if isinstance(it, dict)
        )
        if lines:
            existing_block = ("\n\n## 现有四维记录(在此基础上合并更新,不要丢弃仍成立的旧条目)\n"
                              + lines)

    chunks = _chunk_messages(messages, cap=CHUNK_CHARS)
    total = len(chunks)
    results = []
    for idx, chunk in enumerate(chunks):
        if progress_cb is not None:
            try:
                progress_cb(idx + 1, total)
            except Exception:
                pass  # 进度回调失败不影响提炼本身
        text = _messages_to_text(chunk)
        part = f"(第 {idx + 1}/{total} 段)" if total > 1 else ""
        user_msg = (f"## 项目对话原文 {part}\n\n{text}{existing_block}\n\n"
                    "请通读上面对话,提炼四维,直接输出 JSON 数组。")
        raw = _complete(
            [{"role": "system", "content": dim_prompt},
             {"role": "user", "content": user_msg}],
            model, max_tokens=3000,
        )
        if raw:
            results.append(_parse_json_response(raw, allowed_dims=PROJECT_DIMENSIONS))

    merged = _merge_dedup(results)
    # 跨段汇总:只在真正分了多段时才多花一次 LLM,把段间重复/零散/已解决项收敛。
    # 单段无需汇总(段内提示词已收敛);汇总失败则退回精确去重结果,不丢数据。
    if total > 1 and merged:
        if progress_cb is not None:
            try:
                progress_cb(total, total, phase="merge")
            except TypeError:
                progress_cb(total, total)  # 老式回调不接受 phase
            except Exception:
                pass
        consolidated = _consolidate_items(merged, model)
        if consolidated:
            return consolidated
    return merged


def _consolidate_items(items: list, model: str) -> list:
    """把多段提取、精确去重后仍零散的四维条目,喂 LLM 做一次全局收敛。

    失败(限流/解析空)返回 [],由调用方退回未收敛结果,保证不丢数据。
    """
    if not items:
        return []
    lines = "\n".join(
        f'- [{it.get("dimension", "?")}] (importance={it.get("importance", 0)}) '
        f'{it.get("text", "")}'
        + (f' | 来源: {it.get("evidence")}' if it.get("evidence") else "")
        for it in items if isinstance(it, dict)
    )
    user_msg = (f"## 分段提取出的四维条目(共 {len(items)} 条,存在重复/零散)\n\n"
                f"{lines}\n\n请按合并规则收敛成全局最新的四维清单,直接输出 JSON 数组。")
    raw = _complete(
        [{"role": "system", "content": DIM_MERGE_PROMPT},
         {"role": "user", "content": user_msg}],
        model, max_tokens=3000,
    )
    if not raw:
        return []
    return _parse_json_response(raw, allowed_dims=PROJECT_DIMENSIONS)
