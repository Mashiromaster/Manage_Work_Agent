"""项目进度追踪编排层。

把三件事串起来:
1. 解析 Claude Code 对话日志(增量),把新消息灌入 MemoryStore(project_id 当
   隔离键,复用 user_id 通道);
2. 读取该项目全部记忆,用 LLM 提炼进度 / 问题 / 待办 / 决策四维度;
3. 全量写入 ProjectStore(不遗忘)。

游标文件放在 ProjectStore.base_dir 下的 ``.ingest_cursor.json``,让重复导入只
处理新增消息。project_id 直接作为 MemoryStore / ProjectStore 的 key。
"""

import os

from memory_framework.cc_ingest import (
    collect_new_messages,
    find_project,
    list_cc_projects,
    load_cursor,
    parse_cc_session,
    save_cursor,
)
from memory_framework.config import DEFAULT_LLM_MODEL
from memory_framework.dialog_extractor import extract_dims_from_messages
from memory_framework.project_extractor import synthesize_project_dims


def _cursor_path(project_store) -> str:
    return os.path.join(project_store.base_dir, ".ingest_cursor.json")


# 单条消息过长(大段代码/日志)会稀释 mem0 的事实抽取,截断到该上限。
_MAX_MSG_CHARS = 1500


def _ingest_in_batches(store, project_id: str, msgs: list, batch_size: int = 6) -> None:
    """把消息分小批灌入 mem0。

    mem0 每次 ``add`` 对传入的一段对话做一次 LLM 事实抽取;一次性灌入几百条会
    让抽取返回空。按 ``batch_size`` 分批,每批作为一段小对话抽取,更稳。
    过长单条消息截断到 ``_MAX_MSG_CHARS``,避免大段代码淹没事实。
    """
    for i in range(0, len(msgs), batch_size):
        chunk = msgs[i:i + batch_size]
        payload = [
            {"role": m["role"], "content": m["content"][:_MAX_MSG_CHARS]}
            for m in chunk
        ]
        try:
            store.add(messages=payload, user_id=project_id)
        except Exception:
            # 单批失败不阻断整体导入(游标已按全量推进,重跑不会重复)。
            continue


def refresh_project(store, project_store, project_id: str, model: str = None):
    """读该项目全部记忆 → LLM 提炼四维度 → 全量替换项目记录。

    类比 :func:`chat.update_profile_from_memories`,但用项目维度提炼且不遗忘。
    LLM 失败时保留旧记录不动。

    Returns:
        ``(profile, status)``:status ∈ {"ok", "no_memories", "llm_empty"}。
        "llm_empty" 表示有记忆但提炼返回空(通常限流/解析失败),旧记录未被覆盖。
    """
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    mems = store.get_all(user_id=project_id)
    texts = [m.get("memory", m) if isinstance(m, dict) else m for m in mems]
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return project_store.get_project(project_id), "no_memories"

    existing = project_store.get_project(project_id)
    existing_items = [it.to_dict() for it in existing.items]

    items_data = synthesize_project_dims(
        memories=texts, existing_items=existing_items, model=model,
    )
    if items_data:
        project_store.replace_profile(project_id, items_data)
        return project_store.get_project(project_id), "ok"
    # 提炼返回空:不覆盖旧记录,如实回报 llm_empty 供上层提示。
    return project_store.get_project(project_id), "llm_empty"


def import_and_track(store, project_store, project_id: str, model: str = None) -> dict:
    """一次性/增量导入某项目的 CC 日志并刷新四维度记录。

    Returns:
        ``{"project_id", "new_messages": int, "profile": UserProfile}``,
        ``new_messages`` 为本次新灌入的消息条数(增量,重复导入应为 0)。
    """
    project = find_project(project_id)
    if project is None:
        return {"project_id": project_id, "new_messages": 0,
                "profile": project_store.get_project(project_id),
                "status": "not_found"}

    import json
    cpath = _cursor_path(project_store)
    cursor = {}
    if os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as f:
                cursor = json.load(f).get(project_id, {})
        except (json.JSONDecodeError, OSError):
            cursor = {}

    new_msgs, updated_cursor = collect_new_messages(project, cursor)

    if new_msgs:
        # mem0 每次 add 按一段对话抽取事实。一次性灌 200+ 条会让抽取失效
        # (返回空),故分小批(约 6 条 ≈ 3 轮)逐批灌入,user+assistant 都入。
        _ingest_in_batches(store, project_id, new_msgs, batch_size=6)
        # 写回游标(整个文件的 project_id 段)。
        all_cursor = {}
        if os.path.exists(cpath):
            try:
                with open(cpath, encoding="utf-8") as f:
                    all_cursor = json.load(f)
            except (json.JSONDecodeError, OSError):
                all_cursor = {}
        all_cursor[project_id] = updated_cursor
        os.makedirs(project_store.base_dir, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(all_cursor, f, ensure_ascii=False, indent=2)

    profile, status = refresh_project(store, project_store, project_id, model=model)
    return {"project_id": project_id, "new_messages": len(new_msgs),
            "profile": profile, "status": status}


def _read_project_messages(project: dict) -> list:
    """读某项目全部会话的原文消息(不去重),用于全量重跑。"""
    msgs = []
    for sess_path in project.get("sessions", []):
        msgs.extend(parse_cc_session(sess_path))
    return msgs


def track_from_dialog(project_store, project_id: str, dim_prompt: str,
                      model: str = None, incremental: bool = True,
                      progress_cb=None, write_lock=None) -> dict:
    """从对话**原文**直接提炼四维并写入(不经 mem0)。

    这是 :func:`import_and_track`(mem0 链路)的替代:让 LLM 读会话原文一次性提炼,
    无损、更准。增量模式只喂游标之后的新消息、与旧四维合并;全量模式读全部对话、
    直接替换。

    Args:
        project_store: ProjectStore。
        project_id: 项目 id(目录 basename)。
        dim_prompt: 四维提取系统提示词(persona_store.load_dim_prompt())。
        model: litellm 模型名。
        incremental: True 只处理新消息并合并;False 读全部对话重提、直接替换。
        progress_cb: 可选回调 ``fn(current, total)``,提炼时每段调用一次,供分段进度显示。
        write_lock: 可选上下文管理器(如 threading.Lock)。仅在**写盘瞬间**加锁,
            LLM 提炼期间**不持锁**——避免长时间独占,冻结其他 UI 事件。

    Returns:
        ``{"project_id","new_messages","status","profile"}``;
        status ∈ ok / no_messages / llm_empty / not_found。
    """
    import contextlib
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    project = find_project(project_id)
    if project is None:
        return {"project_id": project_id, "new_messages": 0,
                "status": "not_found",
                "profile": project_store.get_project(project_id)}

    cpath = _cursor_path(project_store)

    if incremental:
        all_cursor = load_cursor(cpath)
        cursor = all_cursor.get(project_id, {})
        msgs, updated_cursor = collect_new_messages(project, cursor)
    else:
        msgs = _read_project_messages(project)
        updated_cursor = None  # 全量重跑后重建游标(标记全部已处理)

    if not msgs:
        return {"project_id": project_id, "new_messages": 0,
                "status": "no_messages",
                "profile": project_store.get_project(project_id)}

    existing = project_store.get_project(project_id)
    existing_items = [it.to_dict() for it in existing.items] if incremental else None

    fresh = extract_dims_from_messages(
        msgs, dim_prompt, existing_items=existing_items, model=model,
        progress_cb=progress_cb)

    if not fresh:
        # 提炼空(限流/解析失败):不覆盖旧记录,如实回报。
        return {"project_id": project_id, "new_messages": len(msgs),
                "status": "llm_empty",
                "profile": project_store.get_project(project_id)}

    if incremental:
        # 与旧四维合并去重(旧条目 LLM 已在 prompt 里见过,fresh 已含合并结果;
        # 这里再兜底并入旧条目里 fresh 未覆盖的,避免丢历史)。
        merged = {(it["dimension"], _norm(it["text"])): it for it in (existing_items or [])}
        for it in fresh:
            merged[(it.get("dimension"), _norm(it.get("text", "")))] = it
        items_data = list(merged.values())
    else:
        items_data = fresh

    # 仅写盘瞬间加锁;LLM 提炼(上面 extract_dims_from_messages)已在锁外完成。
    lock_ctx = write_lock if write_lock is not None else contextlib.nullcontext()
    with lock_ctx:
        project_store.replace_profile(project_id, items_data)

        # 全量重跑:重建游标,把全部会话标记为已处理(下次增量从这里接续)。
        if not incremental:
            _, updated_cursor = collect_new_messages(project, {})
        all_cursor = load_cursor(cpath)
        all_cursor[project_id] = updated_cursor
        save_cursor(cpath, all_cursor)
        profile = project_store.get_project(project_id)

    return {"project_id": project_id, "new_messages": len(msgs),
            "status": "ok", "profile": profile}


def _norm(text: str) -> str:
    return "".join((text or "").split()).strip("。,.!?;:")


def available_projects() -> list[str]:
    """返回可导入的项目 id 列表(供 UI 下拉框填充)。"""
    return [p["project_id"] for p in list_cc_projects()]
