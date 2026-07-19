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
)
from memory_framework.config import DEFAULT_LLM_MODEL
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


def available_projects() -> list[str]:
    """返回可导入的项目 id 列表(供 UI 下拉框填充)。"""
    return [p["project_id"] for p in list_cc_projects()]
