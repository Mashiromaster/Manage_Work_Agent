"""Claude Code 对话日志解析与导入。

Claude Code 把每个项目的会话写在
``~/.claude/projects/<转义后的项目路径>/<sessionId>.jsonl``,每行一个 JSON。
真实对话是 ``type`` 为 ``user`` / ``assistant`` 的行,其余(``ai-title`` /
``mode`` / ``attachment`` / ``system`` 等)是元数据,需过滤。``message.content``
可能是纯字符串,也可能是内容块列表(取 ``type == "text"`` 的块拼接)。

本模块负责:
- 把一个会话文件解析成干净的 ``[{role, content, uuid, timestamp}]``;
- 枚举 ``~/.claude/projects/`` 下可导入的项目及其会话文件;
- 从转义目录名还原出可读的 project_id;
- 维护增量游标(按消息 uuid),让重复导入只处理新增消息。
"""

import json
import os

# Claude Code 会话根目录。
CC_PROJECTS_ROOT = os.path.expanduser("~/.claude/projects")

# 系统注入 / 命令类文本的前缀:这些不是用户真实说的话,应丢弃。
_SYSTEM_TEXT_PREFIXES = (
    "<system-reminder",
    "<command-name",
    "<command-message",
    "<command-args",
    "<local-command",
    "<user-prompt-submit-hook",
    "<task-notification",
    "Caveat:",
)


def _extract_text(content) -> str:
    """从 message.content(str 或内容块 list)中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _is_system_text(text: str) -> bool:
    """判断一段文本是否是系统注入 / 命令包装(应丢弃)。"""
    return text.lstrip().startswith(_SYSTEM_TEXT_PREFIXES)


def parse_cc_session(jsonl_path: str) -> list[dict]:
    """解析一个 CC 会话文件,返回干净的对话消息列表。

    过滤规则:
    - 只保留 ``type in ("user", "assistant")`` 的行;
    - 跳过 ``isSidechain`` 为真的行(子代理侧链,非主对话);
    - 提取纯文本 content,跳过空文本与系统注入 / 命令包装文本。

    Returns:
        ``[{"role", "content", "uuid", "timestamp"}]``,按文件顺序。
    """
    if not os.path.exists(jsonl_path):
        return []
    msgs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") not in ("user", "assistant"):
                continue
            if d.get("isSidechain"):
                continue
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _extract_text(msg.get("content")).strip()
            if not text or _is_system_text(text):
                continue
            msgs.append({
                "role": role,
                "content": text,
                "uuid": d.get("uuid", ""),
                "timestamp": d.get("timestamp", ""),
            })
    return msgs


def derive_project_id(project_dir_name: str) -> str:
    """从转义目录名还原可读 project_id。

    CC 用 ``-Users-mashiro-Mem0`` 表示 ``/Users/mashiro/Mem0``。取最后一段
    作为可读项目名(如 ``Mem0``);无法解析时回退到原目录名。
    """
    name = project_dir_name.strip()
    if not name:
        return project_dir_name
    segments = [s for s in name.split("-") if s]
    return segments[-1] if segments else project_dir_name


def derive_project_path(project_dir_name: str) -> str:
    """从转义目录名还原项目在磁盘上的绝对路径。

    CC 用 ``-Users-mashiro-Mem0`` 表示 ``/Users/mashiro/Mem0``:首字符 ``-`` 表示
    绝对路径的根 ``/``,其余 ``-`` 还原为路径分隔符。仅当还原出的路径确为存在的
    目录时才返回,否则回退空串(无法可靠还原,如目录名本身含 ``-`` 的项目)。
    """
    name = (project_dir_name or "").strip()
    if not name.startswith("-"):
        return ""
    candidate = "/" + name[1:].replace("-", "/")
    return candidate if os.path.isdir(candidate) else ""


def list_cc_projects(root: str = CC_PROJECTS_ROOT) -> list[dict]:
    """枚举 CC 项目目录及其会话文件。

    Returns:
        ``[{"project_id", "dir_name", "dir", "sessions": [jsonl 路径...]}]``,
        仅包含含至少一个 .jsonl 的目录,按 project_id 排序。
    """
    if not os.path.isdir(root):
        return []
    projects = []
    for entry in sorted(os.listdir(root)):
        pdir = os.path.join(root, entry)
        if not os.path.isdir(pdir):
            continue
        sessions = sorted(
            os.path.join(pdir, f)
            for f in os.listdir(pdir)
            if f.endswith(".jsonl")
        )
        if not sessions:
            continue
        projects.append({
            "project_id": derive_project_id(entry),
            "dir_name": entry,
            "dir": pdir,
            "path": derive_project_path(entry),
            "sessions": sessions,
        })
    return projects


def find_project(project_id: str, root: str = CC_PROJECTS_ROOT) -> dict | None:
    """按可读 project_id 查找对应项目条目,找不到返回 None。"""
    for p in list_cc_projects(root):
        if p["project_id"] == project_id:
            return p
    return None


# ---- 增量游标:记录每个会话文件已导入过的消息 key ----

def load_cursor(cursor_path: str) -> dict:
    """读取增量游标(session 路径 → 已处理的 key 列表)。"""
    if not os.path.exists(cursor_path):
        return {}
    try:
        with open(cursor_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cursor(cursor_path: str, cursor: dict) -> None:
    """写回增量游标。"""
    os.makedirs(os.path.dirname(cursor_path) or ".", exist_ok=True)
    with open(cursor_path, "w", encoding="utf-8") as f:
        json.dump(cursor, f, ensure_ascii=False, indent=2)


def _msg_key(m: dict) -> str:
    """消息去重键:优先用 uuid,无 uuid 时用 role+内容哈希兜底。"""
    return m.get("uuid") or f"{m['role']}:{hash(m['content'])}"


def collect_new_messages(project: dict, cursor: dict) -> tuple[list[dict], dict]:
    """收集某项目所有会话里游标之后的新消息,返回 (新消息列表, 更新后游标)。

    游标结构::

        {"<session 路径>": ["已导入的 key", ...]}

    只返回未在游标里出现过的消息(按 key 去重),并把它们并入游标。
    """
    new_msgs = []
    updated = dict(cursor)
    for sess_path in project.get("sessions", []):
        seen = set(updated.get(sess_path, []))
        added_here = []
        for m in parse_cc_session(sess_path):
            key = _msg_key(m)
            if key in seen:
                continue
            seen.add(key)
            added_here.append(key)
            new_msgs.append(m)
        if added_here:
            updated[sess_path] = list(updated.get(sess_path, [])) + added_here
    return new_msgs, updated
