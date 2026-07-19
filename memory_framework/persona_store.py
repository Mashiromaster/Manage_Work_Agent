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
    return sorted(f[:-5] for f in os.listdir(PERSONA_DIR) if f.endswith(".json"))
