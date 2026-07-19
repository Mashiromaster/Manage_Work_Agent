"""Mem0 项目记忆 MCP server —— 只读磁盘产物,绝不打开 Qdrant。

运行(stdio)::

    conda run -n mem0 python -m mcp_server.server

注册到 Claude Code::

    claude mcp add mem0-project -- conda run -n mem0 python -m mcp_server.server

工具(全部只读):
- ``list_projects``            列出有记忆/分析产物的项目 id
- ``get_progress``             读项目进度四维(project_tracking/<id>.json)
- ``get_project_memory``       进度四维 + 代码分析摘录,合成一份可读上下文
- ``get_code_analysis``        读代码深度分析报告(code_analysis/<id>.md)
- ``search_code_analysis``     关键词过滤逐文件摘要(无 LLM / 无向量 / 无 Qdrant)
- ``get_structure_analysis``   读浅层结构分析(project_analysis/<id>.md)
- ``list_users``               列出有个性化定制的 user_id
- ``get_user_persona``         用户人设 + 画像,合成系统级提示词(不含 Qdrant 原始记忆)

设计承重点:这里只用 ``ProjectStore.get_project``(纯文件读)与直接读
``code_analysis/*.{md,json}``、``project_analysis/*.md``。**从不** import 或构造
``MemoryStore``,故不会与 Gradio 争抢 Qdrant 单进程锁。
"""

import json
import os

from memory_framework.code_analyzer import (
    CODE_ANALYSIS_DIR,
    load_code_analysis,
)
from memory_framework.persona_store import list_persona_users, load_persona
from memory_framework.profile import DIMENSION_LABELS
from memory_framework.profile_store import ProfileStore, ProjectStore
from memory_framework.project_dims import PROJECT_DIMENSION_LABELS

# 结构分析产物目录(与 repo_analyzer.ANALYSIS_DIR 对齐,避免 import 触发其副作用)。
STRUCTURE_ANALYSIS_DIR = "./project_analysis"
PROJECT_TRACKING_DIR = "./project_tracking"
PROFILE_DIR = "./profiles"


def _safe_name(project_id: str) -> str:
    return project_id.replace("/", "_").replace("\\", "_").strip() or "unnamed"


def _json_stems(directory: str) -> set:
    if not os.path.isdir(directory):
        return set()
    return {f[:-5] for f in os.listdir(directory) if f.endswith(".json")}


def _md_stems(directory: str) -> set:
    if not os.path.isdir(directory):
        return set()
    out = set()
    for f in os.listdir(directory):
        if f.endswith(".md"):
            out.add(f[:-3])
    return out


def list_projects() -> list:
    """列出所有有记忆/分析产物的项目 id(三处目录并集,按名排序)。"""
    names = set()
    names |= _json_stems(PROJECT_TRACKING_DIR)      # 进度四维
    names |= _json_stems(CODE_ANALYSIS_DIR)          # 代码深度分析
    names |= _md_stems(STRUCTURE_ANALYSIS_DIR)       # 结构分析
    # code_analysis 下的 .snapshots 目录不含 .json 直属文件,已被 _json_stems 天然排除
    return sorted(names)


def get_progress(project_id: str) -> dict:
    """读项目进度四维;返回 {dimension_label: [条目文本]}。无则空 dict。

    只经 ProjectStore.get_project——纯文件读,不开 Qdrant。
    """
    store = ProjectStore(base_dir=PROJECT_TRACKING_DIR)
    profile = store.get_project(project_id)
    items = getattr(profile, "items", None) or []
    grouped = {}
    for it in items:
        label = PROJECT_DIMENSION_LABELS.get(it.dimension, it.dimension)
        grouped.setdefault(label, []).append(it.text)
    return grouped


def get_structure_analysis(project_id: str) -> str:
    """读浅层结构分析报告 md;无则提示串。"""
    path = os.path.join(STRUCTURE_ANALYSIS_DIR, f"{_safe_name(project_id)}.md")
    if not os.path.isfile(path):
        return f"(项目 {project_id} 尚无结构分析报告)"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return f"(读取 {project_id} 结构分析失败)"


def get_code_analysis(project_id: str) -> str:
    """读代码深度分析报告 md;无则提示串。"""
    data = load_code_analysis(project_id)
    if not data or not data.get("md"):
        return f"(项目 {project_id} 尚无代码深度分析报告)"
    return data["md"]


def search_code_analysis(project_id: str, query: str) -> str:
    """在代码深度分析的逐文件摘要里做关键词过滤(无 LLM / 无向量)。

    命中 file / role / summary / key_symbols 任一即返回该文件条目。
    """
    data = load_code_analysis(project_id)
    if not data:
        return f"(项目 {project_id} 尚无代码深度分析)"
    summaries = data.get("summaries") or []
    q = (query or "").strip().lower()
    if not q:
        hits = summaries
    else:
        hits = []
        for s in summaries:
            hay = " ".join([
                str(s.get("file", "")),
                str(s.get("role", "")),
                str(s.get("summary", "")),
                " ".join(s.get("key_symbols") or []),
            ]).lower()
            if q in hay:
                hits.append(s)
    if not hits:
        return f"(在 {project_id} 中未找到匹配 “{query}” 的文件)"
    lines = [f"- {s.get('file', '?')} — {s.get('role', '')} :: {s.get('summary', '')}"
             for s in hits[:30]]
    extra = f"\n(共 {len(hits)} 条,显示前 30)" if len(hits) > 30 else ""
    return "\n".join(lines) + extra


def get_project_memory(project_id: str) -> str:
    """合成一份人读的项目长期记忆上下文:进度四维 + 代码分析摘录。

    这是给 Claude Code 最常调的入口——一次拿到项目全貌。
    """
    parts = [f"# 项目长期记忆:{project_id}"]

    progress = get_progress(project_id)
    if progress:
        parts.append("## 开发进度")
        for label, texts in progress.items():
            parts.append(f"### {label}")
            parts.extend(f"- {t}" for t in texts[:12])

    data = load_code_analysis(project_id)
    summaries = (data or {}).get("summaries") or []
    if summaries:
        parts.append("## 代码结构(逐文件摘要摘录)")
        for s in summaries[:25]:
            parts.append(f"- {s.get('file', '?')} — {s.get('role', '')} :: "
                         f"{s.get('summary', '')}")
        if len(summaries) > 25:
            parts.append(f"(另有 {len(summaries) - 25} 个文件,详见 get_code_analysis)")
        graph = (data or {}).get("graph") or {}
        edges = graph.get("edges") or []
        if edges:
            parts.append("## 内部模块依赖(部分)")
            parts.extend(f"- {a} → {b}" for a, b in edges[:20])

    if len(parts) == 1:
        return f"(项目 {project_id} 暂无任何长期记忆产物)"
    return "\n".join(parts)


def list_users() -> list:
    """列出有个性化定制(人设或结构化画像)的 user_id,并集排序。"""
    names = set(list_persona_users())
    if os.path.isdir(PROFILE_DIR):
        names |= {f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".json")}
    return sorted(names)


def get_user_persona(user_id: str = "default_user") -> str:
    """合成一段可当**系统级提示词**的用户个性化上下文,供 Claude Code 直接使用。

    包含:①用户手写的全局人设/规则(persona/<uid>.json)②结构化画像四维摘要
    (profiles/<uid>.json)。**不含** Qdrant 里的逐条原始记忆——MCP 只读磁盘产物、
    绝不打开 Qdrant(单进程独占约束)。若用户关闭了注入开关(inject=False),
    仍返回内容但在开头标注「用户已关闭自动注入」,由调用方决定是否采用。
    """
    cfg = load_persona(user_id)
    parts = [f"# 用户个性化上下文:{user_id}"]
    if not cfg.get("inject", True):
        parts.append("> 注意:用户已在 Mem0 关闭「自动注入」开关,以下内容供参考。")

    persona = (cfg.get("persona") or "").strip()
    if persona:
        parts.append(f"## 用户设定的人设 / 规则(请严格遵循)\n{persona}")

    # 结构化画像(纯文件读,不碰 Qdrant);include_forgotten 保留全部,
    # 画像作人设上下文不应因时间衰减被过滤。
    try:
        profile = ProfileStore(base_dir=PROFILE_DIR).get_profile(
            user_id, include_forgotten=True)
        items = getattr(profile, "items", None) or []
    except Exception:
        items = []
    if items:
        by_dim = {}
        for it in items:
            by_dim.setdefault(it.dimension, []).append(it.text)
        lines = []
        for dim, label in DIMENSION_LABELS.items():
            texts = by_dim.get(dim)
            if not texts:
                continue
            lines.append(f"### {label}")
            lines.extend(f"- {t}" for t in texts[:8])
        if lines:
            parts.append("## 用户画像(从历史对话提炼)\n" + "\n".join(lines))

    if len(parts) == 1 or (len(parts) == 2 and not cfg.get("inject", True)):
        return f"(用户 {user_id} 暂无人设与画像)"
    return "\n\n".join(parts)


def build_server():
    """构造 FastMCP server 并注册只读工具。延迟 import,便于无 mcp 环境下测纯函数。"""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mem0-project")

    @mcp.tool()
    def list_projects_tool() -> list:
        """列出所有有长期记忆/分析产物的项目 id。"""
        return list_projects()

    @mcp.tool()
    def get_project_memory_tool(project_id: str) -> str:
        """获取某项目的长期记忆全貌(开发进度四维 + 代码结构摘录)。"""
        return get_project_memory(project_id)

    @mcp.tool()
    def get_progress_tool(project_id: str) -> dict:
        """获取某项目的开发进度四维(进度/问题/待办/决策)。"""
        return get_progress(project_id)

    @mcp.tool()
    def get_code_analysis_tool(project_id: str) -> str:
        """获取某项目的代码深度分析报告(markdown)。"""
        return get_code_analysis(project_id)

    @mcp.tool()
    def search_code_analysis_tool(project_id: str, query: str) -> str:
        """在某项目的逐文件摘要里按关键词检索相关文件。"""
        return search_code_analysis(project_id, query)

    @mcp.tool()
    def get_structure_analysis_tool(project_id: str) -> str:
        """获取某项目的浅层结构分析报告(markdown)。"""
        return get_structure_analysis(project_id)

    @mcp.tool()
    def list_users_tool() -> list:
        """列出有个性化定制(人设/画像)的用户 id。"""
        return list_users()

    @mcp.tool()
    def get_user_persona_tool(user_id: str = "default_user") -> str:
        """获取某用户的个性化系统级上下文(人设 + 画像),供当作 system prompt。"""
        return get_user_persona(user_id)

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
