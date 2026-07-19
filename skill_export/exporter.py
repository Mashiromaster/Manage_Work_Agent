"""Skill 导出器:把项目长期记忆写成 Claude Code Skill 目录结构。

产物::

    <target>/<project_id>/
        SKILL.md                     # YAML frontmatter + 使用说明,引用 reference/*
        reference/progress.md        # 进度四维
        reference/structure.md       # 结构分析(若有)
        reference/code_analysis.md   # 代码深度分析报告(若有)

全部从盘上产物读取,不调 LLM、不开 Qdrant。
"""

import os

from mcp_server.server import (
    get_code_analysis,
    get_progress,
    get_structure_analysis,
)

DEFAULT_TARGET = ".claude/skills"


def _safe_name(project_id: str) -> str:
    return project_id.replace("/", "_").replace("\\", "_").strip() or "unnamed"


def _skill_slug(project_id: str) -> str:
    """SKILL frontmatter 的 name:小写、非字母数字转连字符。"""
    s = _safe_name(project_id).lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "project"


def _render_progress_md(project_id: str) -> str:
    prog = get_progress(project_id)
    if not prog:
        return "# 开发进度\n\n(暂无进度记录)\n"
    parts = ["# 开发进度\n"]
    for label, texts in prog.items():
        parts.append(f"## {label}")
        parts.extend(f"- {t}" for t in texts)
        parts.append("")
    return "\n".join(parts)


def export_skill(project_id: str, target: str = DEFAULT_TARGET) -> str:
    """把项目长期记忆导出为 Skill,返回 SKILL.md 路径。

    Args:
        project_id: 项目 id(与 project_tracking/code_analysis 里的名一致)。
        target: skills 根目录,默认 ``.claude/skills``。

    Returns:
        写出的 SKILL.md 绝对/相对路径。
    """
    name = _safe_name(project_id)
    skill_dir = os.path.join(target, name)
    ref_dir = os.path.join(skill_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    # reference 文件
    progress_md = _render_progress_md(project_id)
    structure_md = get_structure_analysis(project_id)
    code_md = get_code_analysis(project_id)

    with open(os.path.join(ref_dir, "progress.md"), "w", encoding="utf-8") as f:
        f.write(progress_md)
    with open(os.path.join(ref_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write(structure_md + "\n")
    with open(os.path.join(ref_dir, "code_analysis.md"), "w", encoding="utf-8") as f:
        f.write(code_md + "\n")

    slug = _skill_slug(project_id)
    description = (
        f"Use when working on the {project_id} project — provides its long-term "
        f"memory: development progress (progress/blockers/todos/decisions), "
        f"repository structure, and deep per-file code analysis. Load the reference "
        f"files to understand project context before answering."
    )
    skill_md = f"""---
name: {slug}
description: {description}
---

# {project_id} 项目长期记忆

本 Skill 由 Mem0 项目工作台导出,包含 {project_id} 的开发长期记忆。回答与本项目
相关的问题前,按需加载下列引用文件以获取上下文。

## 引用文件

- `reference/progress.md` —— 开发进度四维(进度 / 问题 / 待办 / 决策)。
- `reference/structure.md` —— 项目仓库结构分析。
- `reference/code_analysis.md` —— 代码深度分析:逐文件职责摘要、模块依赖、关键路径。

## 使用建议

1. 被问及「做到哪了 / 还有什么待办 / 为什么这么设计」时,读 `progress.md`。
2. 被问及「项目怎么组织 / 某功能在哪个文件」时,读 `structure.md` 与
   `code_analysis.md`。
3. 这份记忆是导出时的快照;如项目已变更,以当前代码为准,并提示用户在 Mem0
   工作台重新分析、重新导出。
"""
    skill_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(skill_md)
    return skill_path
