"""项目结构分析:把 repo_scan 快照交给 LLM 归纳成 markdown,并落盘/读取。

产物是一份人类可读的 markdown 分析报告(项目概述/技术栈/模块划分/目录结构说明/
关键文件),存到 ``./project_analysis/<项目名>.md``。支持手动重跑覆盖更新,界面每次
打开自动读取展示。

与 :mod:`project_extractor` 的区别:那里让 LLM 产 JSON 供结构化存储,这里让 LLM
直接产 markdown 供人阅读,故不走 JSON 校验。
"""

import os
from datetime import datetime

import litellm

from memory_framework.config import DEFAULT_LLM_MODEL

ANALYSIS_DIR = "./project_analysis"

ANALYSIS_PROMPT = """你是一个资深软件架构师。根据给定的项目目录快照(目录树、依赖清单、
README、语言分布、入口文件),输出一份**结构化的中文 markdown 项目分析报告**。

## 输出要求

严格用以下 markdown 结构(不要代码块包裹整篇,直接输出 markdown):

## 项目概述
一两句话说明这个项目是做什么的、解决什么问题(从 README / 目录 / 清单推断)。

## 技术栈
列出语言、框架、关键依赖库(从依赖清单与文件扩展名推断)。用简短要点。

## 模块划分
按目录/包归纳主要模块及其职责。每个模块一行:`- 模块名 — 职责`。

## 目录结构说明
挑关键目录/文件解释其作用(不必逐一列全,聚焦重要的)。

## 关键文件
列出入口文件、配置文件等值得关注的文件及其用途。

## 原则
- 基于给定信息合理推断,不要编造不存在的技术或文件。
- 简洁、准确、可扫读;用中文。
- 如果信息不足以判断某节,如实说明"信息不足"而非杜撰。
"""


def _snapshot_to_prompt(snapshot: dict) -> str:
    """把 scan_repo 快照拼成给 LLM 的 user 消息。"""
    parts = [f"# 项目:{snapshot.get('project_name', '')}"]

    tree = snapshot.get("tree", "")
    if tree:
        parts.append(f"## 目录树\n```\n{tree}\n```")

    lang = snapshot.get("lang_stats") or {}
    if lang:
        lang_line = ", ".join(f"{ext}×{n}" for ext, n in
                              sorted(lang.items(), key=lambda kv: -kv[1]))
        parts.append(f"## 文件类型分布\n{lang_line}")

    entries = snapshot.get("entry_files") or []
    if entries:
        parts.append("## 候选入口文件\n" + "\n".join(f"- {e}" for e in entries))

    manifests = snapshot.get("manifests") or {}
    for name, content in manifests.items():
        parts.append(f"## 依赖清单:{name}\n```\n{content}\n```")

    readme = snapshot.get("readme", "")
    if readme.strip():
        parts.append(f"## README(节选)\n```\n{readme}\n```")

    parts.append("\n请根据以上信息输出结构化的 markdown 项目分析报告。")
    return "\n\n".join(parts)


def analyze_project_structure(snapshot: dict, project_name: str = "",
                              model: str = None) -> str:
    """用 LLM 把项目快照归纳成 markdown 分析报告。

    Args:
        snapshot: :func:`repo_scan.scan_repo` 的返回。
        project_name: 项目名(仅用于日志/兜底,内容以 snapshot 为准)。
        model: litellm 模型名,默认取环境变量。

    Returns:
        markdown 文本;LLM 调用失败或快照为空时返回空串。
    """
    if not snapshot or not snapshot.get("tree"):
        return ""
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": _snapshot_to_prompt(snapshot)},
    ]
    try:
        resp = litellm.completion(
            model=model, messages=messages, temperature=1, max_tokens=3000,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def _safe_name(project_name: str) -> str:
    return project_name.replace("/", "_").replace("\\", "_").strip() or "unnamed"


def _md_path(project_name: str) -> str:
    return os.path.join(ANALYSIS_DIR, f"{_safe_name(project_name)}.md")


def save_analysis(project_name: str, md: str, source_path: str = "",
                  now: datetime = None) -> str:
    """把 markdown 分析报告写盘,顶部加源路径与更新时间元信息。

    Returns:
        写入的 md 文件路径。
    """
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    header = (f"<!-- 项目:{project_name} | 源路径:{source_path} | "
              f"更新时间:{stamp} -->\n\n"
              f"> 源路径 `{source_path}` · 更新于 {stamp}\n\n")
    path = _md_path(project_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + md + "\n")
    return path


def load_analysis(project_name: str) -> str | None:
    """读取已有分析 md;不存在返回 None。"""
    path = _md_path(project_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def list_analyzed_projects() -> list[str]:
    """列出 project_analysis/ 下已分析的项目名(按名排序)。"""
    if not os.path.isdir(ANALYSIS_DIR):
        return []
    names = [f[:-3] for f in os.listdir(ANALYSIS_DIR) if f.endswith(".md")]
    return sorted(names)
