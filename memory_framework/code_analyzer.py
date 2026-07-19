"""项目仓库代码深度分析:逐文件 LLM 摘要 + import/依赖关系图(纯 ast)+ 归纳报告。

与 :mod:`repo_analyzer`(浅层结构归纳)的区别:那里让 LLM 看一眼目录树/清单产一份
概览;这里深入到**每个源文件**——先用 Python ``ast`` 静态解析出模块间 import 依赖
(无 LLM),再分批让 LLM 为每个文件产一句职责摘要,最后一次性归纳成完整代码报告。

成本护栏(硬约束):
- ``MAX_FILES`` 上限:超出的文件不喂 LLM,但在 ``capinfo.skipped`` 与报告 md 里显式列出;
- ``FILE_BATCH`` 批处理:每次 LLM 调用摘要多个文件,压低调用次数;
- 增量:``deep_analyze(changed_only=...)`` 只摘要变更文件,其余复用旧结果。
最坏 LLM 调用数 = ceil(min(N, MAX_FILES) / FILE_BATCH) + 1。

产物:``code_analysis/<名>.md``(人读报告)+ ``code_analysis/<名>.json``(结构化,供 MCP/
增量合并读取)。纯逻辑 + 磁盘 I/O,LLM 可 mock,便于单测。
"""

import ast
import json
import os
import re
import time
from datetime import datetime

import litellm

from memory_framework.config import DEFAULT_LLM_MODEL
from memory_framework.repo_scan import _IGNORE_DIRS, _IGNORE_FILES, _read_text, scan_repo

CODE_ANALYSIS_DIR = "./code_analysis"

# 成本护栏。
MAX_FILES = 120          # 最多喂给 LLM 的源文件数,超出显式截断记录。
MAX_FILE_CHARS = 6000    # 单文件读取上限(喂 LLM 的内容)。
FILE_BATCH = 4           # 每次 LLM 调用摘要的文件数。
LLM_RETRIES = 3          # LLM 调用失败(如 429 限流)重试次数。
LLM_RETRY_BASE = 2.0     # 指数退避基数(秒):2、4、8…


def _complete(messages: list, model: str, max_tokens: int,
              retries: int = LLM_RETRIES) -> str:
    """带指数退避重试的 LLM 调用,返回文本内容;彻底失败返回空串。

    深度分析要串十几次调用,单次撞 429/网络抖动很常见。这里对每次调用做
    最多 ``retries`` 次退避重试,把瞬时失败挡在外面,避免一撞限流就丢结果。
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = litellm.completion(
                model=model, messages=messages, temperature=1,
                max_tokens=max_tokens,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # noqa: BLE001 —— 限流/网络/解析都退避重试
            last_exc = exc
            if attempt < retries:
                time.sleep(LLM_RETRY_BASE * (2 ** attempt))
    if last_exc is not None:
        print(f"[code_analyzer] LLM 调用重试 {retries} 次后仍失败:{last_exc}")
    return ""


# 视为「源码」的扩展名(其余当资源/配置,不逐文件摘要)。
_SOURCE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".vue", ".svelte",
}

FILE_SUMMARY_PROMPT = """你是资深代码审阅者。下面给出若干源文件的路径与内容(可能已截断)。
为**每个文件**产出一句中文职责摘要与关键符号。

严格输出 JSON 数组(不要 markdown 代码块,不要多余文字):

[{"file": "相对路径", "role": "这个文件在项目中的角色(一句话)", "key_symbols": ["主要函数/类名"], "summary": "它做什么(一两句)"}]

原则:
- file 必须原样回填给定的相对路径。
- 基于给定内容判断,不臆造不存在的符号。
- 简洁、准确、可扫读。
"""

REPORT_PROMPT = """你是资深软件架构师。基于给定的项目目录树、逐文件职责摘要、模块 import
依赖关系,输出一份**结构化的中文 markdown 代码分析报告**。

严格用以下 markdown 结构(不要用代码块包裹整篇):

## 项目概述
这个项目整体做什么、核心能力是什么(从文件摘要归纳)。

## 关键路径走读
挑 1-3 条主要执行/数据流路径,按调用顺序讲清楚从入口到落地经过哪些文件/函数。

## 依赖关系
基于给定的 import 依赖,说明模块如何相互依赖、有无核心枢纽模块、有无循环依赖迹象。

## 逐文件摘要
用 markdown 表格列出:文件 | 角色 | 摘要。覆盖给定的所有已分析文件。

## 关键文件
挑最值得关注的入口/核心文件展开一句说明。

原则:基于给定信息合理推断,不编造;简洁准确;用中文。信息不足如实说明。
"""


def collect_source_files(root: str, changed_only: list = None) -> list:
    """遍历项目目录,收集源码文件清单(纯 FS,不读全文)。

    Args:
        root: 项目根目录。
        changed_only: 若给定(相对路径列表),只保留其中的文件(增量分析用)。

    Returns:
        ``[{"rel", "abs", "ext", "size"}, ...]``,按 rel 排序。非目录返回 []。
    """
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        return []
    changed_set = set(changed_only) if changed_only is not None else None

    files = []
    for cur, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for name in names:
            if name in _IGNORE_FILES:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in _SOURCE_EXTS:
                continue
            abs_path = os.path.join(cur, name)
            rel = os.path.relpath(abs_path, root)
            if changed_set is not None and rel not in changed_set:
                continue
            try:
                size = os.path.getsize(abs_path)
            except OSError:
                continue
            files.append({"rel": rel, "abs": abs_path, "ext": ext, "size": size})
    files.sort(key=lambda f: f["rel"])
    return files


def _module_key(rel: str) -> str:
    """相对路径 → 归一化模块键(去扩展名、/ 转 .、去 __init__)。"""
    no_ext = os.path.splitext(rel)[0]
    parts = [p for p in no_ext.replace("\\", "/").split("/") if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_import_graph(files: list) -> dict:
    """用 ast 静态解析 .py 文件的 import,构建内部依赖图(无 LLM)。

    只连内部边:import 目标若能对应到本项目某个模块键(前缀匹配)则算内部依赖,
    否则计入 external。无法 parse 的文件计入 unparsed。

    Returns:
        ``{"nodes": [模块键], "edges": [[src, dst]], "external": [pkg],
          "unparsed": [rel]}``。
    """
    py = [f for f in files if f["ext"] == ".py"]
    node_keys = {_module_key(f["rel"]) for f in py}
    nodes = sorted(k for k in node_keys if k)
    edges = set()
    external = set()
    unparsed = []

    def _resolve(target: str) -> str | None:
        """把 import 目标解析成本项目内部模块键(最长前缀匹配)。"""
        if not target:
            return None
        if target in node_keys:
            return target
        # a.b.c → 命中 a.b 或 a
        parts = target.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in node_keys:
                return cand
        return None

    for f in py:
        src = _module_key(f["rel"])
        text = _read_text(f["abs"], MAX_FILE_CHARS * 2)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            unparsed.append(f["rel"])
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # 相对 import(level>0)只能对应本项目内部,用 module 名匹配。
                if node.module:
                    targets = [node.module]
            for t in targets:
                dst = _resolve(t)
                if dst and dst != src:
                    edges.add((src, dst))
                elif dst is None:
                    external.add(t.split(".")[0])

    return {
        "nodes": nodes,
        "edges": sorted([list(e) for e in edges]),
        "external": sorted(external),
        "unparsed": sorted(unparsed),
    }


def _file_block(f: dict) -> str:
    """把单文件拼成喂给 LLM 的文本块(路径 + 截断内容)。"""
    content = _read_text(f["abs"], MAX_FILE_CHARS)
    return f"### 文件:{f['rel']}\n```\n{content}\n```"


def summarize_files(files: list, model: str = None, cap: int = MAX_FILES) -> tuple:
    """分批让 LLM 为每个文件产职责摘要;超出 cap 的文件显式截断记录。

    Returns:
        ``(summaries, capinfo)``:
        summaries = ``[{"file", "role", "key_symbols", "summary"}, ...]``;
        capinfo   = ``{"total", "analyzed", "skipped": [rel...], "llm_calls"}``。
    """
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    total = len(files)
    analyzed = files[:cap]
    skipped = [f["rel"] for f in files[cap:]]

    summaries = []
    llm_calls = 0
    failed_files = []
    for i in range(0, len(analyzed), FILE_BATCH):
        batch = analyzed[i:i + FILE_BATCH]
        user_msg = ("请为下列文件分别产出职责摘要,直接输出 JSON 数组。\n\n"
                    + "\n\n".join(_file_block(f) for f in batch))
        messages = [
            {"role": "system", "content": FILE_SUMMARY_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        llm_calls += 1
        raw = _complete(messages, model, max_tokens=2048)
        parsed = _parse_code_summaries(raw) if raw else []
        # 用返回结果按 file 回填;LLM 漏掉的文件补一条占位,保证覆盖。
        by_file = {s.get("file"): s for s in parsed if isinstance(s, dict)}
        for f in batch:
            s = by_file.get(f["rel"])
            if s:
                s["file"] = f["rel"]
                summaries.append(s)
            else:
                failed_files.append(f["rel"])
                summaries.append({"file": f["rel"], "role": "", "key_symbols": [],
                                  "summary": "(未获得摘要)"})

    capinfo = {
        "total": total,
        "analyzed": len(analyzed),
        "skipped": skipped,
        "llm_calls": llm_calls,
        "failed_files": failed_files,
    }
    return summaries, capinfo


def _parse_code_summaries(raw: str) -> list:
    """从 LLM 回复中提取文件摘要 JSON 数组(三级:直接 / ```json``` / 首个 [...])。

    不复用 profile 的 _parse_json_response——那个按 dimension 字段校验,会丢弃这里
    的 file/summary 结构。故自带一套宽松提取,只要顶层是 list[dict] 即接受。
    """
    def _try(s: str) -> list | None:
        try:
            data = json.loads(s)
            return data if isinstance(data, list) else None
        except (json.JSONDecodeError, ValueError):
            return None

    got = _try(raw.strip())
    if got is not None:
        return got
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m and (got := _try(m.group(1))) is not None:
        return got
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m and (got := _try(m.group(0))) is not None:
        return got
    return []


def _summaries_to_prompt(tree: str, summaries: list, graph: dict,
                         capinfo: dict) -> str:
    parts = []
    if tree:
        parts.append(f"## 目录树\n```\n{tree}\n```")
    if summaries:
        lines = "\n".join(
            f"- {s.get('file', '?')} — {s.get('role', '')} :: {s.get('summary', '')}"
            for s in summaries
        )
        parts.append(f"## 逐文件摘要\n{lines}")
    if graph and graph.get("edges"):
        edge_lines = "\n".join(f"- {a} → {b}" for a, b in graph["edges"])
        parts.append(f"## 内部 import 依赖\n{edge_lines}")
    if graph and graph.get("external"):
        parts.append("## 外部依赖(部分)\n" + ", ".join(graph["external"][:40]))
    if capinfo and capinfo.get("skipped"):
        parts.append(f"## 未分析(超出上限 {capinfo['analyzed']} 个文件)\n"
                     + "\n".join(f"- {p}" for p in capinfo["skipped"]))
    parts.append("\n请据此输出结构化的 markdown 代码分析报告。")
    return "\n\n".join(parts)


def synthesize_report(project_name: str, tree: str, summaries: list,
                      graph: dict, capinfo: dict, model: str = None) -> str:
    """终一次 LLM 调用:把逐文件摘要 + 依赖图归纳成完整 markdown 报告。"""
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    messages = [
        {"role": "system", "content": REPORT_PROMPT},
        {"role": "user", "content": f"# 项目:{project_name}\n\n"
         + _summaries_to_prompt(tree, summaries, graph, capinfo)},
    ]
    return _complete(messages, model, max_tokens=3000)


def _fallback_report(project_name: str, summaries: list, graph: dict,
                     capinfo: dict) -> str:
    """归纳 LLM 调用失败时的降级报告:直接把已得的逐文件摘要 + 依赖图拼成 md。

    避免「11 个文件摘要成功,只因终归纳那一次撞限流就整份作废」。
    """
    parts = [f"## 项目概述\n> ⚠️ 归纳步骤未能生成(可能限流/网络问题),以下为已成功"
             f"提取的逐文件摘要与依赖关系。可稍后重试以获得完整报告。\n"]
    if summaries:
        rows = ["| 文件 | 角色 | 摘要 |", "| --- | --- | --- |"]
        for s in summaries:
            f = str(s.get("file", "?")).replace("|", "\\|")
            role = str(s.get("role", "")).replace("|", "\\|")
            summ = str(s.get("summary", "")).replace("|", "\\|")
            rows.append(f"| {f} | {role} | {summ} |")
        parts.append("## 逐文件摘要\n" + "\n".join(rows))
    edges = (graph or {}).get("edges") or []
    if edges:
        parts.append("## 内部依赖\n" + "\n".join(f"- {a} → {b}" for a, b in edges))
    return "\n\n".join(parts)


def _truncation_note(capinfo: dict) -> str:
    """把截断信息渲染成报告尾部的显式清单(不静默丢弃)。"""
    skipped = capinfo.get("skipped") or []
    if not skipped:
        return ""
    listing = "\n".join(f"- `{p}`" for p in skipped)
    return (f"\n\n---\n\n## ⚠️ 已截断(未分析)\n"
            f"本次共发现 {capinfo.get('total', 0)} 个源文件,超过单次分析上限 "
            f"{capinfo.get('analyzed', 0)} 个,以下文件未纳入本次逐文件分析:\n\n{listing}")


def _failure_note(capinfo: dict) -> str:
    """把摘要失败的文件渲染成报告尾部清单(限流/网络导致,可重试补齐)。"""
    failed = capinfo.get("failed_files") or []
    if not failed:
        return ""
    listing = "\n".join(f"- `{p}`" for p in failed)
    return (f"\n\n---\n\n## ⚠️ 摘要未获取(可重试)\n"
            f"以下 {len(failed)} 个文件本次未取得摘要(可能限流/网络问题),"
            f"稍后再点「深度分析」将自动重试补齐:\n\n{listing}")


def deep_analyze(root: str, project_name: str = "", model: str = None,
                 changed_only: list = None) -> dict:
    """编排:收集文件 → ast 依赖图 → 分批 LLM 摘要 → 归纳报告。

    Args:
        root: 项目根目录。
        project_name: 项目名(默认取目录 basename)。
        model: litellm 模型名。
        changed_only: 增量分析,仅摘要这些相对路径的文件;其余复用上次已存摘要,
            并把本次结果合并进旧 json。None = 全量。

    Returns:
        ``{"md", "structured", "capinfo"}``;目录非法或无源文件时 md 为空、
        structured 带空清单。structured = ``{summaries, graph, capinfo}``。
    """
    root = os.path.normpath(root)
    project_name = project_name or os.path.basename(root)
    empty = {"md": "", "structured": {"summaries": [], "graph": {}, "capinfo": {}},
             "capinfo": {}}
    if not os.path.isdir(root):
        return empty

    all_files = collect_source_files(root)
    if not all_files:
        return empty

    graph = build_import_graph(all_files)  # 依赖图始终基于全量,更准确

    if changed_only is not None:
        # 增量:只摘要变更文件,与旧摘要合并(旧结果里已删除的文件剔除)。
        target = [f for f in all_files if f["rel"] in set(changed_only)]
        fresh, capinfo = summarize_files(target, model=model)
        prev = load_code_analysis(project_name) or {}
        prev_summaries = prev.get("summaries", [])
        current_rels = {f["rel"] for f in all_files}
        merged = {s.get("file"): s for s in prev_summaries
                  if s.get("file") in current_rels}
        for s in fresh:
            merged[s.get("file")] = s
        summaries = [merged[k] for k in sorted(merged)]
        capinfo["incremental"] = True
        capinfo["reanalyzed"] = [f["rel"] for f in target]
    else:
        summaries, capinfo = summarize_files(all_files, model=model)

    tree = scan_repo(root).get("tree", "")
    md = synthesize_report(project_name, tree, summaries, graph, capinfo, model=model)
    degraded = not md
    if degraded:
        # 归纳失败:不丢弃已成功的逐文件摘要,拼一份降级报告落盘。
        md = _fallback_report(project_name, summaries, graph, capinfo)
    md = md + _truncation_note(capinfo) + _failure_note(capinfo)
    capinfo["degraded"] = degraded

    structured = {"summaries": summaries, "graph": graph, "capinfo": capinfo}
    return {"md": md, "structured": structured, "capinfo": capinfo}


def _safe_name(project_name: str) -> str:
    return project_name.replace("/", "_").replace("\\", "_").strip() or "unnamed"


def _md_path(project_name: str) -> str:
    return os.path.join(CODE_ANALYSIS_DIR, f"{_safe_name(project_name)}.md")


def _json_path(project_name: str) -> str:
    return os.path.join(CODE_ANALYSIS_DIR, f"{_safe_name(project_name)}.json")


def save_code_analysis(project_name: str, md: str, structured: dict,
                       source_path: str = "", now: datetime = None) -> tuple:
    """写盘:md(人读)+ json(结构化)。返回 (md_path, json_path)。"""
    os.makedirs(CODE_ANALYSIS_DIR, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    header = (f"<!-- 项目:{project_name} | 源路径:{source_path} | "
              f"更新时间:{stamp} -->\n\n"
              f"> 源路径 `{source_path}` · 代码深度分析 · 更新于 {stamp}\n\n")
    md_path = _md_path(project_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header + (md or "") + "\n")

    payload = {
        "project_name": project_name,
        "source_path": source_path,
        "updated_at": stamp,
        **(structured or {}),
    }
    json_path = _json_path(project_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return md_path, json_path


def load_code_analysis(project_name: str) -> dict | None:
    """读结构化分析 json(附带 md 文本);不存在返回 None。"""
    jp = _json_path(project_name)
    if not os.path.isfile(jp):
        return None
    try:
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    mp = _md_path(project_name)
    if os.path.isfile(mp):
        try:
            with open(mp, encoding="utf-8") as f:
                data["md"] = f.read()
        except OSError:
            pass
    return data


def list_code_analyzed() -> list:
    """列出已做过代码深度分析的项目名(按名排序)。"""
    if not os.path.isdir(CODE_ANALYSIS_DIR):
        return []
    names = [f[:-5] for f in os.listdir(CODE_ANALYSIS_DIR) if f.endswith(".json")]
    return sorted(names)
