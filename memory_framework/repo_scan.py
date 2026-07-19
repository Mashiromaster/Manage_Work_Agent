"""项目代码目录扫描(纯文件系统,不调用 LLM)。

给定一个项目根目录,产出一份结构化快照,供 LLM 归纳项目结构/技术栈/模块:

- ``tree``:缩进式目录树文本(过滤 .git/node_modules 等,限深度与条目数);
- ``manifests``:依赖清单文件内容(requirements.txt / package.json 等,截断);
- ``readme``:README 前若干字符;
- ``lang_stats``:按扩展名的文件计数(粗略语言分布);
- ``entry_files``:候选入口文件名单(仅路径,不读全文,控 token)。

设计约束:限深(``max_depth``)、限条目(``max_entries``)、截断文件内容,避免大
仓库把 LLM prompt 撑爆。纯函数、无副作用,可对临时目录直接单测。
"""

import os

# 扫描时跳过的目录(依赖/构建产物/VCS/本项目自身的数据目录)。
_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "target", ".next", ".nuxt", "coverage",
    "qdrant_data", "project_tracking", "project_analysis", ".DS_Store",
    "code_analysis", ".claude",
}

# 扫描时跳过的文件(系统噪声/敏感本地配置,不列入树也不喂给 LLM)。
_IGNORE_FILES = {".DS_Store", "settings.local.json", ".env"}

# 依赖/项目清单文件:读内容交给 LLM 判断技术栈。
_MANIFEST_FILES = (
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Pipfile", "package.json", "go.mod", "Cargo.toml", "pom.xml",
    "build.gradle", "Gemfile", "composer.json", "environment.yml",
)

# README 候选名(不区分大小写匹配前缀)。
_README_NAMES = ("readme.md", "readme.rst", "readme.txt", "readme")

# 单个文件读取上限,防止超长清单/README 撑爆 prompt。
_MAX_FILE_CHARS = 3000
_README_CHARS = 2000

# 视为「入口/核心」的顶层文件名。
_ENTRY_HINTS = (
    "main.py", "app.py", "__main__.py", "cli.py", "server.py",
    "index.js", "index.ts", "main.go", "main.rs", "manage.py",
)


def _read_text(path: str, limit: int) -> str:
    """读取文本文件前 ``limit`` 字符,失败返回空串。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _build_tree(root: str, max_depth: int, max_entries: int) -> tuple[str, dict, list]:
    """遍历目录,同时产出树文本、语言统计、入口候选。

    Returns:
        (tree_text, lang_stats, entry_files)
    """
    lines = []
    lang_stats: dict[str, int] = {}
    entry_files: list[str] = []
    count = 0
    truncated = False

    def walk(cur: str, depth: int, prefix: str) -> None:
        nonlocal count, truncated
        if truncated or depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(cur))
        except OSError:
            return
        dirs = [e for e in entries
                if os.path.isdir(os.path.join(cur, e)) and e not in _IGNORE_DIRS]
        files = [e for e in entries
                 if os.path.isfile(os.path.join(cur, e)) and e not in _IGNORE_FILES]

        for d in dirs:
            if count >= max_entries:
                truncated = True
                return
            count += 1
            lines.append(f"{prefix}{d}/")
            walk(os.path.join(cur, d), depth + 1, prefix + "  ")

        for fname in files:
            if count >= max_entries:
                truncated = True
                return
            count += 1
            lines.append(f"{prefix}{fname}")
            ext = os.path.splitext(fname)[1].lower()
            if ext:
                lang_stats[ext] = lang_stats.get(ext, 0) + 1
            rel = os.path.relpath(os.path.join(cur, fname), root)
            if fname.lower() in _ENTRY_HINTS:
                entry_files.append(rel)

    base = os.path.basename(os.path.normpath(root))
    lines.append(f"{base}/")
    walk(root, 1, "  ")
    if truncated:
        lines.append(f"  … (超过 {max_entries} 条,已截断)")
    return "\n".join(lines), lang_stats, entry_files


def scan_repo(root: str, max_depth: int = 3, max_entries: int = 400) -> dict:
    """扫描项目目录,返回结构化快照 dict。

    Args:
        root: 项目根目录绝对路径。
        max_depth: 目录树最大深度。
        max_entries: 树中最多列出的条目数(防大仓库爆炸)。

    Returns:
        ``{"root", "project_name", "tree", "manifests", "readme",
          "lang_stats", "entry_files"}``。root 不存在则各字段为空。
    """
    root = os.path.normpath(root)
    project_name = os.path.basename(root)
    empty = {
        "root": root, "project_name": project_name, "tree": "",
        "manifests": {}, "readme": "", "lang_stats": {}, "entry_files": [],
    }
    if not os.path.isdir(root):
        return empty

    tree, lang_stats, entry_files = _build_tree(root, max_depth, max_entries)

    # 顶层清单文件内容。
    manifests: dict[str, str] = {}
    for name in _MANIFEST_FILES:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            text = _read_text(p, _MAX_FILE_CHARS)
            if text.strip():
                manifests[name] = text

    # README(顶层,不区分大小写)。
    readme = ""
    try:
        for e in os.listdir(root):
            if e.lower() in _README_NAMES and os.path.isfile(os.path.join(root, e)):
                readme = _read_text(os.path.join(root, e), _README_CHARS)
                break
    except OSError:
        pass

    return {
        "root": root,
        "project_name": project_name,
        "tree": tree,
        "manifests": manifests,
        "readme": readme,
        "lang_stats": lang_stats,
        "entry_files": entry_files,
    }
