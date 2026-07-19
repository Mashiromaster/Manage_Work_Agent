"""代码变更感知:对项目源文件做哈希快照,下次分析时 diff 出哪些文件变了。

支撑增量深度分析——只重新摘要 added + changed 的文件,removed 的从旧结果剔除,
未变的复用旧摘要,从而大幅压低 LLM 调用。快照存 ``code_analysis/.snapshots/``。

哈希用 sha256(mtime + size + 前 8k 内容):既能感知内容变更,又不必读全大文件。
纯逻辑 + 磁盘 I/O,可对临时目录直接单测。
"""

import hashlib
import json
import os

SNAPSHOT_DIR = "./code_analysis/.snapshots"
_HASH_READ_BYTES = 8192


def compute_hashes(files: list) -> dict:
    """对文件清单计算 {rel: hash}。files 为 collect_source_files 的返回。"""
    hashes = {}
    for f in files:
        try:
            st = os.stat(f["abs"])
            with open(f["abs"], "rb") as fh:
                head = fh.read(_HASH_READ_BYTES)
        except OSError:
            continue
        h = hashlib.sha256()
        h.update(str(int(st.st_mtime)).encode())
        h.update(str(st.st_size).encode())
        h.update(head)
        hashes[f["rel"]] = h.hexdigest()
    return hashes


def _safe_name(project_name: str) -> str:
    return project_name.replace("/", "_").replace("\\", "_").strip() or "unnamed"


def _snap_path(project_name: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{_safe_name(project_name)}.json")


def load_snapshot(project_name: str) -> dict:
    """读上次快照 {rel: hash};无则返回 {}。"""
    p = _snap_path(project_name)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_snapshot(project_name: str, hashes: dict) -> str:
    """写快照,返回路径。"""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    p = _snap_path(project_name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)
    return p


def diff_snapshot(project_name: str, current: dict) -> dict:
    """对比当前哈希与上次快照,返回 {added, changed, removed}(相对路径列表)。

    空快照(首次)→ 全部计入 added。
    """
    prev = load_snapshot(project_name)
    prev_keys = set(prev)
    cur_keys = set(current)
    added = sorted(cur_keys - prev_keys)
    removed = sorted(prev_keys - cur_keys)
    changed = sorted(k for k in (cur_keys & prev_keys) if prev[k] != current[k])
    return {"added": added, "changed": changed, "removed": removed}
