"""Mem0 长期记忆聊天机器人 —— Gradio 可视化界面。

左侧:与带记忆的助手聊天(复用 memory_framework.chat.reply)。
右侧:实时展示当前 user_id 的长期记忆(每轮对话后刷新)。
顶部:切换 user_id;可一键加载 data/testsets 里的对话集预热记忆。

启动:
    PYTHONPATH=. python app.py
然后浏览器打开终端提示的本地地址(默认 http://127.0.0.1:7860)。
"""

import glob
import os
import threading
from datetime import datetime

from dotenv import load_dotenv

# 在读取任何配置前加载 .env（代码只读环境变量，不会自行读取 .env 文件）。
load_dotenv()

import gradio as gr

from memory_framework.chat import persist_turn, reply
from memory_framework.code_analyzer import (
    collect_source_files,
    deep_analyze,
    list_code_analyzed,
    load_code_analysis,
    save_code_analysis,
)
from memory_framework.code_snapshot import (
    compute_hashes,
    diff_snapshot,
    save_snapshot,
)
from memory_framework.config import DEFAULT_LLM_MODEL, build_config_and_apply_env
from memory_framework.conversation import ingest, load_conversation
from memory_framework.memory_store import MemoryStore
from memory_framework.persona_store import load_persona, save_persona
from memory_framework.profile_store import ProfileStore, ProjectStore
from memory_framework.project_chat import (
    persist_project_turn,
    project_reply,
    sediment_analysis,
    sediment_change,
)
from memory_framework.project_dims import PROJECT_DIMENSION_LABELS
from memory_framework.cc_ingest import list_cc_projects
from memory_framework.project_tracker import available_projects, import_and_track
from memory_framework.repo_analyzer import (
    analyze_project_structure,
    list_analyzed_projects,
    load_analysis,
    save_analysis,
)
from memory_framework.repo_scan import scan_repo
from skill_export.exporter import export_skill

build_config_and_apply_env()
MODEL = os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
_STORE = None
_PROFILE = None
_PROJECT_STORE = None
# 串行化对 Qdrant client / profile 文件的访问：on_send 的落盘、加载测试集、
# 刷新按钮可能并发访问同一个 store，Qdrant 本地 client 非线程安全。
_WRITE_LOCK = threading.Lock()


def get_store() -> MemoryStore:
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore()
    return _STORE


def get_profile_store() -> ProfileStore:
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = ProfileStore()
    return _PROFILE


def get_project_store() -> ProjectStore:
    global _PROJECT_STORE
    if _PROJECT_STORE is None:
        _PROJECT_STORE = ProjectStore()
    return _PROJECT_STORE


# ---------------------------------------------------------------------------
# 视觉设计:清爽浅色 · 纸感卡片
#
# 设计 token（避免 AI 默认三件套:奶油衬线 / 荧光暗底 / 报纸栏）:
#   纸底 #FBFAF7(暖白、比奶油更冷净) · 正文 #1F2328 · 次要 #6B7280
#   四维语义色(降饱和、克制): 进度松绿 / 问题陶红 / 待办芥黄 / 决策石青
#   字体: 标题 Space Grotesk · 正文 PingFang SC + Inter · 数据 JetBrains Mono
#   签名: 每张维度卡左缘语义色条 + 等宽徽标计数(PROGRESS · 3),一眼可扫。
# ---------------------------------------------------------------------------

# 四维度语义色:(主色, 极浅底色)
_DIM_COLORS = {
    "progress": ("#3B7A57", "#EEF5F0"),
    "blocker":  ("#C25A4A", "#FaefEC"),
    "todo":     ("#B8862F", "#F8F2E4"),
    "decision": ("#4A6FA5", "#EDF1F7"),
}
# 维度英文徽标(等宽,呼应「来自终端对话」)。
_DIM_BADGE = {
    "progress": "PROGRESS", "blocker": "BLOCKER",
    "todo": "TODO", "decision": "DECISION",
}

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --paper: #FBFAF7;
    --ink: #1F2328;
    --muted: #6B7280;
    --line: #E7E3DA;
    --card: #FFFFFF;
    --font-display: 'Space Grotesk', 'PingFang SC', system-ui, sans-serif;
    --font-body: 'Inter', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, 'SFMono-Regular', monospace;
}

.gradio-container, .gradio-container * { font-family: var(--font-body); }
.gradio-container { background: var(--paper) !important; color: var(--ink) !important; }

/* 锁定浅色主题:即便浏览器/系统处于深色模式,也强制维持深字浅底,
   避免 Gradio Soft 主题注入的浅色文字覆盖导致「浅字浅底、看不清」。 */
.dark, body.dark, .gradio-container.dark {
    --paper: #FBFAF7;
    --ink: #1F2328;
    --muted: #6B7280;
    --line: #E7E3DA;
    --card: #FFFFFF;
}
.dark .gradio-container,
.gradio-container.dark { background: var(--paper) !important; color: var(--ink) !important; }
/* 兜底:文字类元素统一用深墨色,不被主题变量覆盖。 */
.gradio-container .prose,
.gradio-container p,
.gradio-container li,
.gradio-container span,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label { color: var(--ink); }
.gradio-container .dim-empty,
.gradio-container .dim-meta,
.gradio-container .dim-count,
.gradio-container .dim-src,
.gradio-container .status-note,
.gradio-container .app-sub,
.gradio-container .g-label,
.gradio-container .panel-title .n { color: var(--muted); }

/* 顶部标题条 */
.app-header {
    font-family: var(--font-display);
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--ink);
    padding: 4px 2px 0;
}
.app-header .app-sub {
    font-family: var(--font-mono);
    font-weight: 400;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 2px;
}

/* 卡片网格 */
.dim-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}
@media (max-width: 720px) { .dim-grid { grid-template-columns: 1fr; } }

.dim-card {
    position: relative;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px 14px 18px;
    overflow: hidden;
}
.dim-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--dim-accent);
}
.dim-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 10px;
}
.dim-badge {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: var(--dim-accent);
}
.dim-label {
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.98rem;
    color: var(--ink);
}
.dim-count {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
}
.dim-item { margin: 0 0 11px; line-height: 1.5; }
.dim-item:last-child { margin-bottom: 0; }
.dim-text { font-size: 0.9rem; color: var(--ink); }
.dim-meta {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--muted);
    background: var(--dim-soft);
    padding: 1px 6px;
    border-radius: 5px;
    margin-left: 6px;
    white-space: nowrap;
}
.dim-src {
    display: block;
    font-size: 0.74rem;
    color: var(--muted);
    margin-top: 3px;
    padding-left: 10px;
    border-left: 2px solid var(--line);
}
.dim-empty { font-size: 0.82rem; color: var(--muted); font-style: italic; }

/* 记忆 / 画像 列表卡片 */
.panel-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px;
}
.panel-title {
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.98rem;
    color: var(--ink);
    margin: 0 0 4px;
}
.panel-title .n {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 400;
}
.mem-list { list-style: none; padding: 0; margin: 8px 0 0; }
.mem-list li {
    font-size: 0.86rem;
    line-height: 1.5;
    padding: 7px 0;
    border-top: 1px solid var(--line);
    color: var(--ink);
}
.mem-list li:first-child { border-top: none; }
.prof-group { margin-top: 12px; }
.prof-group .g-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.status-note {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 12px;
}

/* 按钮微调 */
button.primary, .gradio-container button[variant="primary"] {
    background: var(--ink) !important;
    border-color: var(--ink) !important;
}
"""


def _esc(text) -> str:
    """HTML 转义,防止记忆/画像文本里的 < & 破坏卡片结构。"""
    s = "" if text is None else str(text)
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _render_memories(user_id: str) -> str:
    if not user_id.strip():
        return "<div class='panel-card'><p class='dim-empty'>请先输入 user_id</p></div>"
    mems = get_store().get_all(user_id=user_id)
    uid = _esc(user_id)
    if not mems:
        return (f"<div class='panel-card'><p class='panel-title'>🧠 {uid} 的长期记忆</p>"
                f"<p class='dim-empty'>暂无记忆</p></div>")
    items = []
    for m in mems:
        text = m.get("memory", m) if isinstance(m, dict) else m
        items.append(f"<li>{_esc(text)}</li>")
    return (f"<div class='panel-card'>"
            f"<p class='panel-title'>{uid} · 长期记忆 <span class='n'>{len(mems)} 条</span></p>"
            f"<ul class='mem-list'>{''.join(items)}</ul></div>")


def _memory_choices(user_id: str) -> list:
    """把某用户记忆列成 [(显示文本, memory_id)] 供下拉选择删/改。无 id 的条目跳过。"""
    if not user_id or not user_id.strip():
        return []
    mems = get_store().get_all(user_id=user_id.strip())
    choices = []
    for m in mems:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        text = m.get("memory", "")
        if mid and text:
            label = text if len(text) <= 40 else text[:40] + "…"
            choices.append((label, mid))
    return choices


def _memory_text_by_id(user_id: str, mem_id: str) -> str:
    """按 memory_id 取该条记忆全文;找不到返回空串。"""
    if not mem_id:
        return ""
    for m in get_store().get_all(user_id=(user_id or "").strip()):
        if isinstance(m, dict) and m.get("id") == mem_id:
            return m.get("memory", "")
    return ""


def _render_project(project_id: str, note: str = "") -> str:
    if not project_id or not project_id.strip():
        return "<div class='panel-card'><p class='dim-empty'>请先选择一个项目</p></div>"
    pid = project_id.strip()
    prof = get_project_store().get_project(pid)
    grouped = {}
    for it in prof.items:
        grouped.setdefault(it.dimension, []).append(it)

    cards = []
    for dim, label in PROJECT_DIMENSION_LABELS.items():
        accent, soft = _DIM_COLORS[dim]
        items = grouped.get(dim, [])
        if not items:
            body = "<p class='dim-empty'>暂无</p>"
        else:
            rows = []
            for it in items:
                src = (f"<span class='dim-src'>来源: {_esc(it.evidence)}</span>"
                       if it.evidence else "")
                rows.append(f"<div class='dim-item'><span class='dim-text'>{_esc(it.text)}</span>"
                            f"<span class='dim-meta'>{it.importance:.0f}</span>{src}</div>")
            body = "".join(rows)
        cards.append(
            f"<div class='dim-card' style='--dim-accent:{accent};--dim-soft:{soft}'>"
            f"<div class='dim-head'>"
            f"<span class='dim-badge'>{_DIM_BADGE[dim]}</span>"
            f"<span class='dim-label'>{_esc(label)}</span>"
            f"<span class='dim-count'>{len(items)}</span>"
            f"</div>{body}</div>"
        )
    note_html = f"<p class='status-note'>{_esc(note)}</p>" if note else ""
    return (f"<div class='panel-card' style='margin-bottom:14px;border:none;padding:0'>"
            f"<p class='panel-title'>{_esc(pid)} · 进度追踪</p></div>"
            f"<div class='dim-grid'>{''.join(cards)}</div>{note_html}")


def _render_analysis(name: str) -> str:
    """读取某项目已保存的结构分析 md;无则给提示(返回 markdown 文本)。"""
    if not name:
        return "_(输入项目目录路径后点「分析 / 更新」,或从下拉选择已分析的项目)_"
    md = load_analysis(name)
    if md is None:
        return f"_(项目 **{name}** 暂无分析,点「分析 / 更新」生成)_"
    return md


def on_analyze_repo(path: str):
    """扫描项目目录 → LLM 归纳 → 存 md(生成器,两段式)。

    第一次 yield 提示分析中;随后 scan_repo → analyze → save_analysis,第二次
    yield 出 md 报告并刷新「已分析项目」下拉。路径非法则给出错误提示。
    输出顺序:(md_view, dropdown_update)。
    """
    p = (path or "").strip()
    if not p:
        yield "_(请先输入项目目录的绝对路径,如 `/Users/mashiro/Mem0`)_", gr.update()
        return
    if not os.path.isdir(p):
        yield f"_❌ 路径不是有效目录:`{_esc(p)}`_", gr.update()
        return

    name = os.path.basename(os.path.normpath(p))
    yield f"_⟳ 正在扫描并分析 **{_esc(name)}**({_esc(p)})…_", gr.update()

    with _WRITE_LOCK:
        snapshot = scan_repo(p)
        md = analyze_project_structure(snapshot, project_name=name, model=MODEL)
        if not md:
            yield ("_❌ 分析失败(LLM 未返回内容,可能是限流或网络问题),请稍后重试。_",
                   gr.update())
            return
        saved_path = save_analysis(name, md, source_path=p, now=datetime.now())
        choices = _all_projects()
    note = f"\n\n---\n_✅ 已保存到 `{_esc(saved_path)}`_"
    yield md + note, gr.update(choices=choices, value=name)


def on_send(message: str, history: list, user_id: str,
            persona: str = "", inject: bool = True):
    """一轮对话:先出回复(快),再后台落盘记忆,完成后刷新面板。

    生成器:第一次 yield 立即把回复推给前端(用户只等生成回复这一步);
    yield 之后再落盘(mem0 抽取记忆,较慢),最后第二次 yield 刷新记忆概览 +
    记忆管理下拉。persona/inject 决定是否注入个性化上下文。
    输出:(chatbot, mem_view, msg, mem_select)。
    """
    if not message.strip() or not user_id.strip():
        with _WRITE_LOCK:
            yield history, _render_memories(user_id), "", gr.update()
        return

    uid = user_id.strip()
    with _WRITE_LOCK:
        answer = reply(get_store(), uid, message.strip(), MODEL,
                       persona=persona, inject=inject)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    # 第一次 yield:回复立即显示;记忆面板暂标记后台更新中。
    updating = "<p class='status-note'>⟳ 记忆后台更新中…</p>"
    with _WRITE_LOCK:
        mem_now = _render_memories(uid)
    yield history, mem_now + updating, "", gr.update()

    # 落盘:mem0 抽取记忆(慢,已移出回复关键路径)。
    with _WRITE_LOCK:
        persist_turn(get_store(), get_profile_store(), uid,
                     message.strip(), answer, MODEL)
        mem_final = _render_memories(uid)
        choices = _memory_choices(uid)
    # 第二次 yield:刷新出新记忆,去掉更新中标记,刷新记忆下拉。
    yield history, mem_final, "", gr.update(choices=choices, value=None)


def on_load_testset(name: str, user_id: str):
    """把选中的测试集灌入记忆库(用其自带 user_id),并切换到该 user。

    输出:(mem_view, user_id, chatbot)。
    """
    if not name:
        return (_render_memories(user_id), user_id,
                [{"role": "assistant", "content": "请先在下拉框选择一个测试集。"}])
    conv = load_conversation(f"data/testsets/{name}.json")
    uid = conv["user_id"]
    with _WRITE_LOCK:
        store = get_store()
        store.delete_all(user_id=uid)
        ingest(store, conv)
        mem_md = _render_memories(uid)
    greeting = [{"role": "assistant",
                 "content": f"已灌入测试集 **{name}**({len(conv['messages'])} 轮)"
                            f"到用户 `{uid}`。现在可以问我关于 TA 的事,看我能否记住。"}]
    return mem_md, uid, greeting


def _testset_names() -> list:
    return sorted(
        os.path.basename(p)[:-5]
        for p in glob.glob("data/testsets/*.json")
        if not p.endswith(".facts.json")
    )


# ---- 个性定制:人设 + 逐条记忆增删改 ----

def on_save_persona(user_id: str, persona: str, inject: bool) -> str:
    """保存用户人设 + 注入开关,返回状态提示(markdown)。"""
    uid = (user_id or "").strip()
    if not uid:
        return "_(请先填写用户 ID)_"
    with _WRITE_LOCK:
        save_persona(uid, persona or "", bool(inject))
    state = "开启" if inject else "关闭"
    return (f"_✅ 已保存 **{_esc(uid)}** 的人设,注入{state}。_ "
            f"Claude Code 可经 MCP `get_user_persona` 拉取这份系统级上下文。")


def on_switch_user(user_id: str):
    """切换用户 ID:同步刷新记忆概览 / 记忆下拉 / 人设框 / 注入开关 / 编辑框。

    输出:(mem_view, mem_select, persona_box, inject_toggle, mem_edit)。
    """
    uid = (user_id or "").strip()
    with _WRITE_LOCK:
        mem_html = _render_memories(uid)
        mem_ch = _memory_choices(uid)
    cfg = load_persona(uid)
    return (mem_html,
            gr.update(choices=mem_ch, value=None),
            cfg["persona"], cfg["inject"], "")


def on_pick_memory(user_id: str, mem_id: str) -> str:
    """选中一条记忆 → 回填编辑框全文。"""
    with _WRITE_LOCK:
        return _memory_text_by_id(user_id, mem_id)


def on_edit_memory(user_id: str, mem_id: str, text: str):
    """保存对某条记忆的编辑。输出:(mem_view, mem_select, note)。"""
    uid = (user_id or "").strip()
    if not mem_id:
        return _render_memories(uid), gr.update(), "_(请先在下拉里选一条记忆)_"
    if not (text or "").strip():
        return _render_memories(uid), gr.update(), "_(内容不能为空;要删除请用「删除」)_"
    with _WRITE_LOCK:
        try:
            get_store().update(mem_id, text.strip())
        except Exception as exc:
            return _render_memories(uid), gr.update(), f"_❌ 更新失败:{_esc(str(exc))}_"
        mem_html = _render_memories(uid)
        choices = _memory_choices(uid)
    return mem_html, gr.update(choices=choices, value=None), "_✅ 已更新该条记忆。_"


def on_delete_memory(user_id: str, mem_id: str):
    """删除某条记忆。输出:(mem_view, mem_select, mem_edit, note)。"""
    uid = (user_id or "").strip()
    if not mem_id:
        return _render_memories(uid), gr.update(), "", "_(请先在下拉里选一条记忆)_"
    with _WRITE_LOCK:
        try:
            get_store().delete(mem_id)
        except Exception as exc:
            return (_render_memories(uid), gr.update(), "",
                    f"_❌ 删除失败:{_esc(str(exc))}_")
        mem_html = _render_memories(uid)
        choices = _memory_choices(uid)
    return (mem_html, gr.update(choices=choices, value=None), "",
            "_✅ 已删除该条记忆。_")


def on_import_project(project_id: str):
    """导入并分析某项目的 CC 对话日志(生成器,两段式)。

    第一次 yield 提示分析中;随后跑 import_and_track(解析日志→灌记忆→LLM 提炼
    四维度,较慢),第二次 yield 刷新四维面板。
    """
    if not project_id:
        yield "<div class='panel-card'><p class='dim-empty'>请先在下拉框选择一个项目</p></div>"
        return
    pid = project_id.strip()
    with _WRITE_LOCK:
        analyzing = _render_project(pid, note="⟳ 正在解析对话日志并提炼四维度,请稍候…")
    yield analyzing

    with _WRITE_LOCK:
        result = import_and_track(get_store(), get_project_store(), pid, MODEL)
        n = result["new_messages"]
        status = result.get("status")
        if status == "llm_empty":
            note = (f"⚠️ 已导入 {n} 条消息,但四维度提炼未返回结果(通常是 LLM 限流/"
                    f"网络问题,已自动重试仍失败),暂保留旧记录。请稍后再点「导入并分析」重试。"
                    if n else
                    "⚠️ 四维度提炼未返回结果(通常是 LLM 限流/网络问题,已自动重试仍失败),"
                    "暂保留旧记录。请稍后重试。")
        elif status == "no_memories":
            note = "ℹ️ 该项目暂无可提炼的记忆(对话日志可能为空或已被过滤)。"
        elif n:
            note = f"✅ 本次新导入 {n} 条消息并已刷新四维度。"
        else:
            note = "✅ 无新增消息(增量游标生效),已按现有记忆刷新四维度。"
        panel = _render_project(pid, note=note)
    yield panel


# ---------------------------------------------------------------------------
# 项目工作台:代码深度分析 + 项目 Q&A + 导出 Skill
# ---------------------------------------------------------------------------

def _all_projects() -> list:
    """全局项目并集:CC 日志项目 ∪ 结构已分析 ∪ 代码已分析,按名排序。"""
    names = (set(available_projects())
             | set(list_analyzed_projects())
             | set(list_code_analyzed()))
    return sorted(names)


def _project_path_map() -> dict:
    """{项目 id: 绝对路径}——用于选中全局项目时回填路径框。

    来源:CC 项目还原路径 + 代码深度分析 json 记录的 source_path。二者都存在时
    以 CC 还原路径为准(更权威)。取不到路径的项目不入表(路径框留空)。
    """
    mapping = {}
    for p in list_code_analyzed():
        data = load_code_analysis(p)
        sp = (data or {}).get("source_path") or ""
        if sp:
            mapping[p] = sp
    for proj in list_cc_projects():
        path = proj.get("path") or ""
        if path:
            mapping[proj["project_id"]] = path
    return mapping


def on_switch_project(pid: str):
    """全局项目切换:回填路径 + 同步刷新进度四维 / 结构分析 / 代码分析三视图。

    纯读(渲染盘上产物 + 进度 store 文件读),不触发 LLM/落盘,响应快。
    输出顺序:(g_path, proj_view, analysis_view, wb_code_view)。
    """
    name = (pid or "").strip()
    if not name:
        empty = "<div class='panel-card'><p class='dim-empty'>请先在上方选择一个项目</p></div>"
        return "", empty, "_(请选择项目)_", "_(请选择项目)_"
    path = _project_path_map().get(name, "")
    with _WRITE_LOCK:
        proj_html = _render_project(name)
    analysis_md = _render_analysis(name)
    code_md = _render_code_analysis(name)
    return path, proj_html, analysis_md, code_md


def _render_code_analysis(name: str) -> str:
    """读取某项目已保存的代码深度分析 md;无则提示(返回 markdown 文本)。"""
    if not name:
        return "_(输入项目目录路径后点「深度分析」,或从下拉选择已分析的项目)_"
    data = load_code_analysis(name)
    if not data or not data.get("md"):
        return f"_(项目 **{name}** 暂无代码深度分析,点「深度分析」生成)_"
    return data["md"]


def on_deep_analyze(path: str, incremental: bool, selected: str = ""):
    """代码深度分析(生成器,两段式)。incremental=True 时只重分析变更文件。

    第一次 yield 提示分析中;随后 deep_analyze → save → 沉淀记忆 → 存快照,
    第二次 yield 出报告并刷新已分析下拉。输出:(md_view, dropdown_update)。
    路径为空但下拉已选中某项目时,不清空报告——直接展示该项目已存报告。
    """
    p = (path or "").strip()
    if not p:
        sel = (selected or "").strip()
        if sel and load_code_analysis(sel):
            yield _render_code_analysis(sel), gr.update()
        else:
            yield ("_(请先在上方输入项目目录的绝对路径,如 `/Users/mashiro/Mem0`,"
                   "再点「深度分析」。)_"), gr.update()
        return
    if not os.path.isdir(p):
        yield f"_❌ 路径不是有效目录:`{_esc(p)}`_", gr.update()
        return

    name = os.path.basename(os.path.normpath(p))
    mode = "增量" if incremental else "全量"
    yield (f"_⟳ 正在对 **{_esc(name)}** 做{mode}代码深度分析(逐文件摘要 + 依赖图),"
           f"文件较多时耗时较长,请稍候…_"), gr.update()

    with _WRITE_LOCK:
        try:
            files = collect_source_files(p)
            if not files:
                yield f"_❌ 目录 `{_esc(p)}` 下未发现可分析的源码文件。_", gr.update()
                return
            current = compute_hashes(files)
            diff = diff_snapshot(name, current)
            changed_only = None
            if incremental:
                changed = diff["added"] + diff["changed"]
                if not changed and load_code_analysis(name):
                    yield (f"_✅ **{_esc(name)}** 自上次分析以来无文件变更,复用既有报告。_\n\n"
                           + _render_code_analysis(name)), gr.update(
                        choices=_all_projects(), value=name)
                    return
                changed_only = changed or None

            result = deep_analyze(p, project_name=name, model=MODEL,
                                  changed_only=changed_only)
            if not result.get("md"):
                yield ("_❌ 分析失败(未取得任何摘要,可能限流或网络问题),请稍后重试。_",
                       gr.update())
                return
            save_code_analysis(name, result["md"], result["structured"],
                               source_path=p, now=datetime.now())
            save_snapshot(name, current)
            # 沉淀到项目长期记忆(供 Q&A 召回);store 落盘用同一把锁串行。
            try:
                store = get_store()
                sediment_analysis(store, name, result["capinfo"],
                                  result["structured"]["summaries"])
                sediment_change(store, name, diff)
            except Exception as exc:
                print(f"[on_deep_analyze] 记忆沉淀失败(不影响报告):{exc}")
            choices = _all_projects()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield (f"_❌ 分析出错:`{_esc(str(exc))}`。详见终端日志。_", gr.update())
            return

    banner = ""
    if result["capinfo"].get("degraded"):
        banner = "> ⚠️ 归纳步骤未生成,以下为降级报告(逐文件摘要已保存,可稍后重试)。\n\n"
    yield banner + result["md"], gr.update(choices=choices, value=name)


def _load_on_open():
    """界面加载时初始化全局项目选择器 + 路径 + 三 Tab 视图。

    默认优先选中一个已做代码分析的项目(以便加载即展示报告);否则退回全集第一个。
    输出顺序:(g_project, g_path, proj_view, analysis_view, wb_code_view)。
    """
    all_projects = _all_projects()
    if not all_projects:
        empty = "<div class='panel-card'><p class='dim-empty'>暂无项目</p></div>"
        return (gr.update(choices=[], value=None), "",
                empty,
                "_(暂无项目。在上方路径框填目录后点「深度分析」或「分析 / 更新」,"
                "或去「项目进度追踪」页导入对话日志。)_",
                "_(暂无代码深度分析)_")
    analyzed = list_code_analyzed()
    default = analyzed[0] if analyzed else all_projects[0]
    path, proj_html, analysis_md, code_md = on_switch_project(default)
    return (gr.update(choices=all_projects, value=default), path,
            proj_html, analysis_md, code_md)


def on_project_send(message: str, history: list, project_id: str):
    """项目 Q&A 一轮(生成器,两段式):先出回复,再后台落盘项目记忆。"""
    pid = (project_id or "").strip()
    if not message.strip() or not pid:
        yield history, ""
        return
    with _WRITE_LOCK:
        answer = project_reply(get_store(), get_project_store(), pid,
                               message.strip(), MODEL)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    yield history, ""
    with _WRITE_LOCK:
        persist_project_turn(get_store(), pid, message.strip(), answer)


def on_export_skill(project_id: str) -> str:
    """把项目长期记忆导出为 Claude Code Skill,返回状态提示(markdown)。"""
    pid = (project_id or "").strip()
    if not pid:
        return "_(请先选择一个项目)_"
    # 提前探明有哪些产物,导出后如实告知,避免「成功了却是空壳」的误导。
    has_code = load_code_analysis(pid) is not None
    has_progress = bool(get_project_store().get_project(pid).items)
    if not has_code and not has_progress:
        return (f"_⚠️ 项目 **{_esc(pid)}** 目前没有任何可导出的记忆(既无代码深度分析,"
                f"也无进度四维)。请先在上方点「深度分析」,或在「项目进度追踪」页导入对话日志,"
                f"再导出 Skill。_")
    with _WRITE_LOCK:
        path = export_skill(pid)
    missing = []
    if not has_code:
        missing.append("代码深度分析(点上方「深度分析」补上)")
    if not has_progress:
        missing.append("进度四维(去「项目进度追踪」页导入)")
    warn = ("\n\n_⚠️ 其中缺少:" + "、".join(missing) + ",对应 reference 为占位内容。_"
            if missing else "")
    return (f"_✅ 已导出 Skill 到 `{_esc(path)}`。_{warn}\n\n"
            f"在 Claude Code 中,进入该项目目录即可加载 **{_esc(pid)}** 的长期记忆;"
            f"或用 MCP 注册:`claude mcp add mem0-project -- conda run -n mem0 "
            f"python -m mcp_server.server`。")



_FORCE_LIGHT_JS = """
() => {
    const el = document.querySelector('.gradio-container');
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    if (el) el.classList.remove('dark');
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Mem0 长期记忆 · 项目进度追踪",
                   theme=gr.themes.Soft(), css=APP_CSS, js=_FORCE_LIGHT_JS) as demo:
        gr.HTML("<div class='app-header'>"
                "<h2 style='margin:0'>Mem0 长期记忆 · 项目进度追踪</h2>"
                "<div class='app-sub'>Memory & project progress from your Claude Code conversations</div>"
                "</div>")
        # 全局项目选择器:选一次,下面「进度 / 结构 / 工作台」三个 Tab 都基于它工作。
        with gr.Row():
            g_project = gr.Dropdown(label="当前项目(全局 · 驱动进度/结构/工作台)",
                                    choices=_all_projects(), scale=3)
            g_path = gr.Textbox(label="项目目录绝对路径(结构/深度分析用)",
                                placeholder="/Users/mashiro/Mem0", scale=4)
            g_refresh = gr.Button("刷新项目列表", scale=1)
        with gr.Tabs():
            with gr.Tab("💬 记忆聊天 · 个性定制"):
                gr.Markdown(
                    "把这里当作**个性化定制中心**:写一段**全局人设/系统提示词**,"
                    "手动**删除 / 编辑**助手记住的记忆,并用开关决定是否把「人设 + 记忆」"
                    "作为**系统级上下文**注入对话。这套内容也能经 MCP `get_user_persona` "
                    "提供给 Claude Code。记忆按 user_id 隔离。")
                with gr.Row():
                    user_id = gr.Textbox(label="用户 ID(记忆隔离键)", value="demo_user", scale=2)
                    testset = gr.Dropdown(label="加载测试集预热记忆", choices=_testset_names(),
                                          scale=2)
                    load_btn = gr.Button("灌入所选测试集", scale=1)
                with gr.Row():
                    persona_box = gr.Textbox(
                        label="全局人设 / 系统提示词(每次对话注入)", lines=3, scale=4,
                        placeholder="例如:你是我的中文编程助手,回答尽量简洁、给可运行代码。")
                    with gr.Column(scale=1):
                        inject_toggle = gr.Checkbox(label="注入人设与记忆", value=True)
                        persona_save_btn = gr.Button("保存人设", variant="primary")
                persona_note = gr.Markdown("")
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="对话", height=440)
                        msg = gr.Textbox(label="你的消息", placeholder="说点什么…(回车发送)")
                    with gr.Column(scale=2):
                        mem_view = gr.HTML(_render_memories("demo_user"))
                        gr.Markdown("#### 管理记忆(删除 / 编辑)")
                        mem_select = gr.Dropdown(label="选择一条记忆",
                                                 choices=_memory_choices("demo_user"))
                        mem_edit = gr.Textbox(label="记忆内容(可编辑)", lines=2)
                        with gr.Row():
                            mem_save_btn = gr.Button("保存修改", scale=1)
                            mem_del_btn = gr.Button("删除该条", scale=1)
                        mem_note = gr.Markdown("")
                        refresh = gr.Button("刷新记忆")

                # 对话:注入 persona/inject;刷新记忆下拉。
                msg.submit(on_send,
                           [msg, chatbot, user_id, persona_box, inject_toggle],
                           [chatbot, mem_view, msg, mem_select])
                load_btn.click(on_load_testset, [testset, user_id],
                               [mem_view, user_id, chatbot])
                # 切用户 / 刷新 → 刷新记忆概览/下拉/人设/开关/编辑框。
                _user_outputs = [mem_view, mem_select,
                                 persona_box, inject_toggle, mem_edit]
                refresh.click(on_switch_user, user_id, _user_outputs)
                user_id.change(on_switch_user, user_id, _user_outputs)
                # 人设保存
                persona_save_btn.click(on_save_persona,
                                       [user_id, persona_box, inject_toggle],
                                       persona_note)
                # 记忆管理
                mem_select.change(on_pick_memory, [user_id, mem_select], mem_edit)
                mem_save_btn.click(on_edit_memory, [user_id, mem_select, mem_edit],
                                   [mem_view, mem_select, mem_note])
                mem_del_btn.click(on_delete_memory, [user_id, mem_select],
                                  [mem_view, mem_select, mem_edit, mem_note])

            with gr.Tab("📊 项目进度追踪"):
                gr.Markdown("从你与 Claude Code 的对话日志(`~/.claude/projects`)中,"
                            "沉淀**当前项目**的**进度 / 问题 / 待办 / 决策**。"
                            "项目在顶部全局选择;支持增量重跑,再次导入只处理新增对话。")
                with gr.Row():
                    import_btn = gr.Button("导入并分析当前项目", variant="primary", scale=1)
                    proj_refresh = gr.Button("刷新展示", scale=1)
                proj_view = gr.HTML(
                    "<div class='panel-card'><p class='dim-empty'>"
                    "在顶部选择项目后点「导入并分析当前项目」</p></div>")

                def _refresh_project(pid):
                    with _WRITE_LOCK:
                        return _render_project(pid)

                import_btn.click(on_import_project, g_project, proj_view)
                proj_refresh.click(_refresh_project, g_project, proj_view)

            with gr.Tab("🗂 项目结构分析"):
                gr.Markdown("扫描**当前项目目录**结构并让 LLM 归纳出"
                            "**项目概述 / 技术栈 / 模块划分 / 目录结构说明**。"
                            "目录路径取顶部全局路径框;结果保存为本地 md,切项目时自动读取。")
                with gr.Row():
                    analyze_btn = gr.Button("分析 / 更新", variant="primary", scale=1)
                analysis_view = gr.Markdown("_(加载中…)_")

                analyze_btn.click(on_analyze_repo, g_path, [analysis_view, g_project])

            with gr.Tab("🛠 项目工作台"):
                gr.Markdown(
                    "对**当前项目**做**代码深度分析**(逐文件职责摘要 + 模块依赖图 + 关键路径),"
                    "并就它**问答**(自动注入进度四维 + 代码结构 + 历史记忆)。"
                    "项目与路径都在顶部全局选择——选哪个项目,问答就基于那个项目的代码与记忆作答。"
                    "可**一键导出为 Claude Code Skill**,或通过 MCP 让 Claude Code 直接调取。")
                with gr.Row():
                    wb_incremental = gr.Checkbox(label="仅重分析变更文件(增量)",
                                                 value=False, scale=1)
                    wb_analyze_btn = gr.Button("深度分析", variant="primary", scale=1)
                    wb_export_btn = gr.Button("导出 Skill", scale=1)
                wb_export_note = gr.Markdown("")
                wb_code_view = gr.Markdown("_(加载中…)_")

                gr.Markdown("### 项目问答 · 长期记忆")
                wb_chat = gr.Chatbot(label="项目问答", height=360)
                wb_msg = gr.Textbox(label="问这个项目的事",
                                    placeholder="如:这个项目的入口在哪?进度到哪了?(回车发送)")

                wb_analyze_btn.click(on_deep_analyze,
                                     [g_path, wb_incremental, g_project],
                                     [wb_code_view, g_project])
                wb_msg.submit(on_project_send, [wb_msg, wb_chat, g_project],
                              [wb_chat, wb_msg])
                wb_export_btn.click(on_export_skill, g_project, wb_export_note)

        # 全局项目切换 → 同步刷新进度/结构/工作台三视图 + 回填路径。
        _switch_outputs = [g_path, proj_view, analysis_view, wb_code_view]
        g_project.change(on_switch_project, g_project, _switch_outputs)
        g_refresh.click(lambda: gr.update(choices=_all_projects()), None, g_project)

        # 打开界面:初始化全局选择器 + 路径 + 三视图。
        demo.load(_load_on_open, None,
                  [g_project, g_path, proj_view, analysis_view, wb_code_view])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
