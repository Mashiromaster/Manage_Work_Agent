"""Mem0 项目记忆 MCP server 包。

把项目的长期记忆(开发进度四维、结构分析、代码深度分析)以只读方式暴露给
Claude Code。**关键约束**:所有工具只读磁盘上的 JSON/MD 产物,绝不构造
MemoryStore / 打开 Qdrant——因为 Qdrant 本地文件模式单进程独占,MCP 与 Gradio
不能同时持有。Gradio 始终是 Qdrant 的唯一持有者。
"""
