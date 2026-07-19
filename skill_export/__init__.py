"""把某项目的长期记忆导出成 Claude Code Skill。

生成 ``<target>/<project_id>/SKILL.md``(YAML frontmatter: name/description)+
``reference/{progress,structure,code_analysis}.md``,内容全部从盘上已有产物拷贝/
拼接而来。纯读写,无 LLM、无 Qdrant——安全,可对 tmp 目录单测。

Skill 装好后,Claude Code 在该项目下对话时可加载这份长期记忆,理解项目进度、
结构与代码细节。
"""
