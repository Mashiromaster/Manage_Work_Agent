"""项目进度追踪的四个维度定义。

复用用户画像的 ProfileItem / ProfileStore 存储结构,但用一套独立的维度集合,
使项目记忆与用户画像互不污染。四个维度对应用户想追踪的四类信息:

- progress  进度:做到哪了 / 已完成的里程碑
- blocker   问题/障碍:bug、报错、卡点、未解决的困难
- todo      待办/待优化:TODO、待优化、"以后再说"的事项
- decision  决策记录:关键技术选择及其理由
"""

PROGRESS = "progress"
BLOCKER = "blocker"
TODO = "todo"
DECISION = "decision"

PROJECT_DIMENSIONS = [PROGRESS, BLOCKER, TODO, DECISION]

PROJECT_DIMENSION_LABELS = {
    PROGRESS: "进度(做到哪了)",
    BLOCKER: "问题 / 障碍",
    TODO: "待办 / 待优化",
    DECISION: "决策记录",
}
