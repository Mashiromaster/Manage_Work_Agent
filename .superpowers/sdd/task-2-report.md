# Task 2 Report: replace_profile 透传 locked/source

## Status
DONE

## Implemented
`ProfileStore.replace_profile` (memory_framework/profile_store.py) 构造 `ProfileItem`
时新增两字段透传:`locked=bool(d.get("locked", False))` 与
`source=str(d.get("source", "llm"))`,使全量重导入不再静默丢弃锁定态与来源。

## TDD

### RED
Cmd: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile_store.py::test_replace_profile_preserves_locked_and_source -q`
Output: `1 failed` — `AssertionError: assert False is True`,ProfileItem.locked 恒为默认 False。

### GREEN
Cmd: `LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 PYTHONPATH=. conda run -n mem0 python -m pytest tests/test_profile_store.py -q`
Output: `8 passed in 0.04s`

## Files Changed
- memory_framework/profile_store.py (+2):replace_profile 循环内加 locked/source
- tests/test_profile_store.py (+13):test_replace_profile_preserves_locked_and_source

## Commit
3ae1b26 feat(store): replace_profile 透传 locked/source(无 Co-Authored-By trailer)

## Self-Review
- 改动完全匹配 brief Step 3 的规定代码,与 Task 1 的 ProfileItem 字段/默认值一致。
- 仅触及 profile_store.py 与 test_profile_store.py 两文件,无越界。
- ingest 路径未涉及(仅 replace_profile 走全量导入),符合任务范围。

## Concerns
无。
