"""端到端演示(非交互):灌入演示对话 → 语义检索召回。

用法::

    conda run -n mem0 python examples/demo.py
"""

import glob

from memory_framework.conversation import ingest, load_conversation
from memory_framework.memory_store import MemoryStore


def main() -> None:
    store = MemoryStore()

    for path in sorted(glob.glob("data/conversations/*.json")):
        conv = load_conversation(path)
        print(f"灌入 {conv['user_id']} 的对话...")
        ingest(store, conv)

    print("\n--- 检索演示 ---")
    for user_id, query in [
        ("alice", "这个人有什么饮食禁忌"),
        ("bob", "他的技术背景是什么"),
    ]:
        results = store.search(query, user_id=user_id, limit=5)
        print(f"\n[{user_id}] 提问:{query}")
        for r in results:
            print("  •", r.get("memory", r) if isinstance(r, dict) else r)


if __name__ == "__main__":
    main()
