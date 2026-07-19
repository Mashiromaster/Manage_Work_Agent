"""对话测试集的检索召回评测。

对指定的对话集:灌入全部轮次 → 用预埋事实构造 query 检索 →
判断召回结果是否命中预期关键词(expect_any 任一命中即算召回)→ 汇总准确率。

用法:
    # 默认抽样跑前 2 个对话集(省 API)
    PYTHONPATH=. python tests/evaluate.py
    # 指定对话集
    PYTHONPATH=. python tests/evaluate.py chef_lin dev_zhao
    # 跑全部 10 个(慢、多 API 调用)
    PYTHONPATH=. python tests/evaluate.py --all
"""

import glob
import json
import os
import sys

from memory_framework.conversation import ingest, load_conversation
from memory_framework.memory_store import MemoryStore

TESTSET_DIR = "data/testsets"


def _load_facts(user_id: str) -> dict:
    with open(f"{TESTSET_DIR}/{user_id}.facts.json", encoding="utf-8") as f:
        return json.load(f)


def _hit(results: list, expect_any: list) -> bool:
    joined = " ".join(
        str(r.get("memory", r) if isinstance(r, dict) else r) for r in results
    )
    return any(kw in joined for kw in expect_any)


def evaluate_one(store: MemoryStore, user_id: str) -> dict:
    conv = load_conversation(f"{TESTSET_DIR}/{user_id}.json")
    store.delete_all(user_id=user_id)
    ingest(store, conv)

    facts = _load_facts(user_id)
    checks = []
    for aspect, spec in facts.items():
        results = store.search(spec["query"], user_id=user_id, limit=5)
        hit = _hit(results, spec["expect_any"])
        checks.append({"aspect": aspect, "query": spec["query"], "hit": hit,
                       "expect_any": spec["expect_any"]})
    hits = sum(c["hit"] for c in checks)
    return {"user_id": user_id, "turns": len(conv["messages"]),
            "total": len(checks), "hits": hits, "checks": checks}


def _pick_targets(argv: list) -> list:
    all_ids = sorted(os.path.basename(p)[:-5]
                     for p in glob.glob(f"{TESTSET_DIR}/*.json")
                     if not p.endswith(".facts.json"))
    if "--all" in argv:
        return all_ids
    named = [a for a in argv if not a.startswith("--")]
    if named:
        return named
    return all_ids[:2]  # 默认抽样前 2 个


def main() -> None:
    targets = _pick_targets(sys.argv[1:])
    print(f"评测对话集:{targets}\n")
    store = MemoryStore()

    reports = []
    for uid in targets:
        print(f"→ 灌入并评测 {uid}(70 轮,真实调用 Claude,较慢)...")
        rep = evaluate_one(store, uid)
        reports.append(rep)
        for c in rep["checks"]:
            mark = "✓" if c["hit"] else "✗"
            print(f"   {mark} [{c['aspect']}] 查:{c['query']} → 期望含 {c['expect_any']}")
        print(f"   小结:{rep['hits']}/{rep['total']} 召回\n")

    total = sum(r["total"] for r in reports)
    hits = sum(r["hits"] for r in reports)
    print("=" * 50)
    print(f"总召回率:{hits}/{total} = {hits / total * 100:.1f}%"
          if total else "无评测项")

    os.makedirs("data", exist_ok=True)
    with open("data/eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"reports": reports, "summary": {"hits": hits, "total": total}},
                  f, ensure_ascii=False, indent=2)
    print("详细报告 → data/eval_report.json")


if __name__ == "__main__":
    main()
