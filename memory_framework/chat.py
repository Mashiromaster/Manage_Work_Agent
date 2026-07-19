"""长期记忆 CLI 聊天机器人。

每轮对话:
1. 用用户输入去 MemoryStore 语义检索相关长期记忆;
2. 把记忆作为上下文注入 system prompt,调 Claude(经 litellm/中转站)生成回复;
3. 把这轮 user+assistant 对话交给 MemoryStore,由 mem0 抽取并存储新记忆。

用法::

    conda run -n mem0 python -m memory_framework.chat --user alice

命令:输入 ``/mem`` 查看当前用户已存记忆,``/quit`` 退出。
"""

import argparse

import litellm

from memory_framework.config import (
    DEFAULT_LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    build_config_and_apply_env,
)
from memory_framework.memory_store import MemoryStore
from memory_framework.profile_extractor import synthesize_profile
from memory_framework.profile_store import ProfileStore

import os

SYSTEM_BASE = "你是一个有长期记忆的中文助手。请自然、简洁地回答用户。"


def update_profile_from_memories(profile_store, store, user_id, model=None, now=None):
    """用 LLM 从全部记忆中提炼结构化画像,全量替换旧画像。

    与旧版的关键区别:由 Claude 从事件推断性格特征,而非关键词匹配+原文复述。
    """
    model = model or os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    mems = store.get_all(user_id=user_id)
    texts = [m.get("memory", m) if isinstance(m, dict) else m for m in mems]
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return profile_store.get_profile(user_id, now=now)

    # 读取现有画像条目传给 LLM 做增量合并
    existing = profile_store.get_profile(user_id, now=now, include_forgotten=True)
    existing_items = [it.to_dict() for it in existing.items]

    # LLM 提炼结构化画像
    items_data = synthesize_profile(
        memories=texts,
        existing_items=existing_items,
        model=model,
    )

    if items_data:
        return profile_store.replace_profile(user_id, items_data, now=now)
    # LLM 调用失败时回退到旧画像
    return existing


def _build_system_prompt(memories: list, persona: str = "") -> str:
    parts = [SYSTEM_BASE]
    if persona and persona.strip():
        parts.append(f"## 用户设定的人设 / 规则(请严格遵循)\n{persona.strip()}")
    if memories:
        lines = [str(m.get("memory", m)) if isinstance(m, dict) else str(m)
                 for m in memories]
        recall = "\n".join(f"- {line}" for line in lines)
        parts.append(f"关于当前用户你已知道的事(可参考,但不要生硬复述):\n{recall}")
    return "\n\n".join(parts)


def reply(store: MemoryStore, user_id: str, user_input: str, model: str,
          persona: str = "", inject: bool = True) -> str:
    """检索记忆 → 注入上下文 → 调 Claude 生成回复(仅关键路径,不落盘)。

    只做用户等待的那一步:检索+生成。记忆抽取和画像提炼较慢,已拆到
    :func:`persist_turn`,由调用方在回复返回后异步/后续执行,以降低感知延迟。

    Args:
        persona: 用户手写的全局人设/系统提示词,inject=True 时注入 system。
        inject: 是否注入个性化上下文(人设 + 记忆)。False 时退化为无记忆的
            普通助手(system 仅 SYSTEM_BASE),并跳过记忆检索。
    """
    if inject:
        memories = store.search(query=user_input, user_id=user_id, limit=5)
        system = _build_system_prompt(memories, persona=persona)
    else:
        system = SYSTEM_BASE
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp["choices"][0]["message"]["content"]


def persist_turn(store: MemoryStore, profile_store, user_id: str,
                 user_input: str, answer: str, model: str = None) -> None:
    """落盘本轮对话:mem0 抽取并存储新记忆。

    只做记忆抽取(调 Claude,较慢),故从 :func:`reply` 关键路径移出,由调用方
    在回复返回后执行。``profile_store``/``model`` 参数保留以兼容既有调用方,但
    不再自动提炼结构化画像(该功能已从聊天链路移除)。
    """
    store.add(
        messages=[
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ],
        user_id=user_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mem0 长期记忆聊天机器人")
    parser.add_argument("--user", default="default_user", help="用户 ID(记忆隔离键)")
    args = parser.parse_args()

    # 确保 litellm 环境已桥接(MemoryStore 构造时也会做,这里显式一次更清晰)。
    build_config_and_apply_env()
    model = os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)

    store = MemoryStore()
    print(f"记忆聊天已就绪(user={args.user})。输入 /mem 看记忆,/quit 退出。\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_input:
            continue
        if user_input == "/quit":
            print("再见。")
            break
        if user_input == "/mem":
            mems = store.get_all(user_id=args.user)
            if not mems:
                print("(还没有存储任何记忆)\n")
            else:
                for m in mems:
                    print("  •", m.get("memory", m) if isinstance(m, dict) else m)
                print()
            continue

        answer = reply(store, args.user, user_input, model)
        print(f"助手: {answer}\n")
        # CLI 同步补跑落盘(存记忆+提炼画像),保持与旧行为一致。
        persist_turn(store, ProfileStore(), args.user, user_input, answer, model)


if __name__ == "__main__":
    main()
