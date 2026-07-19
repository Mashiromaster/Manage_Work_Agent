import pytest
from memory_framework.memory_store import MemoryStore


@pytest.fixture(scope="module")
def store():
    s = MemoryStore()
    s.delete_all(user_id="pytest_user")
    yield s
    s.delete_all(user_id="pytest_user")


def test_add_then_search_recalls(store):
    """验证 add→抽取→存储→语义检索整条链路:检索能召回相关记忆。

    断言"检索返回非空结果"这一行为,而非模型输出的确切字面,
    以免受存储语言(中/英)或措辞差异影响而脆断。
    """
    store.add(
        messages=[{"role": "user", "content": "我不吃香菜,对海鲜过敏"}],
        user_id="pytest_user",
    )
    results = store.search(query="这个人的饮食禁忌是什么", user_id="pytest_user", limit=5)
    assert isinstance(results, list) and len(results) >= 1, "语义检索应召回至少一条相关记忆"


def test_get_all_returns_list(store):
    store.add(messages=[{"role": "user", "content": "我住在上海"}], user_id="pytest_user")
    allm = store.get_all(user_id="pytest_user")
    assert isinstance(allm, list) and len(allm) >= 1
