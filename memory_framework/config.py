"""mem0 配置构造模块。

负责把环境变量拼装成 mem0ai 2.0.11 可接受的配置 dict:

- LLM:litellm provider,经京东云中转站调用 Claude(anthropic 兼容)。
- Embedding:本地 HuggingFace bge 模型(384 维)。
- Vector store:Qdrant 本地文件模式。

关键实现说明(基于对 mem0ai 2.0.11 源码的探查):

mem0 的 litellm provider 使用 ``BaseLlmConfig`` 作为其 config 类
(见 ``mem0.utils.factory.LlmFactory.provider_to_class``)。
``BaseLlmConfig.__init__`` **不接受** ``api_base`` 参数,因此绝不能把
``api_base`` 放进 ``llm.config``——否则 ``Memory.from_config`` 在实例化
LLM 时会抛 ``TypeError``。

litellm 的 anthropic provider 从环境变量 ``ANTHROPIC_API_BASE`` 读取基址、
从 ``ANTHROPIC_API_KEY`` 读取密钥(见 litellm ``main.py`` 对
``get_secret("ANTHROPIC_API_BASE")`` 的处理)。所以这里的做法是:把用户提供的
``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` 桥接到 litellm 识别的
``ANTHROPIC_API_BASE`` / ``ANTHROPIC_API_KEY``,再把 token 作为 ``api_key``
放进 llm.config(该键 ``BaseLlmConfig`` 接受)。

使用约定
--------

``build_config()`` 是**纯函数**:只读环境变量并构造返回 config dict,
不修改任何进程全局状态。

而把基址/密钥桥接到 litellm 识别的环境变量是一个**显式副作用**,被隔离在
``apply_litellm_env()`` 中。**在创建 mem0 ``Memory`` 之前必须先调用
``apply_litellm_env()``**,否则 litellm 无法拿到中转站基址。便捷入口
``build_config_and_apply_env()`` 一次性完成两件事,推荐在应用启动路径使用。
"""

import os

DEFAULT_LLM_MODEL = "anthropic/Claude-Opus-4.8-hq"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIMS = 512
QDRANT_PATH = "./qdrant_data"
# Opus 4.x 推理模型仅支持 temperature=1(litellm 会对 0.1 报 UnsupportedParamsError)。
LLM_TEMPERATURE = 1
LLM_MAX_TOKENS = 1024
# 注入 mem0 抽取 prompt,强制用中文存储记忆(默认会把中文事实翻成英文)。
CUSTOM_INSTRUCTIONS = "请始终用简体中文提取和记录记忆事实,保留原文中的中文词汇,不要翻译成英文。"


class MissingConfigError(RuntimeError):
    """缺少必需的环境变量时抛出。"""


def build_config() -> dict:
    """构造并返回 mem0 配置 dict(**纯函数,无副作用**)。

    只读取环境变量并组装 dict,不修改任何进程全局状态。若真实创建
    ``Memory`` 前需要 litellm 拿到中转站基址,请另行调用
    :func:`apply_litellm_env`(或直接用 :func:`build_config_and_apply_env`)。

    Returns:
        含 ``llm`` / ``embedder`` / ``vector_store`` 三块的配置 dict。

    Raises:
        MissingConfigError: 缺少 ``ANTHROPIC_AUTH_TOKEN`` 或 ``ANTHROPIC_BASE_URL``。
    """
    token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if not token:
        raise MissingConfigError("缺少环境变量 ANTHROPIC_AUTH_TOKEN")
    if not base_url:
        raise MissingConfigError("缺少环境变量 ANTHROPIC_BASE_URL")

    model = os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)

    return {
        "llm": {
            "provider": "litellm",
            "config": {
                # 注意:不要放 api_base,BaseLlmConfig 不接受该键会导致
                # Memory.from_config 实例化 LLM 时 TypeError。
                "model": model,
                "api_key": token,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": EMBED_MODEL,
                # HuggingFace embedder 用的键是 embedding_dims(非 embedding_model_dims)。
                "embedding_dims": EMBED_DIMS,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": QDRANT_PATH,
                # Qdrant 用的键是 embedding_model_dims。
                "embedding_model_dims": EMBED_DIMS,
            },
        },
        # 顶层字段(与 llm/embedder/vector_store 平级),注入抽取 prompt 强制中文存储。
        "custom_instructions": CUSTOM_INSTRUCTIONS,
    }


def apply_litellm_env() -> None:
    """把中转站基址/密钥桥接到 litellm 识别的环境变量(**显式副作用**)。

    litellm 的 anthropic provider 从 ``ANTHROPIC_API_BASE`` 读取基址、从
    ``ANTHROPIC_API_KEY`` 读取密钥,而 ``BaseLlmConfig`` 不接受 ``api_base``,
    故基址只能走环境变量这条路。**在创建 mem0 ``Memory`` 之前必须调用本函数。**

    Raises:
        MissingConfigError: 缺少 ``ANTHROPIC_AUTH_TOKEN`` 或 ``ANTHROPIC_BASE_URL``。
    """
    token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if not token:
        raise MissingConfigError("缺少环境变量 ANTHROPIC_AUTH_TOKEN")
    if not base_url:
        raise MissingConfigError("缺少环境变量 ANTHROPIC_BASE_URL")

    os.environ["ANTHROPIC_API_BASE"] = base_url
    os.environ["ANTHROPIC_API_KEY"] = token

    # Opus 4.x 等推理模型拒绝 temperature/top_p/top_k 等采样参数覆盖,而 mem0
    # 内部会默认注入这些参数。开启 drop_params 让 litellm 自动丢弃目标模型不支持的
    # 参数,而非逐个从 config 里删(删了也挡不住 mem0 内部再注入默认值)。
    import litellm

    litellm.drop_params = True


def build_config_and_apply_env() -> dict:
    """便捷入口:先桥接 litellm 环境变量,再构造并返回 config dict。

    等价于先调 :func:`apply_litellm_env` 再调 :func:`build_config`。应用启动
    路径(创建 ``Memory`` 前)推荐使用本函数,以保证 litellm 拿得到中转站基址。
    """
    apply_litellm_env()
    return build_config()
