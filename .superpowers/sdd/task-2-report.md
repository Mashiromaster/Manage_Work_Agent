# Task 2 报告:配置模块 config.py

状态:DONE

## 1. mem0ai 2.0.11 真实配置结构关键发现

探查命令(conda 环境 mem0)得到的确切结构:

- `Memory.from_config(config_dict: Dict[str, Any])`。
- `MemoryConfig.model_fields` 键:`vector_store`, `llm`, `embedder`, `history_db_path`, `reranker`, `version`, `custom_instructions`。三大块键名 = `llm` / `embedder` / `vector_store`(与草稿一致)。
- `LlmConfig` / `EmbedderConfig` 结构均为 `{provider: str, config: dict}`;它们对 `config` 只做 provider 白名单校验,**不校验 config 内部键**(浅校验)。
- `litellm` 在 LLM 白名单;`huggingface` 在 embedder 白名单;`qdrant` 在 vector store 映射表。

### 关键发现 A —— litellm 的 config 类是 BaseLlmConfig,不接受 api_base
`mem0.utils.factory.LlmFactory.provider_to_class` 中 `"litellm": (..., BaseLlmConfig)`。
`BaseLlmConfig.__init__` 参数只有:`model, temperature, api_key, max_tokens, top_p, top_k, enable_vision, vision_details, reasoning_effort, http_client_proxies, is_reasoning_model`。
**没有 `api_base`。** 实测:
```
BaseLlmConfig(model='anthropic/x', api_key='k', api_base='https://x')
-> TypeError: BaseLlmConfig.__init__() got an unexpected keyword argument 'api_base'
```
`LlmFactory.create("litellm", config_dict)` 会执行 `BaseLlmConfig(**config_dict)`,因此若把 `api_base` 放进 `llm.config`,`Memory.from_config` 在实例化 LLM 时必崩。

`MemoryConfig(**cfg)`(即任务第 5 步的验证)对 llm.config 只做浅校验,**即使含 api_base 也会“通过”**——所以单靠第 5 步无法发现该 bug。我额外跑了 LlmFactory 实例化验证(见第 4 节 5b)。

### 关键发现 B —— 中转站基址走环境变量,不走 config
litellm anthropic provider 在 `main.py` 从 `get_secret("ANTHROPIC_API_BASE")` 读基址(回退 `https://api.anthropic.com/v1/messages`),从 `ANTHROPIC_API_KEY` 读密钥。
用户提供的是 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`(名字不同)。故 `build_config()` 把它们桥接到 litellm 识别的 `ANTHROPIC_API_BASE` / `ANTHROPIC_API_KEY`,基址不进 config,token 另作为 `api_key` 进 config(该键 BaseLlmConfig 接受)。

### 关键发现 C —— embedder 用 embedding_dims,qdrant 用 embedding_model_dims
`BaseEmbedderConfig` / HuggingFace embedder 用键 `embedding_dims`;`QdrantConfig` 用键 `embedding_model_dims`。两块键名不同,不能混用。QdrantConfig 还禁止额外字段并要求 `path` 或 `host+port` 或 `url+api_key` 之一。

## 2. 最终 config.py

见 `/Users/mashiro/Mem0/memory_framework/config.py`。核心:校验两个环境变量缺失则抛 `MissingConfigError`;把 `ANTHROPIC_BASE_URL`→`ANTHROPIC_API_BASE`、`ANTHROPIC_AUTH_TOKEN`→`ANTHROPIC_API_KEY` 桥接进 env;返回三块 dict,`llm.config` 含 `model/api_key/temperature/max_tokens`(**无 api_base**),`embedder.config` 用 `embedding_dims`,`vector_store.config` 用 `path` + `embedding_model_dims`。常量:`DEFAULT_LLM_MODEL="anthropic/Claude-Sonnet-4.7-hq"`, `EMBED_MODEL="BAAI/bge-small-zh-v1.5"`, `EMBED_DIMS=384`, `QDRANT_PATH="./qdrant_data"`。

## 3. 测试命令 + 输出
```
python -m pytest tests/test_config.py -v
tests/test_config.py::test_build_config_has_three_sections PASSED   [ 33%]
tests/test_config.py::test_build_config_embedder_is_local_bge PASSED [ 66%]
tests/test_config.py::test_missing_token_raises PASSED              [100%]
3 passed in 0.00s
```

## 4. 第 5 步结构合法性验证
- 5a(任务要求):`MemoryConfig(**build_config())` -> `OK: config 被 mem0 MemoryConfig 接受`。
- 5b(更深,自加):`LlmFactory.create('litellm', cfg['llm']['config'])` -> `OK: LlmFactory 实例化 litellm 成功, model=anthropic/Claude-Sonnet-4.7-hq, api_key set=True, ANTHROPIC_API_BASE bridged=https://x`。证明无 api_base 是正确的;若按草稿含 api_base,这一步会 TypeError。
- 附加:`BaseEmbedderConfig(**embedder.config)` 与 `QdrantConfig(**vector_store.config)` 均实例化成功(dims 384,path ./qdrant_data)。

## 5. 与草稿不同处及原因
1. **删除 `llm.config` 中的 `api_base`**。原因:发现 A —— litellm provider 的 config 类是 BaseLlmConfig,不接受 api_base,会导致运行时 TypeError。改为把基址桥接到 `ANTHROPIC_API_BASE` 环境变量(发现 B)。
2. **新增环境变量桥接**:`ANTHROPIC_BASE_URL`→`ANTHROPIC_API_BASE`、`ANTHROPIC_AUTH_TOKEN`→`ANTHROPIC_API_KEY`。原因:litellm 用这两个变量名,而用户/项目用的是另一组名字。
3. 其余(键名 embedding_dims / embedding_model_dims、provider 名、异常、默认模型)与草稿一致,经真实 config 类验证无误。

## 6. commit hash
见仓库(commit message: "feat: config module with litellm+bge+qdrant")。

## 修复:副作用拆分

### 背景
审查发现 `build_config()` 名为纯构造函数,却在调用时偷偷写全局环境变量
(`ANTHROPIC_API_BASE` / `ANTHROPIC_API_KEY`)桥接给 litellm,违反直觉且会造成
测试间 env 污染。

### 新的函数划分
- `build_config() -> dict`:**纯函数**,只读 env 并构造返回 config dict,不改任何全局状态。
- `apply_litellm_env() -> None`:**显式副作用函数**,把 `ANTHROPIC_BASE_URL` /
  `ANTHROPIC_AUTH_TOKEN` 桥接到 litellm 识别的 `ANTHROPIC_API_BASE` /
  `ANTHROPIC_API_KEY`。缺变量时抛 `MissingConfigError`。
- `build_config_and_apply_env() -> dict`:便捷入口,先 apply env 再 build,
  供创建 `Memory` 前的启动路径一次性调用。
- 模块 docstring 新增"使用约定"节,明确说明"创建 Memory 前必须先调用 apply_litellm_env()"。

### 测试命令 + 完整输出
`pytest tests/test_config.py -v`
```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 7 items

tests/test_config.py::test_build_config_has_three_sections PASSED         [ 14%]
tests/test_config.py::test_build_config_embedder_is_local_bge PASSED      [ 28%]
tests/test_config.py::test_missing_token_raises PASSED                    [ 42%]
tests/test_config.py::test_build_config_has_no_side_effects_on_environ PASSED [ 57%]
tests/test_config.py::test_apply_litellm_env_sets_bridge_vars PASSED      [ 71%]
tests/test_config.py::test_apply_litellm_env_missing_base_url_raises PASSED [ 85%]
tests/test_config.py::test_build_config_and_apply_env_does_both PASSED    [100%]

============================== 7 passed in 0.01s ===============================
```
新增测试:
- `test_build_config_has_no_side_effects_on_environ`:断言 build_config 调用前后 os.environ 完全相等,且不出现桥接变量。
- `test_apply_litellm_env_sets_bridge_vars`:断言 apply_litellm_env 正确设置桥接 env(monkeypatch 隔离,测完自动还原)。
- `test_apply_litellm_env_missing_base_url_raises` / `test_build_config_and_apply_env_does_both`:补充覆盖。

### 深验证输出
```
OK
```
(config 仍能被 `MemoryConfig(**build_config())` 消费。)
