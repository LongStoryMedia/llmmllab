# TODO

## Tasks

### [ ] Resolve ModelProfile field shadowing in langchain v1.0.3+:

```log
/opt/venv/shared/lib/python3.12/site-packages/pydantic/_internal/_fields.py:198: UserWarning: Field name "profile" in "BasePipeline" shadows an attribute in parent "BaseChatModel"
  warnings.warn(
/opt/venv/shared/lib/python3.12/site-packages/pydantic/_internal/_fields.py:198: UserWarning: Field name "profile" in "BaseLlamaCppPipeline" shadows an attribute in parent "BasePipeline"
  warnings.warn(
/app/runner/pipelines/llamacpp/base_llamacpp.py:155: LangChainBetaWarning: The method `BaseChatModel.profile` is in beta. It is actively being worked on, so the API may change.
  params = self.profile.parameters
Error creating pipeline for huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated: 'dict' object has no attribute 'parameters'
2025-11-04T00:45:09.131413Z [error    ] Failed create_agent_run        [llmmllab] component=ClassifierAgent conversation_id=None error="'dict' object has no attribute 'parameters'" error_type=AttributeError message_count=1 node_id=62e3e248c3284cd0b1a2ce923011efd2 node_name=IntentClassifier node_type=IntentClassifierNode operation=create_agent_run user_id=test_composer_user_4f71f302
/opt/venv/shared/lib/python3.12/site-packages/structlog/_base.py:173: UserWarning: Remove `format_exc_info` from your processor chain if you want pretty exceptions.
  event_dict = proc(self._logger, method_name, event_dict)
2025-11-04T00:45:09.131768Z [error    ] Intent analysis subgraph execution failed: Node '[IntentClassifier] create_agent_run failed: 'dict' object has no attribute 'parameters'' failed [llmmllab] component=PlanningIntentSubgraph
Traceback (most recent call last):
  File "/app/composer/agents/base_agent.py", line 495, in run
    agent = self._get_or_create_agent(system_prompt, tools, priority, grammar)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/composer/agents/base_agent.py", line 194, in _get_or_create_agent
    chat_model = self.pipeline_factory.get_pipeline(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipeline_factory.py", line 243, in get_pipeline
    pipeline = self.local_cache.get_or_create(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipeline_cache.py", line 103, in get_or_create
    pipeline = create_fn(model, profile, grammar)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipeline_factory.py", line 241, in create_with_coordination
    return self.create_pipeline(m, p, g)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipeline_factory.py", line 372, in create_pipeline
    return self._create_text_pipeline(model, profile, grammar)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipeline_factory.py", line 438, in _create_text_pipeline
    return Qwen3VLPipeline(model, profile, grammar)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipelines/imgtxt2txt/qwen3_vl.py", line 31, in __init__
    super().__init__(model, profile, grammar, **kwargs)
  File "/app/runner/pipelines/llamacpp/base_llamacpp.py", line 108, in __init__
    self.llama_instance = self._initialize_llama(self._get_gguf_path())
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipelines/imgtxt2txt/qwen3_vl.py", line 61, in _initialize_llama
    return super()._initialize_llama(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/runner/pipelines/llamacpp/base_llamacpp.py", line 155, in _initialize_llama
    params = self.profile.parameters
             ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'dict' object has no attribute 'parameters'
```