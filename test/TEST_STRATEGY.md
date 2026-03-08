# Comprehensive Test Strategy for LLMLM Lab Inference Service

## Overview

This document provides a comprehensive test strategy for achieving **90% code coverage** across the FastAPI inference service. The strategy covers unit, integration, and E2E test levels with specific guidance for each component.

## Project Structure

```
/home/lsm/Nextcloud/llmmllab/
├── server/           # FastAPI application layer
│   ├── app.py        # Main entry point with lifespan
│   ├── middleware/   # Auth, validation, db_init
│   ├── routers/      # API endpoints (chat, conversation, model, config)
│   └── db/           # Storage implementations
├── runner/           # Model execution
│   ├── pipeline_factory.py
│   ├── pipeline_cache.py
│   └── pipelines/    # Llama.cpp, txt2img, img2img
├── composer/         # LangGraph orchestration
│   ├── core/         # Service, errors
│   ├── graph/        # State, executor, workflows
│   ├── agents/       # BaseAgent, specialized agents
│   └── tools/        # Dynamic, static tools
├── db/               # Storage interfaces
├── models/           # Generated from YAML schemas
└── test/
    ├── unit/         # Unit tests (create here)
    └── integration/  # Integration tests (existing)
```

---

## 1. UNIT TEST STRATEGY

### 1.1 Server Layer

#### test/unit/server/test_app.py
**Coverage Targets:**
- Lifespan initialization: 100%
- Middleware registration: 100%
- Router inclusion: 100%
- Exception handlers: 100%

**Key Patterns:**
```python
@pytest.fixture
def mock_lifespan_components(mocker):
    """Mock all lifespan initialization components."""
    return {
        'cleanup_service': mocker.patch('server.app.CleanupService'),
        'storage_init': mocker.patch('server.app.storage.initialize'),
        'shutdown_composer': mocker.patch('server.app.shutdown_composer'),
        'pipeline_cache': mocker.patch('server.app.local_pipeline_cache'),
    }

@pytest.mark.asyncio
async def test_lifespan_initialization(mock_lifespan_components):
    """Test lifespan initializes all components in correct order."""
    # Test startup phase
    # Test shutdown phase
```

#### test/unit/server/middleware/test_auth.py
**Coverage Targets:**
- JWT validation: 95%
- API key validation: 100%
- Admin check: 100%
- Cache refresh logic: 90%

**Dependencies to Mock:**
- `httpx.AsyncClient` for JWKS fetch
- `jwt` module for token decoding
- `time` for cache timing

**Key Patterns:**
```python
@pytest.fixture
def mock_jwt_validator(mocker):
    """Mock JWTValidator with controlled JWKS."""
    validator = mocker.create_autospec(JWTValidator)
    validator.validate_token.return_value = TokenValidationResult(
        user_id="test-user",
        claims={},
        is_admin=False
    )
    return validator

@pytest.mark.asyncio
async def test_auth_middleware_valid_token(mock_jwt_validator):
    """Test authenticated request with valid JWT."""
    # Setup: Add token to request headers
    # Execute: Call auth middleware
    # Assert: Request has user_id, is_admin in scope

@pytest.mark.asyncio
async def test_auth_middleware_expired_token(mock_jwt_validator):
    """Test request with expired JWT triggers refresh."""
    # Setup: Mock expired token + JWKS cache timeout
    # Assert: JWKS cache is refreshed
```

#### test/unit/server/routers/test_chat.py
**Coverage Targets:**
- Chat completion: 90%
- File content transformation: 100%
- Error handling: 100%

**Dependencies to Mock:**
- `composer.chat_completion`
- `storage.message.add_message`
- `transform_file_content_to_documents`

#### test/unit/server/routers/test_conversation.py
**Coverage Targets:**
- List conversations: 90%
- Get conversation: 90%
- Delete conversation: 90%
- Replay from timestamp: 85%

**Dependencies to Mock:**
- `storage.conversation` (all methods)
- `storage.message` (all methods)
- `clear_workflow_cache`
- `local_pipeline_cache.cleanup_for_user`

#### test/unit/server/routers/test_model.py
**Coverage Targets:**
- List models: 90%
- Model profile CRUD: 90%
- UUID generation: 100%

**Dependencies to Mock:**
- `ModelLoader.get_available_models()`
- `storage.model_profile` (all methods)

#### test/unit/server/routers/test_config.py
**Coverage Targets:**
- Get user config: 90%
- Update config: 90%
- Default config creation: 100%

**Dependencies to Mock:**
- `storage.user_config` (all methods)
- `create_default_user_config`

---

### 1.2 Database Layer

#### test/unit/db/test_storage.py
**Coverage Targets:**
- Storage initialization: 100%
- Service retrieval: 100%
- Connection recovery: 90%

**Dependencies to Mock:**
- `asyncpg.create_pool`
- All storage classes (ConversationStorage, MessageStorage, etc.)

#### test/unit/db/test_cache_storage.py
**Coverage Targets:**
- Redis operations: 90%
- Cache key generation: 100%
- Error handling (_safe_redis_call): 100%

**Dependencies to Mock:**
- `redis.Redis` client
- `redis.ConnectionError` scenarios

#### test/unit/db/test_interfaces.py
**Coverage Targets:**
- Abstract base classes: 100%
- Interface contracts: 100%

**Pattern:**
```python
def test_message_store_interface():
    """Verify MessageStore interface defines all required methods."""
    required_methods = [
        'add_message', 'get_message', 'get_messages_by_conversation_id',
        'delete_message', 'delete_all_from_message'
    ]
    for method in required_methods:
        assert hasattr(MessageStore, method)
```

---

### 1.3 Runner Layer

#### test/unit/runner/test_pipeline_factory.py
**Coverage Targets:**
- Pipeline creation: 90%
- Cache integration: 90%
- Provider routing (local vs remote): 100%
- Memory coordination: 85%

**Dependencies to Mock:**
- `LocalPipelineCacheManager`
- `LlamaCppServerManager`
- External API clients (OpenAI, Anthropic)

**Key Patterns:**
```python
@pytest.fixture
def mock_local_cache(mocker):
    """Mock local pipeline cache."""
    cache = mocker.create_autospec(LocalPipelineCacheManager)
    cache.get_or_create.return_value = mocker.Mock(spec=BasePipeline)
    cache.is_local.return_value = True
    return cache

@pytest.mark.asyncio
async def test_get_pipeline_local_provider(mock_local_cache):
    """Test pipeline creation for local provider uses cache."""
    # Assert: cache.get_or_create is called
    # Assert: create_pipeline is NOT called for local providers

@pytest.mark.asyncio
async def test_get_pipeline_remote_provider(mock_local_cache):
    """Test pipeline creation for remote provider bypasses cache."""
    # Setup: Model with OpenAI/Anthropic provider
    # Assert: cache.get_or_create is NOT called
    # Assert: create_pipeline IS called
```

#### test/unit/runner/test_pipeline_cache.py
**Coverage Targets:**
- Cache entry creation: 100%
- Eviction score calculation: 100%
- Lock/unlock mechanism: 100%
- Cleanup thread: 90%

**Dependencies to Mock:**
- `weakref.ref` for pipeline references
- `threading` for cleanup thread
- Time functions for eviction scoring

**Key Patterns:**
```python
def test_eviction_score_priority_bonus():
    """Test higher priority pipelines have better eviction scores."""
    entry_high = _PipelineCacheEntry(pipeline, PipelinePriority.HIGH)
    entry_low = _PipelineCacheEntry(pipeline, PipelinePriority.LOW)
    assert entry_high.eviction_score(now) > entry_low.eviction_score(now)

def test_eviction_score_memory_bonus():
    """Test smaller models get eviction bonus."""
    small_pipeline = mocker.Mock()
    entry_small = _PipelineCacheEntry(small_pipeline, PipelinePriority.NORMAL, estimated_memory=1e9)
    entry_large = _PipelineCacheEntry(small_pipeline, PipelinePriority.NORMAL, estimated_memory=15e9)
    assert entry_small.eviction_score(now) > entry_large.eviction_score(now)
```

#### test/unit/runner/test_hardware_manager.py
**Coverage Targets:**
- GPU detection: 100%
- Memory management: 90%
- Process management: 85%
- Thermal monitoring: 80%

**Dependencies to Mock:**
- `torch.cuda` (available, device_count)
- `nvsmi` for GPU stats
- `psutil` for CPU/memory stats

**Key Patterns:**
```python
@pytest.fixture
def mock_torch_cuda(mocker):
    """Mock torch.cuda availability."""
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.device_count', return_value=2)
    mocker.patch('torch.cuda.get_device_name', return_value="GPU-0")

@pytest.mark.asyncio
async def test_gpu_detection_with_cuda(mock_torch_cuda):
    """Test GPU detection when CUDA is available."""
    manager = EnhancedHardwareManager()
    assert manager.has_gpu is True
    assert manager.gpu_count == 2
```

#### test/unit/runner/pipelines/test_llamacpp.py
**Coverage Targets:**
- Server manager integration: 90%
- ChatOpenAI initialization: 90%
- Streaming: 85%
- Tool binding: 90%

**Dependencies to Mock:**
- `LlamaCppServerManager`
- `ChatOpenAI` from langchain_openai
- Server startup/shutdown

---

### 1.4 Composer Layer

#### test/unit/composer/test_service.py
**Coverage Targets:**
- Workflow composition: 90%
- Server interface integration: 85%
- Error handling: 100%

**Dependencies to Mock:**
- `GraphBuilder` implementations
- `ServerAdapter` / `ServerInterface`
- All service methods (user_config, conversation, etc.)

#### test/unit/composer/graph/test_executor.py
**Coverage Targets:**
- Workflow streaming: 90%
- Tool call handling: 90%
- Thinking tag parsing: 100%
- State transitions: 100%

**Dependencies to Mock:**
- `CompiledStateGraph.astream_events`
- `RawToolCallParser`
- `parse_content`, `strip_think_tags`

#### test/unit/composer/agents/test_base.py
**Coverage Targets:**
- Agent creation: 90%
- Middleware handling: 85%
- Tool deduplication: 100%
- Structured output: 85%

**Dependencies to Mock:**
- `create_agent` from langchain.agents
- `BaseChatModel.ainvoke`
- Message conversion functions

#### test/unit/composer/tools/test_dynamic.py
**Coverage Targets:**
- Tool generation: 90%
- Security validation: 100%
- Serialization: 100%

---

## 2. MOCKING STRATEGY

### 2.1 Redis Mocking

**Pattern:**
```python
@pytest.fixture
def mock_redis(mocker):
    """Mock Redis client with controlled behavior."""
    redis_client = mocker.create_autospec(redis.Redis)
    redis_client.ping.return_value = True
    redis_client.get.return_value = None
    redis_client.set.return_value = True
    return redis_client
```

**When to Mock:**
- CacheStorage class
- Any direct Redis operations
- Connection error scenarios

### 2.2 PostgreSQL Mocking

**Pattern:**
```python
@pytest.fixture
def mock_asyncpg_pool(mocker):
    """Mock asyncpg connection pool."""
    pool = mocker.create_autospec(asyncpg.Pool)
    pool.acquire = AsyncMock()
    pool.close = AsyncMock()
    return pool

@pytest.fixture
def mock_db_connection(mock_asyncpg_pool, mocker):
    """Mock database connection."""
    conn = mocker.create_autospec(asyncpg.Connection)
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.execute.return_value = None
    mock_asyncpg_pool.acquire.return_value.__aenter__.return_value = conn
    return conn
```

**When to Mock:**
- All storage classes
- Database queries
- Transaction management

### 2.3 gRPC Mocking

**Pattern:**
```python
@pytest.fixture
def mock_runner_client(mocker):
    """Mock Runner gRPC client."""
    client = mocker.create_autospec(RunnerClient)
    client.execute_pipeline.return_value = PipelineExecutionResponse()
    client.get_cache_stats.return_value = CacheStatsResponse()
    return client
```

### 2.4 External API Mocking

**OpenAI/Anthropic:**
```python
@pytest.fixture
def mock_openai_client(mocker):
    """Mock OpenAI client."""
    client = mocker.MagicMock()
    client.chat.completions.create.return_value = ChatCompletion(...)
    return client
```

**HTTP Clients (JWKS, etc.):**
```python
@pytest.fixture
def mock_httpx_client(mocker):
    """Mock httpx.AsyncClient."""
    client = mocker.AsyncMock()
    client.get.return_value = mocker.Mock(
        json=lambda: {"keys": []},
        raise_for_status=lambda: None
    )
    return client
```

### 2.5 Time and Threading

**Pattern:**
```python
@pytest.fixture
def mock_time(mocker):
    """Mock time.time for eviction scoring tests."""
    current_time = 1000.0
    def time_side_effect():
        return current_time
    return mocker.patch('time.time', side_effect=time_side_effect)
```

---

## 3. COVERAGE TARGETS BY COMPONENT

### Server Layer
| Component | Target | Notes |
|-----------|--------|-------|
| app.py | 95% | Lifespan, middleware |
| middleware/auth.py | 90% | JWT, API key, admin checks |
| routers/chat.py | 85% | Chat completion, file transforms |
| routers/conversation.py | 85% | CRUD operations |
| routers/model.py | 85% | Model/profile management |
| routers/config.py | 85% | User config CRUD |
| **Server Total** | **88%** | |

### Database Layer
| Component | Target | Notes |
|-----------|--------|-------|
| db/__init__.py | 90% | Storage initialization |
| db/cache_storage.py | 85% | Redis operations |
| db/interfaces.py | 100% | Abstract base classes |
| db/*.storage.py | 80% | Per-storage implementations |
| **Database Total** | **86%** | |

### Runner Layer
| Component | Target | Notes |
|-----------|--------|-------|
| pipeline_factory.py | 90% | Pipeline creation, caching |
| pipeline_cache.py | 90% | Cache management, eviction |
| hardware_manager.py | 85% | GPU detection, memory |
| pipelines/llamacpp/*.py | 85% | Chat, embedding pipelines |
| pipelines/txt2img/*.py | 80% | Image generation |
| **Runner Total** | **87%** | |

### Composer Layer
| Component | Target | Notes |
|-----------|--------|-------|
| core/service.py | 85% | Workflow composition |
| graph/executor.py | 90% | Streaming execution |
| graph/state.py | 100% | State schema |
| agents/base.py | 85% | Agent execution |
| tools/dynamic/*.py | 85% | Tool generation |
| **Composer Total** | **88%** | |

### Overall Coverage
| Level | Target |
|-------|--------|
| Unit Tests | 85% |
| Integration Tests | 75% |
| **Combined** | **90%** |

---

## 4. INTEGRATION TEST ENHANCEMENTS

### Existing Integration Tests (test/integration/)

#### test/integration/test_server.py
**Current Coverage:** Health, docs, OpenAPI endpoints
**Enhancement:**
- Add full chat completion integration
- Add conversation CRUD integration
- Add model profile CRUD integration

#### test/integration/test_database.py
**Current Coverage:** Connection, extensions, tables
**Enhancement:**
- Add data insertion and retrieval
- Add query performance benchmarks
- Add connection recovery scenarios

#### test/integration/test_runner.py
**Current Coverage:** Imports, factory methods
**Enhancement:**
- Add end-to-end pipeline execution
- Add model loading from disk
- Add cache eviction scenarios

#### test/integration/test_composer.py
**Current Coverage:** Imports, workflow builders
**Enhancement:**
- Add full workflow execution
- Add tool calling integration
- Add streaming integration

---

## 5. CI/CD INTEGRATION

### pytest Configuration (pytest.ini)
```ini
[pytest]
asyncio_mode = auto
testpaths = test/unit test/integration
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (require services)
    slow: Tests that take > 5 seconds
    database: Tests requiring PostgreSQL
    redis: Tests requiring Redis
```

### Test Execution Strategy
```bash
# Local development
pytest test/unit -v --cov=server --cov=runner --cov=composer --cov-report=term-missing

# CI/CD
pytest test/unit -m "not slow" --cov=server --cov=runner --cov=composer
pytest test/integration --cov=server --cov=runner --cov=composer --cov-report=html
```

### Coverage Thresholds
```yaml
# .coveragerc
[run]
source = server, runner, composer

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:

fail_under = 85
```

---

## 6. FLAKINESS MITIGATION

### Common Flaky Test Patterns

1. **Race Conditions in Cache Tests**
   - Use `asyncio.sleep(0.01)` between operations
   - Use `pytest-asyncio` with strict mode

2. **Time-based Tests**
   - Always mock `time.time()` for eviction tests
   - Use `freezegun` for datetime tests

3. **External Service Tests**
   - Use `pytest.mark.skipif` for external dependencies
   - Provide mock fallbacks

4. **Database Tests**
   - Use transactions with rollback
   - Clean schema between test classes

---

## 7. FILE STRUCTURE

```
test/
├── unit/
│   ├── server/
│   │   ├── test_app.py
│   │   ├── test_config.py
│   │   └── middleware/
│   │       ├── test_auth.py
│   │       ├── test_db_init.py
│   │       └── test_message_validation.py
│   ├── routers/
│   │   ├── test_chat.py
│   │   ├── test_conversation.py
│   │   ├── test_model.py
│   │   └── test_config.py
│   ├── db/
│   │   ├── test_storage.py
│   │   ├── test_cache_storage.py
│   │   └── test_interfaces.py
│   ├── runner/
│   │   ├── test_pipeline_factory.py
│   │   ├── test_pipeline_cache.py
│   │   ├── test_hardware_manager.py
│   │   └── pipelines/
│   │       ├── test_llamacpp.py
│   │       └── test_txt2img.py
│   └── composer/
│       ├── test_service.py
│       ├── test_executor.py
│       ├── test_state.py
│       ├── agents/
│       │   └── test_base.py
│       └── tools/
│           ├── test_dynamic.py
│           └── test_static.py
├── integration/
│   ├── conftest.py
│   ├── test_server.py
│   ├── test_database.py
│   ├── test_runner.py
│   ├── test_composer.py
│   └── test_e2e_chat.py
└── e2e/
    └── test_full_workflow.py
```

---

## 8. TESTING BEST PRACTICES

### Do
1. Use `pytest-asyncio` with `asyncio_mode=auto`
2. Mock external dependencies (Redis, DB, gRPC)
3. Use fixtures for common test data
4. Assert on exact error types and messages
5. Keep tests isolated and deterministic

### Don't
1. Test implementation details (private methods)
2. Create integration tests that don't need them
3. Share mutable state between tests
4. Rely on external services without mocks
5. Use `time.sleep()` in tests (use time mocking)

---

## 9. NEXT STEPS

1. **Create unit test structure** following the file layout above
2. **Implement high-priority tests** (90%+ coverage items first)
3. **Set up CI/CD** with coverage reporting
4. **Add integration enhancements** for end-to-end workflows
5. **Document test patterns** in this file as they're discovered

---

*Last updated: 2026-03-08*