# Integration Test Strategy

This document outlines the integration testing strategy for the llmmllab project, covering testing approach, test types, and test execution for the Server, Composer, and Runner components.

## Overview

The project uses a multi-layered integration testing approach:

1. **Component-level tests** - Test individual components (Server, Composer, Runner) in isolation
2. **Cross-component tests** - Test interactions between components
3. **End-to-end tests** - Test full request flows through the system

## Test Organization

```
test/
└── integration/
    ├── test_composer.py      # Composer component tests
    ├── test_runner.py        # Runner component tests
    ├── test_server.py        # Server component tests
    ├── test_e2e_flow.py      # End-to-end request flows
    ├── test_database.py      # Database integration tests
    └── test_integration_setup.py  # Test environment validation
```

## Component Tests

### Composer Tests (`test_composer.py`)

Tests the LangGraph workflow composition and orchestration:

| Test | Description | Status |
|------|-------------|--------|
| `test_composer_imports` | Verifies composer modules can be imported | ✓ |
| `test_composer_service_exists` | Confirms ComposerService class is accessible | ✓ |
| `test_server_interface_protocol` | Validates ServerInterface protocol definition | ✓ |
| `test_workflow_builder_factory` | Tests workflow builder factory (IDE/Dialog) | ✓ |
| `test_composer_with_mocked_server` | Tests composer with mocked server services | ✓ |
| `test_workflow_caching` | Validates workflow caching functionality | ✓ |
| `test_composer_models` | Verifies composer model classes are accessible | ✓ |

### Runner Tests (`test_runner.py`)

Tests the pipeline factory and model execution:

| Test | Description | Status |
|------|-------------|--------|
| `test_runner_imports` | Verifies runner modules can be imported | ✓ |
| `test_pipeline_factory_exists` | Confirms pipeline factory is accessible | ✓ |
| `test_local_pipeline_cache_exists` | Validates pipeline cache functionality | ✓ |
| `test_hardware_manager_exists` | Tests GPU hardware manager | ✓ |
| `test_pipeline_types` | Verifies pipeline type definitions | ✓ |
| `test_runner_models` | Confirms runner model classes accessible | ✓ |
| `test_pipeline_cache_stats` | Tests cache statistics reporting | ✓ |
| `test_gpu_detection` | Validates GPU detection (if available) | ✓ |
| `test_pipeline_factory_methods` | Verifies factory method signatures | ✓ |
| `test_pipeline_cache_methods` | Validates cache method availability | ✓ |

## Cross-Component Integration

### Server-Composer Integration

The ServerInterface protocol enables clean separation between Server and Composer:

```
┌─────────────┐          ┌─────────────┐
│   Server    │─────────>│   Composer  │
│  Services   │<─────────│   Workflow  │
│             │   Protocol│             │
└─────────────┘           └─────────────┘
```

**ServerInterface Protocol Methods:**
- `user_config`: UserConfigService for configuration retrieval
- `conversation`: ConversationService for conversation management
- `message`: MessageService for message storage/retrieval
- `memory`: MemoryService for vector search and storage
- `summary`: SummaryService for conversation summaries
- `model_profile`: ModelProfileService for model configuration
- `dynamic_tool`: DynamicToolService for tool management

### Composer-Runner Integration

Composer communicates with Runner via gRPC for pipeline management:

```
Composer ──gRPC──> Runner
                    ├─ CreatePipeline
                    ├─ ExecutePipeline
                    ├─ GenerateEmbeddings
                    ├─ GetCacheStats
                    └─ EvictPipeline
```

## End-to-End Tests

### E2E Test Flow (`test_e2e_flow.py`)

End-to-end tests verify the complete request flow:

1. **Full Request Flow** - Test complete request through all layers
2. **Chat Completion Flow** - Test OpenAI-compatible chat completions
3. **Conversation Lifecycle** - Test conversation creation and management
4. **Message Flow** - Test message storage and retrieval
5. **Database Persistence** - Test data persistence across layers

**Note:** E2E tests require running services (database, server) and may need to be run separately.

## Test Execution

### Running Component Tests

```bash
# Run composer tests
cd test/integration
pytest test_composer.py -v

# Run runner tests
pytest test_runner.py -v

# Run all integration tests
pytest test/integration -v

# Run with specific markers
pytest test/integration -m "composer"
pytest test/integration -m "runner"
pytest test/integration -m "database"
```

### Test Environment Setup

Tests require the following environment setup:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Set test mode
export TEST_MODE=true

# Configure test user
export TEST_USER_ID="test-user-$(date +%s)"
```

### Running All Tests

```bash
make test                    # Run all tests (inference + UI)
pytest test/integration/     # Run integration tests
pytest test/unit/           # Run unit tests
```

## Test Fixtures and Helpers

### UserConfig Factory

`test_composer.py` provides a `create_test_user_config()` helper that creates a minimal valid UserConfig for testing:

```python
def create_test_user_config() -> "UserConfig":
    """Create a minimal UserConfig for testing."""
    return UserConfig(
        user_id="test-user",
        summarization=SummarizationConfig(...),
        memory=MemoryConfig(...),
        # ... other required configs
    )
```

### Mock Server Interface

Tests can use a mocked ServerInterface for isolated testing:

```python
class MockServerInterface:
    async def get_user_config(self, user_id: str) -> UserConfig:
        return create_test_user_config()
```

## Best Practices

1. **Test isolation**: Each test should be independent and reproducible
2. **Mock external dependencies**: Use mocks for server/database in component tests
3. **Verify behavior, not implementation**: Test interfaces, not internal details
4. **Async support**: All async tests use pytest-asyncio with `@pytest.mark.asyncio`
5. **Clean state**: Tests should leave no persistent state (use unique IDs)

## Known Limitations

1. **E2E tests require services**: Full E2E tests need database and server running
2. **GPU tests are conditional**: GPU-specific tests only run if GPU is available
3. **Database tests need connection**: Database tests require PostgreSQL connection

## Future Enhancements

1. Add database migration tests
2. Add gRPC contract tests
3. Add performance benchmarks
4. Add chaos testing for failure scenarios
5. Add distributed tracing integration