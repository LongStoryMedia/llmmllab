# gRPC Refactoring Plan

## Overview

This document outlines the refactoring plan for the gRPC architecture to address:
- Runtime path manipulation issues
- Missing protocol interfaces
- Package naming confusion
- Generated code structure unification

## Current Issues

### 1. Runtime `sys.path.insert()` in `runner/server/grpc.py`

**Problem**: Lines 32 and 45 use `sys.path.insert(0, ...)` to add generated code to the Python path at runtime.

```python
# Current problematic code
sys.path.insert(0, gen_python_dir)
sys.path.insert(0, runner_pkg_dir)
```

**Issues**:
- Fragile and unpythonic
- Creates import order dependencies
- Conflicts with Python's import system
- Makes testing difficult
- Breaks type checking

### 2. Package Naming Confusion

| Location | Package | Issue |
|----------|---------|-------|
| `grpcs/` | `composer_runner`, `server_composer` | Old flat structure |
| `proto/` | `composer_runner.v1`, `server_composer.v1` | New versioned structure |
| `gen/python/` | - | Generated code at root |
| `runner/gen/python/` | - | Duplicate generated code |

### 3. Missing Protocol Interfaces

No Protocol classes exist to define clean service contracts. This causes:
- Tight coupling between components
- Difficulty in testing (can't easily mock)
- No clear API boundaries

### 4. Generated Code Structure

Generated code is scattered across:
- `gen/python/runner/v1/`
- `gen/python/composer/v1/`
- `runner/gen/python/runner/v1/`
- `runner/gen/python/common/`
- `composer/gen/python/server_composer_v1/`

## Proposed Architecture

### Directory Structure

```
proto/
├── common/                   # Shared proto definitions
│   └── timestamp.proto
├── runner/                   # Runner service definitions
│   └── v1/
│       ├── composer_runner.proto
│       └── __init__.py
├── composer/                 # Composer service definitions
│   └── v1/
│       ├── server_composer.proto
│       └── __init__.py
└── server/                   # Server-specific definitions
    └── v1/
        ├── message.proto
        └── __init__.py

gen/proto/                    # Generated code (gitignored)
├── runner/
│   └── v1/
│       ├── composer_runner_pb2.py
│       ├── composer_runner_pb2_grpc.py
│       └── __init__.py
└── composer/
    └── v1/
        ├── server_composer_pb2.py
        ├── server_composer_pb2_grpc.py
        └── __init__.py

runner/
├── adapters/                 # gRPC adapters with Protocol interfaces
│   ├── __init__.py
│   ├── protocol.py           # Protocol definitions
│   └── runner.py             # RunnerService adapter
└── server/
    └── grpc.py               # gRPC server (uses adapters)

composer/
├── adapters/                 # gRPC adapters with Protocol interfaces
│   ├── __init__.py
│   ├── protocol.py           # Protocol definitions
│   ├── runner.py             # RunnerService adapter
│   └── composer.py           # ComposerService adapter
└── grpc/                     # gRPC server (uses adapters)

server/
├── adapters/                 # gRPC client adapters with Protocol interfaces
│   ├── __init__.py
│   ├── protocol.py           # Protocol definitions
│   ├── runner.py             # RunnerClient adapter
│   └── composer.py           # ComposerClient adapter
└── grpc_client.py            # gRPC client (uses adapters)
```

### Protocol Interfaces

Each service will have a `protocol.py` defining clean Python interfaces:

```python
# runner/adapters/protocol.py
from typing import Protocol, AsyncIterator, Optional
from dataclasses import dataclass

@dataclass
class PipelineHandle:
    pipeline_id: str
    model_name: str
    is_cached: bool

@dataclass
class CacheStats:
    total_pipelines: int
    cached_pipelines: int
    active_pipelines: int
    total_memory_bytes: int
    available_memory_bytes: int
    cache_hits: int
    cache_misses: int
    hit_rate: float

class RunnerService(Protocol):
    """Protocol for Runner service operations."""

    async def create_pipeline(
        self,
        model_name: str,
        provider: str,
        task_type: str,
        priority: Optional[str] = None,
    ) -> PipelineHandle: ...

    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: bytes,
        stream_output: bool = True,
    ) -> AsyncIterator[bytes]: ...

    async def generate_embeddings(
        self,
        texts: list[str],
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> list[list[float]]: ...

    async def get_cache_stats(
        self,
        pipeline_type: Optional[str] = None,
    ) -> CacheStats: ...

    async def evict_pipeline(
        self,
        pipeline_id: str,
        force: bool = False,
    ) -> bool: ...


# composer/adapters/protocol.py
from typing import Protocol, AsyncIterator
from dataclasses import dataclass

@dataclass
class WorkflowHandle:
    workflow_id: str
    created_at: int

@dataclass
class WorkflowState:
    user_id: str
    conversation_id: int
    workflow_type: str
    variables: dict[str, str]

class ComposerService(Protocol):
    """Protocol for Composer service operations."""

    async def compose_workflow(
        self,
        user_id: str,
        workflow_type: str,
        model_name: Optional[str] = None,
    ) -> WorkflowHandle: ...

    async def execute_workflow(
        self,
        workflow_id: str,
        initial_state: WorkflowState,
    ) -> AsyncIterator[bytes]: ...

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        workflow_type: str,
    ) -> WorkflowState: ...

    async def clear_workflow_cache(
        self,
        user_id: str,
    ) -> bool: ...
```

### Adapters

Adapters wrap gRPC clients/servicers and implement the Protocol interfaces:

```python
# runner/adapters/runner.py
from proto.runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc
from runner.adapters.protocol import RunnerService, PipelineHandle, CacheStats

class RunnerServiceGrpcAdapter(RunnerService):
    """Adapter implementing RunnerService protocol using gRPC."""

    def __init__(self, channel):
        self._stub = composer_runner_pb2_grpc.RunnerServiceStub(channel)

    async def create_pipeline(
        self,
        model_name: str,
        provider: str,
        task_type: str,
        priority: Optional[str] = None,
    ) -> PipelineHandle:
        request = composer_runner_pb2.CreatePipelineRequest(
            profile=composer_runner_pb2.ModelProfile(
                model_name=model_name,
                provider=provider,
                task_type=task_type,
            ),
            priority=priority or "NORMAL",
        )
        response = await self._stub.CreatePipeline(request)
        return PipelineHandle(
            pipeline_id=response.pipeline_id,
            model_name=response.model_name,
            is_cached=response.is_cached,
        )

    # ... other methods ...


# server/adapters/composer.py
from proto.composer.v1 import server_composer_pb2, server_composer_pb2_grpc
from server.adapters.protocol import ComposerService, WorkflowHandle, WorkflowState

class ComposerServiceGrpcAdapter(ComposerService):
    """Adapter implementing ComposerService protocol using gRPC."""

    def __init__(self, channel):
        self._stub = server_composer_pb2_grpc.ComposerServiceStub(channel)

    async def compose_workflow(
        self,
        user_id: str,
        workflow_type: str,
        model_name: Optional[str] = None,
    ) -> WorkflowHandle:
        request = server_composer_pb2.ComposeWorkflowRequest(
            user_id=user_id,
            workflow_type=workflow_type,
            model_name=model_name or "",
        )
        response = await self._stub.ComposeWorkflow(request)
        return WorkflowHandle(
            workflow_id=response.workflow_id,
            created_at=response.created_at.seconds,
        )

    # ... other methods ...
```

### gRPC Server

The gRPC server will use adapters internally:

```python
# runner/server/grpc.py
from runner.adapters.runner import RunnerServiceGrpcAdapter
from runner.adapters.protocol import RunnerService

class RunnerServicer(composer_runner_pb2_grpc.RunnerServiceServicer):
    """gRPC servicer for RunnerService."""

    def __init__(self, runner_service: RunnerService):
        self._runner_service = runner_service

    async def CreatePipeline(
        self,
        request: composer_runner_pb2.CreatePipelineRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.PipelineHandle:
        handle = await self._runner_service.create_pipeline(
            model_name=request.profile.model_name,
            provider=request.profile.provider,
            task_type=request.profile.task_type,
            priority=request.priority,
        )
        return composer_runner_pb2.PipelineHandle(
            pipeline_id=handle.pipeline_id,
            model_name=handle.model_name,
            is_cached=handle.is_cached,
        )
```

### Import Flow

```python
# Before (problematic):
import sys
sys.path.insert(0, "gen/python")
import runner.v1.composer_runner_pb2  # Runtime path manipulation

# After (clean):
from gen.proto.runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc
from runner.adapters.runner import RunnerServiceGrpcAdapter
```

## Implementation Steps

### Phase 1: Proto Structure Cleanup
1. Remove `grpcs/` directory
2. Consolidate to `proto/[service]/v1/` structure
3. Update all proto package declarations to use versioned names

### Phase 2: Generate Code to Unified Location
1. Create `gen/proto/` directory (gitignored)
2. Update protoc command to generate to `gen/proto/[service]/v1/`
3. Add generated `__init__.py` files with proper package structure

### Phase 3: Protocol Interfaces
1. Create `adapters/protocol.py` for each service
2. Define clean Protocol interfaces
3. Document service contracts

### Phase 4: Adapters
1. Create `adapters/[service].py` for each service
2. Implement Protocol interfaces
3. Add proper error handling and type conversions

### Phase 5: gRPC Server/Client Updates
1. Update `runner/server/grpc.py` to use adapters
2. Update `server/grpc_client.py` to use adapters
3. Update `composer/grpc/server.py` to use adapters
4. Remove all `sys.path.insert()` calls

### Phase 6: Documentation
1. Update `CLAUDE.md` files for each service
2. Document gRPC architecture
3. Add migration guide

## Benefits

1. **No Runtime Path Manipulation**: Static imports from `gen/proto/`
2. **Clean Separation**: Adapters provide Protocol-based decoupling
3. **Testability**: Easy to mock Protocol interfaces
4. **Type Safety**: Full type hints through adapters
5. **Maintainability**: Clear directory structure
6. **Versioning**: Versioned proto definitions

## Migration Checklist

- [ ] Phase 1: Proto Structure Cleanup
  - [ ] Remove `grpcs/` directory
  - [ ] Verify `proto/` structure is correct
  - [ ] Update all proto imports

- [ ] Phase 2: Generate Code
  - [ ] Create `gen/proto/` directory
  - [ ] Update `regenerate_models.sh` to generate to new location
  - [ ] Add `gen/proto/` to `.gitignore`
  - [ ] Test imports from new location

- [ ] Phase 3: Protocol Interfaces
  - [ ] Create `runner/adapters/protocol.py`
  - [ ] Create `composer/adapters/protocol.py`
  - [ ] Create `server/adapters/protocol.py`

- [ ] Phase 4: Adapters
  - [ ] Create `runner/adapters/runner.py`
  - [ ] Create `composer/adapters/composer.py`
  - [ ] Create `composer/adapters/runner.py`
  - [ ] Create `server/adapters/composer.py`
  - [ ] Create `server/adapters/runner.py`

- [ ] Phase 5: gRPC Updates
  - [ ] Update `runner/server/grpc.py`
  - [ ] Update `server/grpc_client.py`
  - [ ] Update `composer/grpc/server.py`
  - [ ] Remove `sys.path.insert()` calls

- [ ] Phase 6: Documentation
  - [ ] Update `runner/CLAUDE.md`
  - [ ] Update `composer/CLAUDE.md`
  - [ ] Create `docs/GRPC_ARCHITECTURE.md`

---

## Additional Updates (Post-Refactor)

### Authentication Interceptor

Added gRPC interceptors for authentication using the existing `JWTValidator` from `server.middleware.auth`:

**File**: `server/grpc_interceptors.py`

**Features**:
- `AuthenticationUnaryUnaryServerInterceptor`: Validates JWT tokens on unary-unary calls
- `AuthenticationUnaryStreamServerInterceptor`: Validates JWT tokens on unary-stream calls
- Uses existing `JWTValidator` with JWKS caching
- Extracts token from `authorization` metadata header with Bearer prefix support
- Returns `AuthContext` with user_id, claims, and validation status

**Client Interceptors**:
- `AuthenticationClientInterceptor`: Adds authentication token to outgoing unary-unary requests
- `AuthenticationStreamClientInterceptor`: Adds authentication token to streaming requests
- Token provider pattern for dynamic token retrieval

### Compression Interceptor

Added gRPC compression support for large messages:

**File**: `server/grpc_interceptors.py`

**Features**:
- `CompressionClientInterceptor`: Enables compression for unary-unary and unary-stream calls
- `CompressionStreamClientInterceptor`: Enables compression for stream-unary and stream-stream calls
- Configurable compression algorithm (default: Gzip)
- Compressed call wrappers that modify metadata with compression header

**Usage**:
```python
from server.grpc_interceptors import CompressionClientInterceptor
from grpc import Compression

# Use Gzip compression
interceptor = CompressionClientInterceptor(Compression.Gzip)
```

### Logging and Metrics Interceptors

**File**: `server/grpc_interceptors.py`

**Features**:
- `LoggingUnaryUnaryServerInterceptor`: Logs unary-unary calls with timing and metadata
- `LoggingUnaryStreamServerInterceptor`: Logs streaming calls
- `MetricsUnaryUnaryServerInterceptor`: Collects metrics (total requests, success/failure counts, duration)
- Metrics are thread-safe using async locks

### Updated gRPC Client

**File**: `server/grpc_client.py`

**Features**:
- `ComposerGRPCClient`: gRPC client for Composer service communication
- `create_composer_channel()`: Helper function to create gRPC channels
- Methods: `compose_workflow()`, `execute_workflow()`, `create_initial_state()`, `clear_workflow_cache()`
- Proper async cleanup with `close()` method

### Server Initialization

**File**: `server/__init__.py`

**Changes**:
- Updated imports to use service-local generated gRPC code
- Removed runtime path manipulation
- gRPC clients now import from service-local packages:
  - `server_composer.v1` for ComposerService
  - `runner.v1` for RunnerService (via composer)

---

## Implementation Status

| Task | Status | Notes |
|------|--------|-------|
| Proto file reorganization | ✅ | `proto/v1/` structure with versioned packages |
| Service-local gRPC code generation | ✅ | `build.sh` generates to service-local directories |
| Authentication interceptor | ✅ | Uses existing `JWTValidator`, supports unary-unary and unary-stream |
| Compression interceptor | ✅ | Supports all 4 RPC types with Gzip |
| Logging interceptor | ✅ | Unary-unary and unary-stream logging |
| Metrics interceptor | ✅ | Thread-safe metrics collection |
| Protocol interfaces | ⚠️ | Partial - adapters exist but no separate protocol.py files |
| sys.path.insert() removal | ✅ | No runtime path manipulation |
| Client interceptor metadata | ⚠️ | Type hints need adjustment for internal grpc types |
| GRPC_REFACTORING_PLAN.md update | ✅ | Updated with interceptor documentation |