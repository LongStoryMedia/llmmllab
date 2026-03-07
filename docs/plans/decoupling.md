# Microservices Refactor: Service-Local gRPC Code Generation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make runner, composer, and server into independent microservices communicating ONLY via gRPC by implementing service-local gRPC code generation and removing the shared `grpcs/` directory tight coupling.

**Architecture:** Each service (server, composer, runner) will generate gRPC code to its own `gen/python/` directory with service-specific package structure (`server.v1`, `composer.v1`, `runner.v1`). Services will communicate via gRPC using these local packages, eliminating the shared `grpcs/` directory that currently creates circular dependencies.

**Tech Stack:** gRPC Python, Protocol Buffers 7.34.0, Python 3.12, Docker multi-stage builds

---

## Current State Analysis

### Problem: Tight Coupling via Shared grpcs/

The `grpcs/` directory contains generated gRPC code used by all services:

| File | Purpose | Current Users |
|------|---------|---------------|
| `grpcs/__init__.py` | Package exports | All services |
| `grpcs/server_composer_pb2.py` | Composer service messages | server/, composer/ |
| `grpcs/server_composer_pb2_grpc.py` | Composer service stubs | server/, composer/ |
| `grpcs/composer_runner_pb2.py` | Runner service messages | server/, composer/ |
| `grpcs/composer_runner_pb2_grpc.py` | Runner service stubs | server/, composer/ |

### Services Currently Using grpcs/

| Service | File | Import Pattern | Issue |
|---------|------|----------------|-------|
| **server** | `__init__.py` | `from grpcs import server_composer_pb2_grpc` | Lines 114, 129, 250, 262, 300, 318, 335 |
| **server** | `grpc_client.py` | `from grpcs import ...` | Lines 14-19 |
| **composer** | `grpc/server.py` | `from grpcs import ...` | Lines 14-19, 93 |
| **composer** | `grpc/server.py` (line 93) | `from runner import pipeline_factory` | Direct import bypasses gRPC |

### Services Already Correct

| Service | File | Status |
|---------|------|--------|
| **runner** | `server/grpc.py` | Already uses `from runner.v1 import ...` (lines 24-27) |
| **runner** | `gen/python/setup.py` | Already exists with correct package structure |

### Proto Files

| Proto | Package | Service | Current Gen Location |
|-------|---------|---------|---------------------|
| `proto/runner/v1/composer_runner.proto` | `composer_runner.v1` | RunnerService | grpcs/ |
| `proto/composer/v1/server_composer.proto` | `server_composer.v1` | ComposerService | grpcs/ |
| `proto/common/timestamp.proto` | `google.protobuf` | Timestamp | grpcs/ |

---

## Implementation Plan

### Phase 1: Service-Local gRPC Code Generation

#### Step 1.1: Update build.sh for service-local generation

**File:** `build.sh` (lines 86-114)

**Current behavior:** Generates to shared `gen/python/` location

**Required changes:**

```bash
# Generate service-local gRPC code

# Generate runner gRPC (already correct structure)
mkdir -p runner/gen/python/runner/v1
python -m grpc_tools.protoc \
    -I "proto" \
    --python_out="runner/gen/python/runner/v1" \
    --grpc_python_out="runner/gen/python/runner/v1" \
    "proto/runner/v1/composer_runner.proto" \
    "proto/common/timestamp.proto" 2>&1 || true

# Generate composer gRPC (NEW - service-local)
mkdir -p composer/gen/python/composer/v1
python -m grpc_tools.protoc \
    -I "proto" \
    --python_out="composer/gen/python/composer/v1" \
    --grpc_python_out="composer/gen/python/composer/v1" \
    "proto/composer/v1/server_composer.proto" \
    "proto/common/timestamp.proto" 2>&1 || true

# Generate server gRPC (NEW - service-local, if server needs it)
mkdir -p server/gen/python/server/v1
# Only generate if server needs to act as a gRPC client to other services
```

#### Step 1.2: Create composer/gen/python/setup.py

**File:** `composer/gen/python/setup.py` (NEW FILE)

```python
"""Setup script for composer gRPC generated code."""

from setuptools import setup, find_packages

setup(
    name="composer-grpc",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "grpcio>=1.78.0",
        "protobuf>=7.34.0",
    ],
    python_requires=">=3.12",
)
```

#### Step 1.3: Create composer/gen/python/composer/__init__.py

**File:** `composer/gen/python/composer/__init__.py` (NEW FILE)

```python
"""Composer gRPC generated modules."""

from composer.v1 import (
    server_composer_pb2,
    server_composer_pb2_grpc,
)

__all__ = [
    "server_composer_pb2",
    "server_composer_pb2_grpc",
]
```

#### Step 1.4: Create composer/gen/python/composer/v1/__init__.py

**File:** `composer/gen/python/composer/v1/__init__.py` (NEW FILE)

```python
"""Composer v1 gRPC generated modules."""

from composer.v1.server_composer_pb2 import (
    WorkflowHandle,
    WorkflowState,
    ComposeWorkflowRequest,
    ExecuteWorkflowRequest,
    MessageContent,
    Message,
    ToolCall,
    Document,
    ChatDelta,
    WorkflowComplete,
    TodoItem,
    ChatResponseDelta,
    ChatResponseComplete,
    ChatResponse,
    CreateInitialStateRequest,
    ClearWorkflowCacheRequest,
    ClearWorkflowCacheResponse,
)
from composer.v1.server_composer_pb2_grpc import (
    ComposerServiceStub,
    ComposerServiceServicer,
    add_ComposerServiceServicer_to_server,
)
from google.protobuf.timestamp_pb2 import Timestamp

__all__ = [
    "WorkflowHandle",
    "WorkflowState",
    "ComposeWorkflowRequest",
    "ExecuteWorkflowRequest",
    "MessageContent",
    "Message",
    "ToolCall",
    "Document",
    "ChatDelta",
    "WorkflowComplete",
    "TodoItem",
    "ChatResponseDelta",
    "ChatResponseComplete",
    "ChatResponse",
    "CreateInitialStateRequest",
    "ClearWorkflowCacheRequest",
    "ClearWorkflowCacheResponse",
    "ComposerServiceStub",
    "ComposerServiceServicer",
    "add_ComposerServiceServicer_to_server",
    "Timestamp",
]
```

---

### Phase 2: Update Server to Use Service-Local gRPC

#### Step 2.1: Update server/__init__.py

**File:** `server/__init__.py`

**Change 1:** Replace grpcs imports with local package (line 114)

```python
# Before:
from grpcs import server_composer_pb2_grpc
self._stub = server_composer_pb2_grpc.ComposerServiceStub(self.channel)

# After:
from server.v1 import server_composer_pb2_grpc
self._stub = server_composer_pb2_grpc.ComposerServiceStub(self.channel)
```

**Change 2:** Update import for messages (line 129)

```python
# Before:
from grpcs import server_composer_pb2

# After:
from server.v1 import server_composer_pb2
```

**Change 3:** Update Runner gRPC imports (lines 250, 262, 300, 318, 335)

```python
# Before:
from grpcs import composer_runner_pb2_grpc
from grpcs import composer_runner_pb2

# After:
from runner.v1 import composer_runner_pb2_grpc
from runner.v1 import composer_runner_pb2
```

#### Step 2.2: Update server/grpc_client.py

**File:** `server/grpc_client.py` (if exists)

**Change:** Replace grpcs imports with service-local imports

```python
# Before:
from grpcs import (
    server_composer_pb2,
    server_composer_pb2_grpc,
    composer_runner_pb2,
    composer_runner_pb2_grpc,
)

# After:
from server.v1 import (
    server_composer_pb2,
    server_composer_pb2_grpc,
)
from runner.v1 import (
    composer_runner_pb2,
    composer_runner_pb2_grpc,
)
```

---

### Phase 3: Update Composer to Use Service-Local gRPC

#### Step 3.1: Update composer/grpc/server.py

**File:** `composer/grpc/server.py`

**Change 1:** Replace grpcs imports (lines 14-19)

```python
# Before:
from grpcs import (
    server_composer_pb2,
    server_composer_pb2_grpc,
    composer_runner_pb2,
    composer_runner_pb2_grpc,
)

# After:
from composer.v1 import (
    server_composer_pb2,
    server_composer_pb2_grpc,
)
from runner.v1 import (
    composer_runner_pb2,
    composer_runner_pb2_grpc,
)
```

**Change 2:** Remove direct runner import (line 93) - use gRPC instead

```python
# Before:
from runner import pipeline_factory

# After: (remove this line entirely - access via gRPC)
```

**Change 3:** Add gRPC client for runner communication

```python
# Add after imports:
from runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc

class RunnerClient:
    """gRPC client for Runner service communication."""
    def __init__(self, target: str = "localhost:50052"):
        self.target = target
        self._channel = None
        self._stub = None
    
    @property
    def stub(self):
        if self._stub is None:
            self._stub = composer_runner_pb2_grpc.RunnerServiceStub(self.channel)
        return self._stub
    
    @property
    def channel(self):
        if self._channel is None:
            import grpc as grpcio
            self._channel = grpcio.aio.insecure_channel(self.target)
        return self._channel
    
    async def create_pipeline(self, profile, priority="normal", grammar_type="auto", metadata=None):
        request = composer_runner_pb2.CreatePipelineRequest(
            profile=profile,
            priority=priority,
            grammar_type=grammar_type,
            metadata=metadata or {}
        )
        return await self._stub.CreatePipeline(request)
    
    async def execute_pipeline(self, pipeline_id, input_data, stream_output=True):
        request = composer_runner_pb2.ExecutePipelineRequest(
            pipeline_id=pipeline_id,
            input_data=input_data,
            stream_output=stream_output
        )
        return self._stub.ExecutePipeline(request)
    
    async def close(self):
        if self._channel:
            await self._channel.close()
            self._channel = None
```

#### Step 3.2: Update composer/__init__.py

**File:** `composer/__init__.py`

**Change:** Update imports to use local packages

```python
# Before:
from grpcs import server_composer_pb2, server_composer_pb2_grpc
from grpcs import composer_runner_pb2, composer_runner_pb2_grpc

# After:
from composer.v1 import server_composer_pb2, server_composer_pb2_grpc
from runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc
```

---

### Phase 4: Update Dockerfiles

#### Step 4.1: Update server/k8s/Dockerfile

**File:** `server/k8s/Dockerfile`

**Change 1:** Add gRPC code generation step

```dockerfile
# After copying requirements and installing dependencies:
# Generate gRPC code for server
RUN pip install --no-cache-dir grpcio-tools && \
    python -m grpc_tools.protoc \
        -I "proto" \
        --python_out="gen/python/server/v1" \
        --grpc_python_out="gen/python/server/v1" \
        "proto/composer/v1/server_composer.proto" \
        "proto/common/timestamp.proto" && \
    pip uninstall -y grpcio-tools

# Install server package in development mode
RUN pip install --no-cache-dir -e .
```

**Change 2:** Update COPY to include generated code

```dockerfile
# Before:
COPY server/ ./server/
COPY schemas/ ./schemas/

# After:
COPY server/ ./server/
COPY schemas/ ./schemas/
COPY gen/ ./gen/
```

#### Step 4.2: Update composer/k8s/Dockerfile

**File:** `composer/k8s/Dockerfile`

**Change 1:** Remove grpcs directory copy (line 35)

```dockerfile
# Before:
COPY composer/ ./composer/
COPY schemas/ ./schemas/
COPY grpcs/ ./grpcs/

# After:
COPY composer/ ./composer/
COPY schemas/ ./schemas/
```

**Change 2:** Add gRPC code generation

```dockerfile
# After pip install requirements:
# Generate gRPC code
RUN pip install --no-cache-dir grpcio-tools && \
    python -m grpc_tools.protoc \
        -I "proto" \
        --python_out="gen/python/composer/v1" \
        --grpc_python_out="gen/python/composer/v1" \
        "proto/composer/v1/server_composer.proto" \
        "proto/common/timestamp.proto" && \
    pip uninstall -y grpcio-tools

# Install composer package in development mode
RUN pip install --no-cache-dir -e .
```

#### Step 4.3: Update runner/k8s/Dockerfile

**File:** `runner/k8s/Dockerfile`

**Status:** Already correct - uses `from runner.v1 import ...` and installs runner package with `pip install -e /app/runner`

---

### Phase 5: Update CLAUDE.md Files

#### Step 5.1: Update server/CLAUDE.md

**File:** `server/CLAUDE.md`

**Add section on gRPC clients:**

```markdown
### gRPC Clients

The server provides gRPC client interfaces for inter-service communication:

**ComposerClient:**
- `compose_workflow()`: Create a workflow via Composer service
- `execute_workflow()`: Execute a workflow with streaming
- `create_initial_state()`: Create initial workflow state
- `clear_workflow_cache()`: Clear cached workflows
- `health_check()`: Check Composer service health

**RunnerClient:**
- `execute_pipeline()`: Execute a pipeline via Runner service
- `get_model_info()`: Get model information
- `generate_embeddings()`: Generate embeddings for texts
- `get_cache_stats()`: Get pipeline cache statistics
- `evict_pipeline()`: Evict a pipeline from cache
- `health_check()`: Check Runner service health

**Protocol Interfaces** (for type-safe DI):
- `server.ComposerClientProtocol`
- `server.RunnerClientProtocol`

Use `server.get_composer_client()` and `server.get_runner_client()` to get singleton instances.
```

#### Step 5.2: Update composer/CLAUDE.md

**File:** `composer/CLAUDE.md`

**Update gRPC Client Integration section:**

```markdown
### gRPC Client Integration

Composer uses gRPC to communicate with the Runner service for pipeline management:

```python
from runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc

# Create pipeline
stub = composer_runner_pb2_grpc.RunnerServiceStub(channel)
response = stub.CreatePipeline(request)

# Execute pipeline
response = stub.ExecutePipeline(request)
```

### Service-Local gRPC Generation

Composer generates gRPC code to `composer/gen/python/composer/v1/` with package structure:
- `composer.v1` package for ComposerService
- `runner.v1` package for RunnerService

This enables independent deployment of composer as a microservice.
```

---

### Phase 6: Testing and Verification

#### Step 6.1: Test gRPC Code Generation

```bash
# Clean and regenerate
make clean
./build.sh

# Verify generated files exist
ls -la composer/gen/python/composer/v1/
ls -la runner/gen/python/runner/v1/
ls -la server/gen/python/server/v1/
```

#### Step 6.2: Test Imports

```bash
# Test composer imports
python -c "from composer.v1 import server_composer_pb2, server_composer_pb2_grpc; print('OK')"

# Test runner imports
python -c "from runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc; print('OK')"

# Test server imports
python -c "from server.v1 import server_composer_pb2, server_composer_pb2_grpc; print('OK')"
```

#### Step 6.3: Verify No grpcs Imports Remain

```bash
# Check for any remaining grpcs imports
grep -r "from grpcs import" --include="*.py" server/ composer/
grep -r "import grpcs" --include="*.py" server/ composer/
```

Expected: No results

#### Step 6.4: Integration Test

```bash
# Start services
make start

# Test Composer->Runner gRPC
# (Add integration test to verify gRPC communication works)
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `composer/gen/python/setup.py` | Package setup for composer gRPC |
| `composer/gen/python/composer/__init__.py` | Package init for composer.v1 |
| `composer/gen/python/composer/v1/__init__.py` | Exports for composer v1 gRPC |
| `server/gen/python/server/v1/__init__.py` | Exports for server v1 gRPC (if needed) |

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `build.sh` | 86-114 | Update to service-local generation |
| `server/__init__.py` | 114, 129, 250, 262, 300, 318, 335 | Replace grpcs imports |
| `server/grpc_client.py` | 14-19 | Replace grpcs imports |
| `composer/grpc/server.py` | 14-19, 93 | Replace grpcs imports, remove direct runner import |
| `composer/__init__.py` | Various | Update imports to local packages |
| `server/k8s/Dockerfile` | 29-30, 36 | Remove grpcs copy, add generation |
| `composer/k8s/Dockerfile` | 35 | Remove grpcs copy |

## Files to Keep (No Changes)

| File | Reason |
|------|--------|
| `runner/server/grpc.py` | Already uses correct imports |
| `runner/gen/python/setup.py` | Already correct |
| `runner/gen/python/runner/v1/__init__.py` | Already correct |
| `proto/runner/v1/composer_runner.proto` | Source of truth |
| `proto/composer/v1/server_composer.proto` | Source of truth |

---

## Verification Steps

1. **Clean build:** `make clean && ./build.sh` - generates all gRPC code
2. **No grpcs imports:** `grep -r "from grpcs" --include="*.py"` - should show zero results
3. **Type checking:** `pyright` on server/composer - no import errors
4. **Unit tests:** `pytest server/ composer/ runner/` - all tests pass
5. **Integration test:** Start services and verify gRPC communication

---

## Rollback Plan

If issues occur:
1. Revert changes to `build.sh` to restore shared generation
2. Restore `grpcs/` directory from git
3. Revert changes to import statements

## Notes

1. The `grpcs/` directory can be deprecated after all services are updated
2. Consider adding a `gen/python` directory to `.gitignore` to prevent committing generated code
3. The multi-stage Docker builds ensure gRPC code generation happens in the final image without build tools


## PLAN

Microservices Refactor Plan                                                                                                                                                                                      
                                                                                                                                                                                                                  
 For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.                                                                                                             
                                                                                                                                                                                                                  
 Goal: Make runner, composer, and server into independent microservices communicating only via gRPC                                                                                                               
                                                                                                                                                                                                                  
 Architecture: Proto files in proto/v1/, generated code service-local, gRPC-only communication

 Tech Stack: Python, gRPC, Protocol Buffers, Docker, Kubernetes

 ---
 Phase 1: Proto File Reorganization

 Task 1.1: Verify proto directory structure

 Files:
 - Verify: proto/v1/composer_runner.proto
 - Verify: proto/v1/server_composer.proto
 - Verify: proto/common/timestamp.proto

 Step 1: Check proto file contents

 ls -la proto/v1/
 cat proto/v1/composer_runner.proto
 cat proto/v1/server_composer.proto

 Expected: Both proto files have correct package names (composer_runner.v1, server_composer.v1)

 ---
 Phase 2: Update build.sh for Service-Local Generation

 Task 2.1: Update build.sh to generate service-local code

 Files:
 - Modify: build.sh

 Current State:
 - Generates to gen/python/ (shared location)

 Target State:
 - Generate to service-local directories:
   - runner/gen/python/runner/v1/
   - composer/gen/python/composer/v1/
   - server/gen/python/server_composer/v1/

 Step 1: Read current build.sh

 cat build.sh

 Step 2: Update build.sh

 The script should:
 1. Create service-local directories
 2. Generate proto code for each service
 3. Copy common types to each service's gen directory

 Example structure:
 # Generate for runner (uses composer_runner.v1)
 mkdir -p runner/gen/python/runner/v1
 python -m grpc_tools.protoc -Iproto --python_out=runner/gen/python/runner/v1 --grpc_python_out=runner/gen/python/runner/v1 proto/v1/composer_runner.proto

 # Generate for composer (uses both composer_runner.v1 and server_composer.v1)
 mkdir -p composer/gen/python/composer/v1
 # ... generate commands

 # Generate for server (uses server_composer.v1)
 mkdir -p server/gen/python/server_composer/v1
 # ... generate commands

 ---
 Phase 3: Update gRPC Client/Server Implementations

 Task 3.1: Update server/init.py

 Files:
 - Modify: server/__init__.py

 Current State (lines 114-115):
 from grpcs import server_composer_pb2_grpc
 from grpcs import server_composer_pb2

 Target State:
 # Import from locally generated code (or use gRPC client)
 # After build: server/gen/python/server_composer/v1/
 from server_composer.v1 import server_composer_pb2_grpc, server_composer_pb2

 Step 1: Read current imports

 head -150 server/__init__.py | tail -50

 Step 2: Update imports

 Update all imports from grpcs to use locally generated code:
 - server_composer_pb2_grpc → server_composer.v1.server_composer_pb2_grpc
 - server_composer_pb2 → server_composer.v1.server_composer_pb2

 Also update composer_runner imports for RunnerClient.

 Task 3.2: Update composer/grpc/server.py

 Files:
 - Modify: composer/grpc/server.py

 Current State (lines 14-19):
 from grpcs import server_composer_pb2_grpc
 from grpcs import server_composer_pb2
 from grpcs import composer_runner_pb2_grpc
 from grpcs import composer_runner_pb2

 Target State:
 # Import from locally generated code
 from composer.v1 import server_composer_pb2_grpc, server_composer_pb2
 from composer.v1 import composer_runner_pb2_grpc, composer_runner_pb2

 Step 1: Read current imports

 head -30 composer/grpc/server.py

 Step 2: Update imports

 Update all grpcs imports to use locally generated code.

 Task 3.3: Update runner/server/grpc.py

 Files:
 - Verify: runner/server/grpc.py (already correct)

 Current State (lines 24-27):
 from runner.v1 import (
     composer_runner_pb2,
     composer_runner_pb2_grpc,
 )

 Status: Already follows correct pattern - no changes needed.

 ---
 Phase 4: Update Dockerfiles

 Task 4.1: Update server/k8s/Dockerfile

 Files:
 - Modify: server/k8s/Dockerfile

 Current State:
 - Copies grpcs/ directory

 Target State:
 - Generate or copy locally generated gRPC code
 - Remove grpcs/ copy

 Step 1: Read current Dockerfile

 cat server/k8s/Dockerfile

 Step 2: Update Dockerfile

 - Remove COPY grpcs/ ./grpcs/
 - Add generation or copy of locally generated code
 - Update PYTHONPATH if needed

 Task 4.2: Update composer/k8s/Dockerfile

 Files:
 - Modify: composer/k8s/Dockerfile

 Current State:
 - Copies grpcs/ directory

 Target State:
 - Generate or copy locally generated gRPC code
 - Remove grpcs/ copy

 Step 1: Read current Dockerfile

 cat composer/k8s/Dockerfile

 Step 2: Update Dockerfile

 - Remove COPY grpcs/ ./grpcs/
 - Add generation or copy of locally generated code

 Task 4.3: Update runner/k8s/Dockerfile

 Files:
 - Modify: runner/k8s/Dockerfile

 Current State:
 - Already correct - uses local generated code

 Status: May need minor update to ensure generated code path is correct.

 ---
 Phase 5: Update CLAUDE.md Files

 Task 5.1: Update server/CLAUDE.md

 Files:
 - Modify: server/CLAUDE.md

 Current State:
 - Documents gRPC clients for Composer and Runner
 - References grpcs/ directory

 Target State:
 - Update to reflect service-local gRPC code
 - Document gRPC client usage without grpcs/ reference

 Step 1: Read current CLAUDE.md

 cat server/CLAUDE.md

 Step 2: Update

 - Update "gRPC Clients" section to reference locally generated code
 - Remove references to grpcs/ directory

 Task 5.2: Update composer/CLAUDE.md

 Files:
 - Modify: composer/CLAUDE.md

 Current State:
 - Documents gRPC client integration
 - References grpcs/ package

 Target State:
 - Update to reflect service-local gRPC code
 - Document gRPC server implementation

 Step 1: Read current CLAUDE.md

 cat composer/CLAUDE.md

 Step 2: Update

 - Update "gRPC Client Integration" section
 - Add "gRPC Server" section documenting ComposerService

 Task 5.3: Update runner/CLAUDE.md

 Files:
 - Modify: runner/CLAUDE.md

 Current State:
 - Already correct - references local generated code

 Status: May need minor update to confirm runner.v1 package structure.

 ---
 Phase 6: Verification and Testing

 Task 6.1: Verify proto generation

 Step 1: Run build.sh

 ./build.sh

 Step 2: Verify generated files

 ls -la runner/gen/python/runner/v1/
 ls -la composer/gen/python/composer/v1/
 ls -la server/gen/python/server_composer/v1/

 Expected: All services have their gRPC code generated locally.

 Task 6.2: Verify imports work

 Step 1: Test server imports

 cd server && python -c "from server_composer.v1 import server_composer_pb2_grpc; print('OK')"

 Step 2: Test composer imports

 cd composer && python -c "from composer.v1 import server_composer_pb2_grpc; from composer.v1 import composer_runner_pb2_grpc; print('OK')"

 Step 3: Test runner imports

 cd runner && python -c "from runner.v1 import composer_runner_pb2_grpc; print('OK')"

 Task 6.3: Verify no grpcs imports remain

 Step 1: Search for grpcs imports

 grep -r "from grpcs" server/ composer/ runner/
 grep -r "import grpcs" server/ composer/ runner/

 Expected: No results (or only in deprecated code marked for removal).

 Task 6.4: Test gRPC communication

 Step 1: Start services

 # Start Runner gRPC server
 make inference-dev  # or run runner server directly

 # Start Composer gRPC server
 # (new command needed or run composer/grpc/server.py directly)

 # Start Server
 make start-ui  # or run server directly

 Step 2: Test gRPC calls

 # Test Composer gRPC call
 python -c "
 import grpc
 from server_composer.v1 import server_composer_pb2, server_composer_pb2_grpc
 # ... test code
 "

 # Test Runner gRPC call
 python -c "
 import grpc
 from runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc
 # ... test code
 "

 Task 6.5: Verify Docker builds

 Step 1: Build Docker images

 docker build -f server/k8s/Dockerfile -t server:test .
 docker build -f composer/k8s/Dockerfile -t composer:test .
 docker build -f runner/k8s/Dockerfile -t runner:test .

 Expected: All builds succeed without grpcs errors.

 ---
 Files Summary

 ┌─────────────────────────┬────────┬───────────────────────────────────┐
 │          File           │ Action │              Reason               │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ build.sh                │ Modify │ Generate service-local gRPC code  │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ server/__init__.py      │ Modify │ Update gRPC client imports        │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ composer/grpc/server.py │ Modify │ Update gRPC server/client imports │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ server/k8s/Dockerfile   │ Modify │ Remove grpcs copy, add local code │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ composer/k8s/Dockerfile │ Modify │ Remove grpcs copy, add local code │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ runner/k8s/Dockerfile   │ Verify │ Confirm correct structure         │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ server/CLAUDE.md        │ Update │ Document new gRPC pattern         │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ composer/CLAUDE.md      │ Update │ Document new gRPC pattern         │
 ├─────────────────────────┼────────┼───────────────────────────────────┤
 │ runner/CLAUDE.md        │ Verify │ Confirm correct                   │
 └─────────────────────────┴────────┴───────────────────────────────────┘

 ---
 Notes

 - The grpcs/ directory can be removed after all services are updated and tested
 - Consider adding a deprecation warning in grpcs/__init__.py before removal
 - Ensure CI/CD pipeline is updated to run proto generation before builds
 - Consider using python -m pip install -e runner/gen/python/ pattern for local development
