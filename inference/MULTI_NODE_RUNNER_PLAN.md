# Plan: Multi-Node Runner Architecture

## Status: Draft — pending user review

## TL;DR
Separate the runner from the server so it can be deployed independently on different k8s nodes with different hardware. The cleanest path: runner nodes expose an OpenAI-compatible HTTP API (they already do internally via llama-server), and the server's pipeline factory routes to them as "remote local" providers — same interface as OpenAI/Anthropic but pointing at internal runner services.

## Current State
- Monolith: FastAPI server + composer + runner + llama.cpp subprocesses all in one pod on `lsnode-3`
- Runner is tightly coupled: `pipeline_factory` is a module-level singleton imported directly by composer
- Local pipelines (llama.cpp) spawn subprocesses, wrap them with ChatOpenAI pointing at localhost
- Remote pipelines (OpenAI, Anthropic) are just raw ChatOpenAI/ChatAnthropic instances — no wrapper
- Model config is a YAML file loaded at startup with per-model GPU params
- Pipeline cache manages GPU memory, eviction, locking — all assumes local hardware

## Proposed Architecture

### Phase 1: Runner as Independent Service
1. Extract runner into its own deployable service that:
   - Loads its own `.models.yaml` subset (only models for its hardware)
   - Manages llama-server subprocesses as today
   - Exposes an OpenAI-compatible `/v1/chat/completions` endpoint (llama-server already does this)
   - Exposes a `/v1/models` endpoint listing available models
   - Exposes `/health` for k8s probes
2. Each runner deployment gets its own node selector, GPU config, resource limits
3. The existing server pod becomes "server-only" — no local llama-server processes

### Phase 2: Model Registry + Routing
1. Server discovers runner instances via k8s service discovery or a model registry
2. `pipeline_factory.get_pipeline()` becomes a routing decision:
   - Look up which runner(s) host the requested model
   - Return a `ChatOpenAI` client pointing at that runner's service URL
   - This is already the exact pattern used for remote OpenAI/Anthropic providers
3. Model config gains a `runner` or `endpoint` field (URL of the runner hosting it)

### Phase 3: Pipeline Cache Becomes Distributed
1. Runner nodes manage their own pipeline cache locally (as today)
2. Server doesn't need a pipeline cache — it just holds HTTP clients
3. Load/unload commands route to the appropriate runner

## Key Insight: The Hard Part is Already Solved
The llama-server subprocess already exposes an OpenAI-compatible API on localhost. The ChatLlamaCppPipeline already wraps it with ChatOpenAI. Moving from `http://127.0.0.1:{port}/v1` to `http://runner-embeddings.llmmll.svc.cluster.local:8000/v1` is a URL change, not an architecture change.

## Steps

### Phase 1: Runner Service Extraction
1. Create a lightweight FastAPI app for the runner service that:
   - Loads model config, starts llama-server processes
   - Proxies `/v1/chat/completions`, `/v1/embeddings` to the right llama-server subprocess
   - Reports available models via `/v1/models`
   - Handles health checks
   - Files: new `runner/app.py`, `runner/k8s/deployment.yaml`
   
2. Refactor `pipeline_factory` to support remote runners:
   - Add `endpoint` field to Model config
   - When `endpoint` is set, return a `ChatOpenAI(base_url=endpoint)` instead of spawning a local process
   - When `endpoint` is not set, keep current local behavior (backward compatible)
   - Files: `runner/pipeline_factory.py`, `models/model.py`

3. Create k8s manifests for the embeddings runner node:
   - Separate Deployment with node selector for the embeddings hardware
   - Service exposing the runner
   - Mount the same `/models` hostPath (or use a shared PV)
   - Files: new `runner/k8s/` manifests

### Phase 2: Model Registry (can defer)
4. Add model discovery so the server auto-detects runner capabilities:
   - Option A: Static config — server's `.models.yaml` lists `endpoint` per model
   - Option B: Dynamic — server polls runner `/v1/models` endpoints on startup
   - Option C: K8s annotations on runner services

### Phase 3: Advanced Routing (can defer)  
5. Smart routing: load balancing, failover, model migration between nodes

## Relevant Files
- `inference/runner/pipeline_factory.py` — `get_pipeline()`, `_create_text_pipeline()`, `_create_embedding_pipeline()` — the routing decision point
- `inference/runner/pipelines/llamacpp/chat.py` — `ChatLlamaCppPipeline._initialize_chat_openai()` shows the ChatOpenAI-over-localhost pattern
- `inference/runner/pipeline_cache.py` — `LocalPipelineCacheManager` — would stay on runner nodes
- `inference/runner/server_manager/llamacpp.py` — `LlamaCppServerManager` — would stay on runner nodes
- `inference/models/model.py` — `Model` — needs `endpoint` field
- `inference/models/model_parameters.py` — `ModelParameters` — GPU config stays, used by runner
- `inference/runner/utils/model_loader.py` — `ModelLoader` — loads model config
- `inference/k8s/deployment.yaml` — current monolith deployment
- `inference/server/app.py` — server startup, pipeline cache lifecycle hooks

## Verification
1. Deploy embeddings runner on separate node, verify `/v1/models` and `/v1/embeddings` work
2. Configure server's model config with `endpoint` for embedding models
3. Verify existing chat completions still work (no regression)
4. Verify embedding requests route to the remote runner
5. Load test: concurrent embedding + chat requests don't contend for GPU

## Decisions
- Phase 1 is backward compatible — models without `endpoint` use local pipelines as today
- Runner service is intentionally thin — it's basically a managed llama-server pool behind an HTTP facade
- No service mesh needed — k8s Services provide sufficient routing
- The server's composer/agents don't change at all — they still call `pipeline_factory.get_pipeline()` which returns a LangChain `BaseChatModel`

## Further Considerations
1. **Runner API surface**: Should runners expose raw llama-server ports directly (simpler, one less proxy hop) vs. a thin FastAPI wrapper (model management, health aggregation, future load balancing)? Recommend: thin wrapper for Phase 1, could bypass later if perf matters.
2. **Model file access**: Shared NFS/hostPath vs. copying model files to each node? Current setup uses hostPath `/models` — works if all nodes mount the same NAS.
3. **Pipeline cache on server side**: Should the server cache remote ChatOpenAI clients or create fresh per-request? Current behavior for remote providers is create-fresh. Recommend: keep it simple, create-fresh — these are just HTTP clients with connection pooling.
