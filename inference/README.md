# Inference Service

Python FastAPI service that serves OpenAI- and Anthropic-compatible inference endpoints backed by `llama.cpp` and diffusion pipelines. Deployed to Kubernetes.

## Layout

```
inference/
├── server/         FastAPI app, routers (openai/, anthropic/, common/), middleware
├── runner/         Model execution: pipeline_factory, pipeline_cache, pipelines/
├── composer/       LangGraph agent orchestration, workflows, tool generation
├── db/             Multi-tier storage (memory → Redis → Postgres)
├── models/         *Generated* from schemas/ — do not edit
├── utils/          Shared helpers (logging, message conversion, tool-call types)
├── k8s/            Deployment manifests + apply.sh
├── Dockerfile      CUDA 12.8 runtime image, single shared venv
└── requirements.txt
```

Architecture, entry points, and developer conventions live in the repo-root [CLAUDE.md](../CLAUDE.md).

## Running

All commands run from the repo root via the Makefile:

```bash
make inference-dev   # sync code to k8s, tail logs (primary dev loop)
make deploy          # deploy inference + maistro + ui
make test            # pytest + UI tests
make validate        # tsc --noEmit, Python compileall, Pyright
```

The dev loop pushes local changes into the running pod via `inference/sync-code.sh` — no image rebuild required for Python edits.

In-pod commands use `/app/v.sh`:

```bash
kubectl exec -it -n llmmll <pod> -- /app/v.sh server python -m <module>
```

## Schema-driven models

Data contracts live as YAML in [`schemas/`](../schemas/). Regenerate after schema edits:

```bash
./regenerate_models.sh              # both Python + TypeScript
./regenerate_models.sh python       # inference/models/ only
./regenerate_models.sh typescript   # ui/src/types/ only
```

Never hand-edit files under `inference/models/`.

## Building the image

```bash
docker build -t llmmllab:latest -f inference/Dockerfile .
```

Everything installs into a single venv at `/opt/venv/shared` and is wired up for editable imports.

## Endpoints

- `POST /v1/chat/completions` — OpenAI chat (streaming)
- `POST /v1/embeddings` — OpenAI embeddings
- `POST /v1/messages` — Anthropic messages (streaming)
- `POST /v1/images/generations` — text-to-image
- `POST /v1/audio/transcriptions` — Whisper speech-to-text
- `GET /openapi.json`, `/docs` — FastAPI-generated, unauthenticated

See individual router files under `server/routers/` for the full list.
