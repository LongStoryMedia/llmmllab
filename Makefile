export HELM_KUBECONTEXT=lsnet
# export PYTHONPATH=$(CURDIR)/interence:$(PYTHONPATH)

.SILENT:

# =============================================================================
# DEVELOPMENT SERVERS
# =============================================================================

start:
	@echo "Starting all development servers..."
	$(MAKE) -j2 start-inference start-ui

start-inference:
	@echo "Starting inference service in development mode..."
	chmod +x ./inference/sync-code.sh
	kubectl logs -f -n llmmll deployment/llmmll & ./inference/sync-code.sh -w

start-ui:
	@echo "Starting UI development server..."
	@export LOCAL=true && cd ui && npm run dev

start-maistro:
	@echo "Starting maistro..."
	@export LOCAL=true && cd maistro && air

# =============================================================================
# CODE GENERATION
# =============================================================================

gen:
	@echo "Generating models from schemas..."
	chmod +x ./build.sh
	./build.sh

gen-python:
	@echo "Generating Python models from schemas..."
	chmod +x ./regenerate_models.sh
	./regenerate_models.sh python

gen-typescript:
	@echo "Generating TypeScript types from schemas..."
	chmod +x ./regenerate_models.sh
	./regenerate_models.sh typescript

gen-all:
	@echo "Generating all models (Python + TypeScript)..."
	chmod +x ./build.sh
	./build.sh

# =============================================================================
# BUILD & DEPLOYMENT - KUBERNETES
# =============================================================================

deploy: deploy-server deploy-composer deploy-runner
	@echo "All services deployed successfully."

deploy-server:
	@echo "Deploying server service (multi-arch)..."
	$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	chmod +x ./server/k8s/build.sh
	DOCKER_TAG=$(BRANCH_NAME) ./server/k8s/build.sh
	DOCKER_TAG=$(BRANCH_NAME) ./server/k8s/apply.sh
	kubectl rollout restart deployment llmmll-server -n llmmll

deploy-composer:
	@echo "Deploying composer service (multi-arch)..."
	$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	chmod +x ./composer/k8s/build.sh
	DOCKER_TAG=$(BRANCH_NAME) ./composer/k8s/build.sh
	DOCKER_TAG=$(BRANCH_NAME) ./composer/k8s/apply.sh
	kubectl rollout restart deployment llmmll-composer -n llmmll

deploy-runner:
	@echo "Deploying runner service (GPU-enabled)..."
	$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	chmod +x ./runner/k8s/build.sh
	DOCKER_TAG=$(BRANCH_NAME) ./runner/k8s/build.sh
	DOCKER_TAG=$(BRANCH_NAME) ./runner/k8s/apply.sh
	kubectl rollout restart deployment llmmll-runner -n llmmll

# =============================================================================
# BUILD & DEPLOYMENT - INFERENCE (legacy)
# =============================================================================

inference:
	@echo "Deploying inference service..."
	$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	inference/sync-code.sh
	ssh root@lsnode-3.local "cd /data/code-base && docker build -t 192.168.0.71:31500/inference:$(BRANCH_NAME) . --push"
	chmod +x ./inference/k8s/apply.sh
	DOCKER_TAG=$(BRANCH_NAME) ./inference/k8s/apply.sh
	kubectl rollout restart deployment llmmll -n llmmll

ui:
	@echo "Deploying UI service..."
	chmod +x ./ui/deploy.sh
	./ui/deploy.sh

# =============================================================================
# LOCAL DEVELOPMENT
# =============================================================================

dev-server:
	@echo "Starting server in development mode..."
	@export LOCAL=true && cd server && python -m server.grpc.server

dev-composer:
	@echo "Starting composer in development mode..."
	@export LOCAL=true && cd composer && python -m composer.grpc.server

dev-runner:
	@echo "Starting runner in development mode..."
	@export LOCAL=true && cd runner && python -m runner.server

# =============================================================================
# VALIDATION & TESTING
# =============================================================================

validate:
	@echo "Validating TypeScript in UI project..."
	@cd ui && npx tsc --noEmit
	@echo "Validating Python syntax in inference project..."
	@python -m compileall -q -x '(venv|\.venv)' ./inference
	@echo "Checking for Python type errors using Pyright..."
	@if command -v pyright >/dev/null 2>&1; then \
		pyright -p ./pyrightconfig.json; \
	else \
		echo "Pyright not found. Installing..."; \
		npm install -g pyright && pyright -p ./pyrightconfig.json; \
	fi
	@echo "Validation complete!"

test:
	@echo "Running tests for inference and UI"
	cd inference && pytest test/
	cd ui && npm run test

test-inference:
	@echo "Running inference tests..."
	cd inference && pytest test/

test-ui:
	@echo "Running UI tests..."
	cd ui && npm run test

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	@echo "Cleaning artifacts..."
	rm -rf ./inference/debug/out/
	rm -rf ./ui/build/
	rm -rf ./inference/models/
	@echo "Artifacts cleaned."

clean-py:
	@echo "Cleaning Python artifacts..."
	rm -rf ./inference/__pycache__/
	rm -rf ./inference/*/__pycache__/
	rm -rf ./inference/*/*/__pycache__/
	rm -rf ./inference/*/*/*/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Python artifacts cleaned."

clean-pycache:
	$(MAKE) clean-py

# =============================================================================
# SUBMODULE MANAGEMENT
# =============================================================================

sync-submodules:
	@echo "Syncing submodules..."
	@for submodule in composer runner schemas server ui; do \
		echo "Processing $$submodule..."; \
		branch=$$(git -C $$submodule rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"); \
		echo "  Branch: $$branch"; \
		git -C $$submodule pull origin $$branch --ff-only 2>/dev/null || true; \
		git -C $$submodule add -A 2>/dev/null || true; \
		if [ -n "$$(git -C $$submodule status --porcelain 2>/dev/null)" ]; then \
			git -C $$submodule commit -m "Update from root repository"; \
			git -C $$submodule push origin $$branch 2>/dev/null || true; \
		else \
			echo "  No changes to commit"; \
		fi; \
	done; \
	git submodule update --remote; \
	git add composer runner schemas server ui; \
	if [ -n "$$(git status --porcelain)" ]; then \
		git commit -m "Update submodules"; \
		git push origin $$(git rev-parse --abbrev-ref HEAD) 2>/dev/null || true; \
	else \
		echo "Root repo: No changes to commit"; \
	fi
	@echo "Submodules synced successfully."

push-all: sync-submodules
	@echo "Pushing all changes..."
	@TIMESTAMP=$$(date +%s); \
	git add .; \
	git commit -m "Update: $$TIMESTAMP" || true; \
	git push origin $$(git rev-parse --abbrev-ref HEAD);
	@echo "All changes pushed successfully."

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "LLM ML Lab - Makefile Commands"
	@echo ""
	@echo "DEVELOPMENT SERVERS"
	@echo "  start              - Start all development servers (inference + UI)"
	@echo "  start-inference    - Start inference service in dev mode (syncs to k8s)"
	@echo "  start-ui           - Start UI development server"
	@echo "  start-maistro      - Start maistro service"
	@echo ""
	@echo "CODE GENERATION"
	@echo "  gen                - Generate all models (Python + TypeScript)"
	@echo "  gen-python         - Generate Python models only"
	@echo "  gen-typescript     - Generate TypeScript types only"
	@echo "  gen-all            - Alias for gen"
	@echo ""
	@echo "KUBERNETES DEPLOYMENT"
	@echo "  deploy             - Deploy all services (server, composer, runner)"
	@echo "  deploy-server      - Deploy server service (multi-arch)"
	@echo "  deploy-composer    - Deploy composer service (multi-arch)"
	@echo "  deploy-runner      - Deploy runner service (GPU-enabled, lsnode-3)"
	@echo "  inference          - Deploy legacy inference service"
	@echo "  ui                 - Deploy UI service"
	@echo ""
	@echo "LOCAL DEVELOPMENT"
	@echo "  dev-server         - Run server locally (without k8s)"
	@echo "  dev-composer       - Run composer locally (without k8s)"
	@echo "  dev-runner         - Run runner locally (without k8s)"
	@echo ""
	@echo "VALIDATION & TESTING"
	@echo "  validate           - Run TypeScript and Python validation"
	@echo "  test               - Run all tests (inference + UI)"
	@echo "  test-inference     - Run inference tests only"
	@echo "  test-ui            - Run UI tests only"
	@echo ""
	@echo "CLEANUP"
	@echo "  clean              - Remove build artifacts"
	@echo "  clean-py           - Remove Python cache files"
	@echo "  clean-pycache      - Alias for clean-py"
	@echo ""
	@echo "SUBMODULE MANAGEMENT"
	@echo "  sync-submodules    - Sync all submodules and push changes"
	@echo "  push-all           - Sync submodules and push all changes"
	@echo ""
	@echo "For more information, see the README.md"

.PHONY: start start-inference start-ui start-maistro \
	gen gen-python gen-typescript gen-all \
	deploy deploy-server deploy-composer deploy-runner \
	inference ui \
	dev-server dev-composer dev-runner \
	validate test test-inference test-ui \
	clean clean-py clean-pycache \
	sync-submodules push-all help