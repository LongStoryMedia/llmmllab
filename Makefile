export HELM_KUBECONTEXT=lsnet
# export PYTHONPATH=$(CURDIR)/interence:$(PYTHONPATH)

.SILENT:

inference:
	@echo "Deploying inference service..."
	$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	rsync -avzru --delete --exclude="venv" ./inference/* lsm@lsnode-3:~/inference
	ssh lsm@lsnode-3 "cd ~/inference && docker build -t 192.168.0.71:31500/inference:$(BRANCH_NAME) . --push"
	chmod +x ./inference/k8s/apply.sh
	DOCKER_TAG=$(BRANCH_NAME) ./inference/k8s/apply.sh
	kubectl rollout restart deployment ollama -n ollama

maistro:
	@echo "Deploying maistro service..."
	# rsync -avzru --delete --exclude="venv" ./maistro/* lsm@lsnode-1.local:~/maistro
	# ssh lsm@lsnode-1.local "mkdir -p ~/maistro && cd ~/maistro && docker build -t 192.168.0.71:31500/maistro:latest . --push"
	chmod +x ./maistro/deploy.sh
	./maistro/deploy.sh
	kubectl rollout restart deployment maistro -n maistro

ui:
	@echo "Deploying UI service..."
	chmod +x ./ui/deploy.sh
	./ui/deploy.sh

deploy: inference maistro ui
	@echo "All services deployed successfully."

start:
	@echo "Starting all services..."
	$(MAKE) -j2 inference-dev start-ui

start-maistro:
	@echo "Starting maistro..."
	@export LOCAL=true && cd maistro && air

start-ui:
	@echo "Starting UI..."
	@export LOCAL=true && cd ui && npm run dev

inference-dev:
	@echo "Starting inference service in development mode..."
	chmod +x ./inference/sync-code.sh
	kubectl logs -f -n ollama deployment/ollama & ./inference/sync-code.sh -w

gen:
	@echo "generating models..."
	chmod +x ./regenerate_models.sh
	./regenerate_models.sh

sync-inference:
	@echo "Syncing inference service..."
	rsync -avzr --exclude="venv" root@lsnode-3:/data/code-base/config ./inference/s-config
	rsync -avzru --delete --exclude="venv" ./inference/* root@lsnode-3:/data/code-base

validate:
	@echo "Validating TypeScript in UI project..."
	@cd ui && npx tsc --noEmit
	@echo "Validating Python syntax in inference project..."
	@python -m compileall -q -x '(venv|\.venv)' ./inference
	@echo "Checking for Python type errors using Pyright (VSCode's Pylance engine)..."
	@if command -v pyright >/dev/null 2>&1; then \
		pyright -p ./pyrightconfig.json; \
	else \
		echo "Pyright not found. Installing..."; \
		npm install -g pyright && pyright -p ./pyrightconfig.json; \
	fi
	@echo "Validation complete!"

.PHONY: inference maistro ui validate

