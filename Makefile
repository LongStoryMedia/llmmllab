export HELM_KUBECONTEXT=lsnet
# export PYTHONPATH=$(CURDIR)/interence:$(PYTHONPATH)

.SILENT:

inference:
	@echo "Deploying inference service..."
	$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	inference/sync-code.sh
	ssh root@lsnode-3 "cd /data/code-base && docker build -t 192.168.0.71:31500/inference:$(BRANCH_NAME) . --push"
	chmod +x ./inference/k8s/apply.sh
	DOCKER_TAG=$(BRANCH_NAME) ./inference/k8s/apply.sh
	kubectl rollout restart deployment ollama -n ollama

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

e2e-%:
	kubectl exec -it -n ollama $$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}') -- /app/v.sh server python -m debug.test_real_end_to_end_pipeline $*

clear-debug:
	rm ./inference/debug/out/*.txt 
	rm ./inference/debug/out/*.json
	rm ./inference/debug/out/*.md
	./inference/sync-code.sh -R

.PHONY: inference maistro ui validate

