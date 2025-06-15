export HELM_KUBECONTEXT=lsnet
# export PYTHONPATH=$(CURDIR)/interence:$(PYTHONPATH)

.SILENT:

inference:
	@echo "Deploying inference service..."
	rsync -avzru --delete --exclude="venv" ./inference/* lsm@lsnode-3:~/inference
	ssh lsm@lsnode-3 "cd ~/inference && docker build -t 192.168.0.71:31500/inference:latest . --push"
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
	$(MAKE) -j2 start-maistro start-ui

start-maistro:
	@echo "Starting maistro..."
	@export LOCAL=true && cd maistro && air

start-ui:
	@echo "Starting UI..."
	@export LOCAL=true && cd ui && npm run dev

gen:
	@echo "generating models..."
	chmod +x ./regenerate_models.sh
	./regenerate_models.sh

gen-%:
	@echo "generating models..."
	chmod +x ./generate.sh
	./generate.sh $*

genlang-go-%:
	@echo "generating go models..."
	chmod +x ./generate.sh
	./generate.sh $* go

genlang-ts-%:
	@echo "generating ts models..."
	chmod +x ./generate.sh
	./generate.sh $* ts

genlang-py-%:
	@echo "generating python models..."
	chmod +x ./generate.sh
	./generate.sh $* py

.PHONY: inference maistro ui

