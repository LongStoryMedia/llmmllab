export HELM_KUBECONTEXT=lsnet

.SILENT:

start:
	@export LOCAL=true && air

deploy:
	@chmod +x ./install.sh
	@echo "Deploying image..."
	@./install.sh $(CURDIR)