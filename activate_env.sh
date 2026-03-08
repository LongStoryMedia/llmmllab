#!/bin/bash
# Helper script to activate specific environments
case "$1" in
"runner")
source /opt/venv/runner/bin/activate
;;
"server") 
source /opt/venv/server/bin/activate
;;
"evaluation")
source /opt/venv/evaluation/bin/activate
;;
*)
echo "Usage: source activate_env.sh [runner|server|evaluation]"
return 1
;;
esac
echo "Activated $1 environment"
echo "Available modules: runner, server, evaluation (cross-accessible)"