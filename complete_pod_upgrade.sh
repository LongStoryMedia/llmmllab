#!/bin/bash

echo "🔄 Complete Pod Restart and LangChain 1.0.0a1 Setup"

echo "1. Restarting pod..."
kubectl delete pod -n ollama --all
sleep 15

echo "2. Waiting for new pod..."
kubectl wait --for=condition=ready pod -n ollama -l app=ollama --timeout=300s

POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
echo "3. New pod ready: $POD_NAME"

echo "4. Upgrading all virtual environments to LangChain 1.0.0a1..."

echo "   Upgrading server venv..."
kubectl exec -n ollama $POD_NAME -- bash -c '
    source /opt/venv/server/bin/activate
    pip install --upgrade langchain==1.0.0a1 langchain-core langchain-community langchain-openai
    echo "Server venv upgraded"
'

echo "   Upgrading composer venv..."  
kubectl exec -n ollama $POD_NAME -- bash -c '
    source /opt/venv/composer/bin/activate
    pip install --upgrade langchain==1.0.0a1 langchain-core langchain-community langchain-openai
    echo "Composer venv upgraded"
'

echo "   Upgrading runner venv..."
kubectl exec -n ollama $POD_NAME -- bash -c '
    source /opt/venv/runner/bin/activate  
    pip install --upgrade langchain==1.0.0a1 langchain-core langchain-community langchain-openai
    echo "Runner venv upgraded"
'

echo ""
echo "5. Final compatibility test..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "
import langchain
print(f'✅ Server LangChain: {langchain.__version__}')
from langchain.agents import ToolNode
print('✅ Server ToolNode works')
"

kubectl exec -n ollama $POD_NAME -- /app/v.sh composer python -c "
import langchain  
print(f'✅ Composer LangChain: {langchain.__version__}')
from langchain.agents import ToolNode
print('✅ Composer ToolNode works')
from composer.workflows.chat import build_chat_workflow
print('✅ Composer workflows work')
"

echo ""
echo "🎉 Pod fully upgraded to LangChain 1.0.0a1!"