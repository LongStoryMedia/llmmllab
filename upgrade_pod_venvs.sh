#!/bin/bash

echo "🔄 Upgrading Pod Virtual Environments to LangChain 1.0.0a1"

POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD_NAME"

echo ""
echo "📋 Step 1: Upgrading server venv..."
kubectl exec -n ollama $POD_NAME -- /opt/venv/server/bin/pip install --upgrade \
    langchain==1.0.0a1 \
    langchain-core \
    langchain-community \
    langchain-openai

echo ""
echo "📋 Step 2: Upgrading composer venv..."
kubectl exec -n ollama $POD_NAME -- /opt/venv/composer/bin/pip install --upgrade \
    langchain==1.0.0a1 \
    langchain-core \
    langchain-community \
    langchain-openai

echo ""
echo "📋 Step 3: Upgrading runner venv..."
kubectl exec -n ollama $POD_NAME -- /opt/venv/runner/bin/pip install --upgrade \
    langchain==1.0.0a1 \
    langchain-core \
    langchain-community \
    langchain-openai

echo ""
echo "📋 Step 4: Testing upgraded environments..."

echo "  Testing server environment..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "
import langchain
print(f'✅ Server LangChain: {langchain.__version__}')
try:
    from langchain.agents import ToolNode
    print('✅ Server ToolNode import works')
except Exception as e:
    print(f'❌ Server ToolNode: {e}')
"

echo ""
echo "  Testing composer environment..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh composer python -c "
import langchain
print(f'✅ Composer LangChain: {langchain.__version__}')
try:
    from langchain.agents import ToolNode
    print('✅ Composer ToolNode import works')
except Exception as e:
    print(f'❌ Composer ToolNode: {e}')

try:
    from composer.workflows.chat import build_chat_workflow
    print('✅ Composer chat workflow import works')
except Exception as e:
    print(f'❌ Composer chat workflow: {e}')
"

echo ""
echo "  Testing runner environment..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh runner python -c "
import langchain
print(f'✅ Runner LangChain: {langchain.__version__}')
try:
    from langchain.agents import ToolNode
    print('✅ Runner ToolNode import works')
except Exception as e:
    print(f'❌ Runner ToolNode: {e}')
"

echo ""
echo "🎉 Pod upgrade to LangChain 1.0.0a1 complete!"