#!/bin/bash

echo "🧪 Testing Pod LangChain v1.0 Compatibility"

# Get pod name
POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "❌ No pod found in ollama namespace"
    exit 1
fi

echo "📋 Testing pod: $POD_NAME"

echo "1. Testing composer imports..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh composer python -c "
import sys; sys.path.insert(0, '/app')

try:
    from langchain.agents import ToolNode
    print('✅ ToolNode import works')
except Exception as e:
    print(f'❌ ToolNode: {e}')

try:
    from composer.workflows.chat import build_chat_workflow
    print('✅ Chat workflow import works') 
except Exception as e:
    print(f'❌ Chat workflow: {e}')

try:
    from composer.workflows.multi_agent import build_multi_agent_workflow
    print('✅ Multi-agent workflow import works')
except Exception as e:
    print(f'❌ Multi-agent: {e}')
"

echo ""
echo "2. Testing server imports..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "
try:
    from langchain.agents import ToolNode
    print('✅ Server ToolNode import works')
except Exception as e:
    print(f'❌ Server ToolNode: {e}')
"

echo ""
echo "3. Testing runner imports..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh runner python -c "
try:
    from langchain.agents import ToolNode
    print('✅ Runner ToolNode import works')
except Exception as e:
    print(f'❌ Runner ToolNode: {e}')
"

echo ""
echo "🎯 Pod compatibility test complete"