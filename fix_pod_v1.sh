#!/bin/bash

echo "🔄 Fixing Pod and Testing LangChain v1.0 Compatibility"

# Delete the problematic pod
echo "1. Deleting problematic pod..."
kubectl delete pod -n ollama --all

echo "2. Waiting for new pod to start..."
sleep 10

# Wait for pod to be ready
echo "3. Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod -n ollama -l app=ollama --timeout=300s

# Get new pod name
POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
echo "4. New pod ready: $POD_NAME"

echo "5. Testing LangChain v1.0 compatibility in pod..."

# Test basic imports
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "
print('🧪 Testing LangChain v1.0 in server environment...')

try:
    import langchain
    print(f'✅ LangChain version: {langchain.__version__}')
except Exception as e:
    print(f'❌ LangChain import failed: {e}')

try:
    from langchain.agents import ToolNode
    print('✅ ToolNode import successful')
except Exception as e:
    print(f'❌ ToolNode import failed: {e}')
    
print('🎯 Server environment test complete')
"

kubectl exec -n ollama $POD_NAME -- /app/v.sh composer python -c "
print('🧪 Testing composer workflows...')

try:
    from composer.workflows.chat import build_chat_workflow
    print('✅ Chat workflow import successful')
except Exception as e:
    print(f'❌ Chat workflow failed: {e}')

try:
    from composer.workflows.multi_agent import build_multi_agent_workflow
    print('✅ Multi-agent workflow import successful')
except Exception as e:
    print(f'❌ Multi-agent workflow failed: {e}')

print('🎯 Composer environment test complete')
"

echo ""
echo "🎉 Pod restart and compatibility test complete!"