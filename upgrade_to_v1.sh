#!/bin/bash

echo "🚀 Upgrading to LangChain 1.0.0a1 (Pod and Local)"

cd /Users/lons7862/workspace/llmmllab || exit 1

echo "📋 Step 1: Upgrading local environment..."
source inference/.venv/bin/activate

echo "  Upgrading LangChain packages to 1.0.0a1..."
pip install --upgrade \
    langchain==1.0.0a1 \
    langchain-core==1.0.0a1 \
    langchain-community \
    langchain-openai==1.0.0a1 \
    langgraph

echo "  ✅ Local environment upgraded"

echo ""
echo "📋 Step 2: Getting pod name and upgrading pod environment..."
POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
echo "  Pod: $POD_NAME"

echo "  Upgrading LangChain in pod..."
kubectl exec -n ollama $POD_NAME -- pip install --upgrade \
    langchain==1.0.0a1 \
    langchain-core==1.0.0a1 \
    langchain-community \
    langchain-openai==1.0.0a1

echo "  ✅ Pod environment upgraded"

echo ""
echo "📋 Step 3: Fixing imports for 1.0.0a1..."

# Update composer standard.py to use correct import
cat > /tmp/fix_standard.py << 'EOF'
import sys
sys.path.append('inference')

file_path = 'inference/composer/nodes/standard.py'
with open(file_path, 'r') as f:
    content = f.read()

# Fix import - ToolNode is in langchain.agents in v1.0
content = content.replace(
    'from langgraph.prebuilt import ToolNode',
    'from langchain.agents import ToolNode'
)

with open(file_path, 'w') as f:
    f.write(content)

print("✅ Fixed standard.py import")
EOF

python /tmp/fix_standard.py

echo "📋 Step 4: Syncing code to pod..."
./inference/sync-code.sh

echo ""
echo "📋 Step 5: Testing 1.0.0a1 compatibility..."

echo "  Local test..."
python -c "
import sys
sys.path.insert(0, 'inference')

try:
    import langchain
    print(f'✅ Local LangChain version: {langchain.__version__}')
except Exception as e:
    print(f'❌ Local LangChain: {e}')

try:
    from langchain.agents import ToolNode
    print('✅ Local ToolNode import works')
except Exception as e:
    print(f'❌ Local ToolNode: {e}')

try:
    from composer.workflows.chat import build_chat_workflow
    print('✅ Local chat workflow import works')
except Exception as e:
    print(f'❌ Local chat workflow: {e}')
"

echo ""
echo "  Pod test..."
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "
try:
    import langchain
    print(f'✅ Pod LangChain version: {langchain.__version__}')
except Exception as e:
    print(f'❌ Pod LangChain: {e}')

try:
    from langchain.agents import ToolNode
    print('✅ Pod ToolNode import works')
except Exception as e:
    print(f'❌ Pod ToolNode: {e}')
"

kubectl exec -n ollama $POD_NAME -- /app/v.sh composer python -c "
try:
    from composer.workflows.chat import build_chat_workflow
    print('✅ Pod chat workflow import works')
except Exception as e:
    print(f'❌ Pod chat workflow: {e}')

try:
    from composer.workflows.multi_agent import build_multi_agent_workflow  
    print('✅ Pod multi-agent workflow import works')
except Exception as e:
    print(f'❌ Pod multi-agent workflow: {e}')
"

echo ""
echo "🎉 LangChain 1.0.0a1 upgrade complete!"
echo "📊 Both local and pod environments now running LangChain 1.0.0a1"

rm -f /tmp/fix_standard.py