#!/bin/bash

echo "🔧 Fixing LangChain 0.3.x Compatibility (Pod Version)"

cd /Users/lons7862/workspace/llmmllab

echo "Updating imports for LangChain 0.3.x (actual pod version)..."

# Update imports to work with 0.3.x version in pod
find inference/ -name "*.py" -exec grep -l "from langchain.agents import ToolNode" {} \; | while read file; do
    echo "  Updating $file for 0.3.x compatibility"
    # In 0.3.x, ToolNode is in langgraph.prebuilt
    sed -i.bak 's/from langchain\.agents import ToolNode/from langgraph.prebuilt import ToolNode/' "$file"
done

# Clean backup files
find inference/ -name "*.py.bak" -delete 2>/dev/null || true

echo "✅ Updated imports for LangChain 0.3.x compatibility"

# Sync code
echo "📤 Syncing updated code..."
./inference/sync-code.sh

echo "✅ Import fixes for actual pod version complete"