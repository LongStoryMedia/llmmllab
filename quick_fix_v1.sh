#!/bin/bash

echo "🔧 Quick LangChain v1.0 Fix"

cd /Users/lons7862/workspace/llmmllab

# Update specific known files with tools_condition issues
echo "Fixing runner pipeline imports..."

for file in $(find inference/runner -name "*.py" -exec grep -l "tools_condition" {} \; 2>/dev/null); do
    echo "  Updating $file"
    sed -i.bak 's/from langchain\.agents import ToolNode, tools_condition/from langchain.agents import ToolNode/' "$file"
    sed -i.bak 's/from langgraph\.prebuilt import ToolNode, tools_condition/from langchain.agents import ToolNode/' "$file"
done

# Clean backup files
find inference/ -name "*.py.bak" -delete 2>/dev/null || true

echo "✅ Import fixes complete"