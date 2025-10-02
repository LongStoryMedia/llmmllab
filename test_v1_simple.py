#!/usr/bin/env python3

import sys
sys.path.insert(0, 'inference')

print("Testing LangChain v1.0 compatibility...")

# Test 1: Basic imports
try:
    from langchain.agents import ToolNode
    print("✅ ToolNode import works")
except Exception as e:
    print(f"❌ ToolNode import failed: {e}")

# Test 2: Message model
try:
    from models import LangChainMessage
    msg = LangChainMessage(type='human', content='test')
    print("✅ LangChainMessage works")
except Exception as e:
    print(f"❌ LangChainMessage failed: {e}")

# Test 3: LangGraph imports
try:
    from langgraph.graph import StateGraph, END, add_messages
    print("✅ LangGraph imports work")
except Exception as e:
    print(f"❌ LangGraph imports failed: {e}")

print("Basic compatibility test complete.")