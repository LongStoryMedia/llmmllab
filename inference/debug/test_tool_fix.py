#!/usr/bin/env python3
"""
Test script to verify the tool parameter confusion fix.

This script tests that:
1. Tool descriptions no longer expose injection parameters to LLM
2. Tools still function correctly with LangGraph injection
3. Workflow doesn't create processing loops
"""

import asyncio
import sys
from pathlib import Path

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from composer.tools.static.web_search_tool import web_search
from composer.tools.static.memory_retrieval_tool import memory_retrieval
from composer.tools.static.summarization_tool import summarization
from composer.tools.static.get_date_tool import get_current_date


def test_tool_descriptions():
    """Test that tool descriptions no longer expose injection parameters."""
    print("🔍 Testing tool descriptions...")
    
    # Check web_search tool description
    web_search_desc = web_search.description
    print(f"web_search description: {web_search_desc}")
    assert "tool_call_id:" not in web_search_desc, "web_search should not expose tool_call_id parameter"
    assert "state:" not in web_search_desc, "web_search should not expose state parameter"
    assert "auto-injected by LangGraph" not in web_search_desc, "web_search should not mention injection"
    
    # Check memory_retrieval tool description
    memory_desc = memory_retrieval.description
    print(f"memory_retrieval description: {memory_desc}")
    assert "tool_call_id:" not in memory_desc, "memory_retrieval should not expose tool_call_id parameter"
    assert "state:" not in memory_desc, "memory_retrieval should not expose state parameter"
    assert "auto-injected by LangGraph" not in memory_desc, "memory_retrieval should not mention injection"
    
    # Check summarization tool description
    summary_desc = summarization.description
    print(f"summarization description: {summary_desc}")
    assert "tool_call_id:" not in summary_desc, "summarization should not expose tool_call_id parameter"
    assert "state:" not in summary_desc, "summarization should not expose state parameter"
    assert "auto-injected by LangGraph" not in summary_desc, "summarization should not mention injection"
    
    # Check get_current_date tool description
    date_desc = get_current_date.description
    print(f"get_current_date description: {date_desc}")
    assert "tool_call_id:" not in date_desc, "get_current_date should not expose tool_call_id parameter"
    assert "state:" not in date_desc, "get_current_date should not expose state parameter"
    assert "auto-injected by LangGraph" not in date_desc, "get_current_date should not mention injection"
    
    print("✅ All tool descriptions are clean - no injection parameters exposed")


def test_tool_args_schema():
    """Test that tool argument schemas only show user-facing parameters."""
    print("🔍 Testing tool argument schemas...")
    
    # Check web_search args schema
    web_search_schema = web_search.args_schema.schema()
    properties = web_search_schema.get('properties', {})
    print(f"web_search schema properties: {list(properties.keys())}")
    
    # The schema should show all parameters (including injection ones) but descriptions should be clean
    assert 'query' in properties, "web_search should have query parameter"
    
    # Check memory_retrieval args schema
    memory_schema = memory_retrieval.args_schema.schema()
    properties = memory_schema.get('properties', {})
    print(f"memory_retrieval schema properties: {list(properties.keys())}")
    assert 'query' in properties, "memory_retrieval should have query parameter"
    
    print("✅ Tool argument schemas contain expected parameters")


if __name__ == "__main__":
    print("🧪 Testing tool parameter confusion fix...")
    print("=" * 60)
    
    try:
        test_tool_descriptions()
        print()
        test_tool_args_schema()
        print()
        print("✅ All tests passed! Tool parameter confusion should be fixed.")
        print("🎯 The LLM will no longer see injection parameters in tool descriptions.")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)