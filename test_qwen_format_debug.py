#!/usr/bin/env python3
"""
Targeted debug for Test 2 Qwen format issue.
"""

import sys
import os
sys.path.append('/app')

from runner.pipelines.imgtxt2txt.qwen25_vl import Qwen25VLPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_qwen_format_debug():
    print("🔍 Debugging Test 2 Qwen Format Issue")
    print("=" * 50)
    
    # Create a minimal pipeline instance just for the parsing method
    pipeline = Qwen25VLPipeline(None, None, None)
    
    test_content = '''This appears to be a beautiful nature boardwalk. {"function_call": {"name": "web_search", "arguments": "{\"query\": \"nature boardwalk photography\"}"}}'''
    
    print(f"Input content: {repr(test_content)}")
    print()
    
    # Test the parsing method directly
    tool_calls = pipeline._parse_qwen_tool_calls(test_content)
    
    print(f"🎯 Result: Found {len(tool_calls)} tool calls")
    for i, call in enumerate(tool_calls):
        print(f"  Call {i}: {call}")
    
    print()
    print("🧹 Testing content cleaning:")
    cleaned = pipeline._clean_tool_calls_from_content(test_content)
    print(f"Original length: {len(test_content)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"Cleaned content: {repr(cleaned)}")

if __name__ == "__main__":
    test_qwen_format_debug()