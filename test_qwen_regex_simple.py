#!/usr/bin/env python3
"""
Simple regex test for Qwen format parsing.
"""

import json
import re

def test_qwen_regex():
    print("🔍 Testing Qwen Regex Patterns")
    print("=" * 40)
    
    test_content = '''This appears to be a beautiful nature boardwalk. {"function_call": {"name": "web_search", "arguments": "{\"query\": \"nature boardwalk photography\"}"}}'''
    
    print(f"Test content: {repr(test_content)}")
    print(f"Raw content: {test_content}")
    print()
    
    # Try the new pattern
    pattern = r'\{"function_call":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\}'
    
    print(f"Pattern: {pattern}")
    matches = re.findall(pattern, test_content, re.DOTALL)
    
    print(f"Matches found: {len(matches)}")
    
    for i, match in enumerate(matches):
        print(f"Match {i}: {repr(match)}")
        try:
            parsed = json.loads(match)
            print(f"  Parsed: {parsed}")
            
            name = parsed.get('name')
            args_str = parsed.get('arguments')
            print(f"  Name: {name}")
            print(f"  Args string: {repr(args_str)}")
            
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                    print(f"  Parsed args: {args}")
                except Exception as e:
                    print(f"  Args parse error: {e}")
            else:
                print(f"  Args (direct): {args_str}")
                
        except Exception as e:
            print(f"  Parse error: {e}")

if __name__ == "__main__":
    test_qwen_regex()