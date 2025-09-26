#!/usr/bin/env python3
"""
Debug the Qwen format issue.
"""

import json
import re

def debug_qwen_format():
    test_content = '''This appears to be a beautiful nature boardwalk. {"function_call": {"name": "web_search", "arguments": "{\\"query\\": \\"nature boardwalk photography\\"}"}}'''
    
    print("Debug Qwen Function Call Format")
    print("=" * 40)
    print(f"Test content: {repr(test_content)}")
    
    # Try different patterns
    patterns = [
        r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*"([^"]+)"\s*\}',
        r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*("[^"]*")\s*\}',
        r'"function_call":\s*\{[^}]*"name":\s*"([^"]+)"[^}]*"arguments":\s*"([^"]*)"[^}]*\}',
    ]
    
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}: {pattern}")
        matches = re.findall(pattern, test_content, re.DOTALL)
        print(f"Matches: {matches}")
        
        if matches:
            for j, (name, args_str) in enumerate(matches):
                print(f"  Match {j}: name='{name}', args='{args_str}'")
                try:
                    args = json.loads(args_str)
                    print(f"  Parsed args: {args}")
                except Exception as e:
                    print(f"  Parse error: {e}")

if __name__ == "__main__":
    debug_qwen_format()