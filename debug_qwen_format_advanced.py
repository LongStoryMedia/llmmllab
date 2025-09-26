#!/usr/bin/env python3
"""
Debug the Qwen format issue directly on cluster.
"""

import json
import re

def debug_qwen_format():
    # This is what the actual test content looks like
    test_content = '''This appears to be a beautiful nature boardwalk. {"function_call": {"name": "web_search", "arguments": "{\\"query\\": \\"nature boardwalk photography\\"}"}}'''
    
    print("Debug Qwen Function Call Format")
    print("=" * 40)
    print(f"Test content: {repr(test_content)}")
    print(f"Raw content: {test_content}")
    
    # More robust pattern that handles nested quotes properly
    # Look for the JSON structure and extract the entire function_call object
    function_call_pattern = r'\{"function_call":\s*(\{[^}]*\})\}'
    
    matches = re.findall(function_call_pattern, test_content)
    print(f"\nFunction call objects found: {len(matches)}")
    
    for i, match in enumerate(matches):
        print(f"Match {i}: {match}")
        try:
            # Try to parse the function call object
            func_obj = json.loads(match)
            print(f"  Parsed function object: {func_obj}")
            
            name = func_obj.get('name')
            args_str = func_obj.get('arguments', '{}')
            print(f"  Name: {name}")
            print(f"  Args string: {repr(args_str)}")
            
            # Try to parse the arguments
            try:
                args = json.loads(args_str)
                print(f"  Parsed args: {args}")
            except Exception as e:
                print(f"  Args parse error: {e}")
                
        except Exception as e:
            print(f"  Parse error: {e}")
    
    # Also try the full JSON parse approach
    print("\n" + "="*40)
    print("Trying full JSON extraction...")
    
    # Extract the entire JSON object
    json_pattern = r'\{[^{}]*"function_call"[^{}]*\{[^{}]*\}[^{}]*\}'
    json_matches = re.findall(json_pattern, test_content)
    
    for i, json_match in enumerate(json_matches):
        print(f"JSON match {i}: {json_match}")
        try:
            parsed = json.loads(json_match)
            print(f"  Parsed JSON: {parsed}")
            
            if 'function_call' in parsed:
                func_call = parsed['function_call']
                print(f"  Function call: {func_call}")
                name = func_call.get('name')
                args_str = func_call.get('arguments', '{}')
                print(f"  Name: {name}, Args: {repr(args_str)}")
                
                try:
                    args = json.loads(args_str)
                    print(f"  Final args: {args}")
                except Exception as e:
                    print(f"  Final args parse error: {e}")
                    
        except Exception as e:
            print(f"  JSON parse error: {e}")

if __name__ == "__main__":
    debug_qwen_format()