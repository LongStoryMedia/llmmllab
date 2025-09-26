#!/usr/bin/env python3
"""
Simple test for Qwen2.5VL tool call parsing functionality.
"""

import sys
import json
import re

def test_parse_qwen_tool_calls(content: str):
    """Test the parsing logic for Qwen function calls (updated pattern)."""
    tool_calls = []
    
    # Pattern 1: Look for proper Qwen function call format - extract full JSON object first
    # This handles both string and object arguments correctly
    function_call_json_pattern = r'\{"function_call":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\}'
    function_json_matches = re.findall(function_call_json_pattern, content, re.DOTALL)
    
    for i, func_json in enumerate(function_json_matches):
        try:
            # Parse the function call object
            func_obj = json.loads(func_json)
            name = func_obj.get('name')
            args_data = func_obj.get('arguments', '{}')
            
            if name:
                # If arguments is a string, parse it as JSON
                if isinstance(args_data, str):
                    try:
                        args = json.loads(args_data)
                    except json.JSONDecodeError:
                        args = {"raw": args_data}
                else:
                    args = args_data
                
                formatted_call = {
                    "name": name,
                    "args": args,
                    "id": f"call_{i}_{name}",
                    "type": "tool_call"
                }
                tool_calls.append(formatted_call)
                print(f"Parsed Qwen function_call: {formatted_call}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse function_call JSON '{func_json[:100]}...': {e}")
            continue

    # Pattern 2: Look for mixed function_call tags
    if not tool_calls:
        mixed_pattern = r'<function_call>\s*(\{.*?\})\s*</(?:function_call|FunctionCall)>'
        mixed_matches = re.findall(mixed_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(mixed_matches):
            try:
                tool_data = json.loads(match)
                
                if "name" in tool_data:
                    formatted_call = {
                        "name": tool_data["name"],
                        "args": tool_data.get("arguments", {}),
                        "id": f"call_{i}_{tool_data['name']}",
                        "type": "tool_call"
                    }
                    tool_calls.append(formatted_call)
                    print(f"Parsed mixed function_call: {formatted_call}")
                else:
                    print(f"Mixed function call missing 'name' field: {match[:100]}...")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse mixed function call from: {match[:100]}... Error: {e}")
                continue

    return tool_calls

def test_clean_tool_calls_from_content(content: str) -> str:
    """Remove tool call patterns from content to get clean user-facing text."""
    
    # Remove function_call JSON patterns (proper Qwen format) - updated pattern
    func_call_pattern = r'\{"function_call":\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}'
    content = re.sub(func_call_pattern, '', content, flags=re.DOTALL)
    
    # Remove mixed function_call tags
    mixed_pattern = r'<function_call>\s*\{.*?\}\s*</(?:function_call|FunctionCall)>'
    content = re.sub(mixed_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    return content

def main():
    """Run the tests."""
    print("🧪 Testing Qwen2.5VL Tool Call Parsing")
    print("=" * 50)
    
    # Test 1: Mixed function call format (from logs)
    test_content1 = '''I can see this is a nature scene with a wooden boardwalk. Let me search for more information.

<function_call>
{"name": "web_search", "arguments": {"query": "wooden boardwalk nature park Wisconsin"}}
</FunctionCall>'''
    
    print("\n📝 Test 1: Mixed function_call format")
    result1 = test_parse_qwen_tool_calls(test_content1)
    print(f"✅ Found {len(result1)} tool calls")
    
    # Test 2: Proper Qwen format
    test_content2 = '''This appears to be a beautiful nature boardwalk. {"function_call": {"name": "web_search", "arguments": "{\\"query\\": \\"nature boardwalk photography\\"}"}}'''
    
    print("\n📝 Test 2: Proper Qwen function_call format") 
    result2 = test_parse_qwen_tool_calls(test_content2)
    print(f"✅ Found {len(result2)} tool calls")
    
    # Test 3: Content cleaning
    test_content3 = '''This is a scenic wooden boardwalk through what appears to be a natural area.

<function_call>
{"name": "web_search", "arguments": {"query": "nature boardwalk Wisconsin"}}
</function_call>

The image shows a peaceful walkway extending into the distance with natural vegetation on both sides.'''
    
    print("\n📝 Test 3: Content cleaning")
    cleaned = test_clean_tool_calls_from_content(test_content3)
    print(f"Original length: {len(test_content3)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print(f"Cleaned content: {repr(cleaned)}")
    
    # Test 4: No strip issues (preserve spaces)
    test_content4 = "The year 2024 has been remarkable for quantum computing breakthroughs."
    cleaned4 = test_clean_tool_calls_from_content(test_content4)
    
    print("\n📝 Test 4: Preserve spaces and formatting")
    print(f"Original: {repr(test_content4)}")
    print(f"Cleaned: {repr(cleaned4)}")
    
    if "2024" in cleaned4 and " 2024 " in cleaned4:
        print("✅ Spaces around numbers preserved correctly")
    else:
        print("❌ Spaces around numbers were removed")
    
    print("\n🎉 Qwen2.5VL parsing tests completed!")
    
    total_tests = 4
    passed_tests = 0
    
    if len(result1) > 0:
        passed_tests += 1
    if len(result2) > 0:
        passed_tests += 1
    if len(cleaned) < len(test_content3):
        passed_tests += 1
    if " 2024 " in cleaned4:
        passed_tests += 1
        
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())