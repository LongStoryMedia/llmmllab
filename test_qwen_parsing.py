#!/usr/bin/env python3
"""
Test script for Qwen tool call parsing.
"""
import json
import re

def test_parse_qwen_tool_calls(content: str):
    """Test the parsing logic for Qwen function calls."""
    tool_calls = []
    
    # Pattern 1: Look for proper Qwen function call format (arguments as JSON string)
    function_call_pattern_str = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*"([^"]+)"\s*\}'
    function_matches_str = re.findall(function_call_pattern_str, content, re.DOTALL)
    
    for i, (name, args_str) in enumerate(function_matches_str):
        try:
            args = json.loads(args_str)
            formatted_call = {
                "name": name,
                "args": args,
                "id": f"call_{i}_{name}",
                "type": "tool_call"
            }
            tool_calls.append(formatted_call)
            print(f"Parsed Qwen function_call (string args): {formatted_call}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse function_call arguments '{args_str}': {e}")
            continue
    
    # Pattern 1b: Look for proper Qwen function call format (arguments as JSON object)
    if not tool_calls:
        function_call_pattern_obj = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\s*\}'
        function_matches_obj = re.findall(function_call_pattern_obj, content, re.DOTALL)
        
        for i, (name, args_str) in enumerate(function_matches_obj):
            try:
                args = json.loads(args_str)
                formatted_call = {
                    "name": name,
                    "args": args,
                    "id": f"call_{i}_{name}",
                    "type": "tool_call"
                }
                tool_calls.append(formatted_call)
                print(f"Parsed Qwen function_call (object args): {formatted_call}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse function_call arguments '{args_str}': {e}")
                continue
    
    # Pattern 2: Mixed function_call tags (what we see in logs)
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

def main():
    print("🧪 Testing Qwen Tool Call Parsing")
    print("=" * 50)
    
    # Test 1: Mixed format (from user's logs)
    test_content1 = '''I'll help you get the weather information for New York.

<function_call>
{"name": "get_weather", "arguments": {"location": "New York"}}
</FunctionCall>'''
    
    print("\n📝 Test 1: Mixed format (function_call/FunctionCall)")
    print(f"Content: {test_content1}")
    result1 = test_parse_qwen_tool_calls(test_content1)
    print(f"✅ Found {len(result1)} tool calls")
    
    # Test 2: Proper Qwen format  
    test_content2 = '''{"function_call": {"name": "get_weather", "arguments": "{\\"location\\": \\"New York\\"}"}}'''
    
    print("\n📝 Test 2: Proper Qwen format")
    print(f"Content: {test_content2}")
    
    # Debug regex matching
    function_call_pattern = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*"([^"]+)"\s*\}'
    function_matches = re.findall(function_call_pattern, test_content2, re.DOTALL)
    print(f"Debug: function_call regex found {len(function_matches)} matches: {function_matches}")
    
    result2 = test_parse_qwen_tool_calls(test_content2)
    print(f"✅ Found {len(result2)} tool calls")
    
    # Test 3: No tool calls
    test_content3 = '''Just a regular response without any tool calls.'''
    
    print("\n📝 Test 3: No tool calls")
    print(f"Content: {test_content3}")
    result3 = test_parse_qwen_tool_calls(test_content3)
    print(f"✅ Found {len(result3)} tool calls")

    print("\n🎉 Testing complete!")

if __name__ == "__main__":
    main()