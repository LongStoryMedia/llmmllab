#!/usr/bin/env python3
"""
Simple tool name validation test to check if the smart tool detection is working.
"""

import requests

def test_tool_name_detection():
    """Test tool name detection with a simple request."""
    print("🧪 Testing Tool Name Detection...")
    print("=" * 50)
    
    try:
        # Simple request that should trigger web search
        response = requests.post(
            'http://localhost:8000/chat/completions',
            json={
                'messages': [{'role': 'user', 'content': 'Search for AI news'}],
                'model': 'qwen3-30b-a3b-q4-k-m',
                'stream': False
            },
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if there are tool calls in the response
            message = result.get('message', {})
            tool_calls = message.get('tool_calls', [])
            
            print(f"Tool Calls Found: {len(tool_calls)}")
            
            for i, tool_call in enumerate(tool_calls):
                name = tool_call.get('name', 'unknown')
                print(f"  Tool {i+1}: {name}")
                
                if name != 'unknown_tool':
                    print(f"  ✅ SUCCESS: Tool name detected as '{name}'")
                    return True
                else:
                    print("  ❌ FAILED: Tool name still shows as 'unknown_tool'")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    success = test_tool_name_detection()
    if success:
        print("\n🎉 Tool name detection is working!")
        exit(0)
    else:
        print("\n❌ Tool name detection needs more work")
        exit(1)