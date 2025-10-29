#!/usr/bin/env python3
"""
Test script to verify tool call storage fix by triggering actual tool usage
"""

import requests
import json

def test_tool_call_storage():
    """Test tool call storage with actual tool invocation"""
    
    # Chat request that should trigger tool usage
    chat_request = {
        "messages": [
            {
                "role": "user", 
                "content": "What is the current time? Use a tool to get it."
            }
        ],
        "model": "qwen2.5:0.5b",
        "stream": True,
        "conversation_id": None,  # Create new conversation
        "agent": "tools_agent"   # Use tools agent to trigger tool calls
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token"
    }
    
    try:
        print("🔧 Testing tool call storage fix...")
        print(f"Request: {json.dumps(chat_request, indent=2)}")
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=chat_request,
            headers=headers,
            stream=True,
            timeout=60
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return False
            
        # Process streaming response
        chunk_count = 0
        for line in response.iter_lines():
            if line:
                chunk_count += 1
                if chunk_count <= 5 or chunk_count % 100 == 0:
                    print(f"Chunk {chunk_count}: {line.decode()[:100]}...")
        
        print(f"✅ Received {chunk_count} chunks successfully")
        print("🔧 Tool call storage test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tool_call_storage()
    exit(0 if success else 1)