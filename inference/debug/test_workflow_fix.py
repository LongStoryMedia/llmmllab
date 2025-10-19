#!/usr/bin/env python3
"""
Test script to verify the workflow execution works after the parameter fix.
"""

import asyncio
import json
import sys
import os

# Add the app directory to Python path
sys.path.append('/app')

async def test_chat_completion():
    """Test a basic chat completion to verify the workflow works."""
    print("🧪 Testing chat completion workflow after parameter fix...")
    
    try:
        import httpx
        
        # Test payload similar to what caused the original error
        test_payload = {
            "model": "Qwen3-4B", 
            "messages": [
                {"role": "user", "content": "Hello, can you help me with a simple question?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print("   📤 Sending chat completion request...")
        
        # Send request to local server
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8080/v1/chat/completions",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Chat completion successful!")
                print(f"   Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')[:100]}...")
            else:
                print(f"❌ Chat completion failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat_completion())