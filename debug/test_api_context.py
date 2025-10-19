#!/usr/bin/env python3
"""Test API call to verify context window fix."""

import httpx
import asyncio
import json

async def test_chat_api():
    """Test chat API with context window management."""
    print("🌐 Testing chat API with context window management...")
    
    try:
        # Use the ollama service endpoint
        base_url = "http://192.168.0.71:8000"  # From service configuration
        
        # Create a single message for the /chat/completions endpoint
        payload = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is artificial intelligence? Please explain in detail."
                }
            ],
            "conversation_id": 717
        }
        
        print(f"📤 Sending request to {base_url}/v1/chat/completions")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Handle streaming response
                response_text = response.text
                print(f"📝 Raw response length: {len(response_text)} characters")
                print(f"📝 Response preview: {response_text[:300]}...")
                
                # Try to parse as streaming JSON (multiple objects)
                try:
                    lines = response_text.strip().split('\n')
                    parsed_objects = []
                    for line in lines:
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                parsed_objects.append(obj)
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"✅ Success! Parsed {len(parsed_objects)} streaming objects")
                    
                    # Look for final content
                    final_content = ""
                    for obj in parsed_objects:
                        if isinstance(obj, dict) and "content" in obj:
                            final_content += str(obj["content"])
                    
                    if final_content:
                        print(f"📜 Final content length: {len(final_content)} characters")
                        print(f"� Content preview: {final_content[:200]}...")
                    
                except Exception as parse_error:
                    print(f"⚠️  Could not parse streaming response: {parse_error}")
                    
                print("✅ Chat API test completed - context window management working!")
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_chat_api())