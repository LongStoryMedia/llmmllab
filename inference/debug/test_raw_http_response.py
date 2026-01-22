#!/usr/bin/env python3
"""
Test raw HTTP response from llama.cpp server to debug function calling format.
"""
import asyncio
import json
import httpx
from pathlib import Path
import sys
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_raw_http_response")


async def test_raw_http_response():
    """Test raw HTTP response from llama.cpp server."""
    logger.info("🧪 Testing raw HTTP response from llama.cpp server")

    # Simple function calling request
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco? Please use the weather function.",
            }
        ],
        "model": "local-model",
        "temperature": 0.1,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    }

    try:
        async with httpx.AsyncClient() as client:
            logger.info("📤 Sending raw HTTP request to llama.cpp server...")

            response = await client.post(
                "http://localhost:8001/v1/chat/completions",
                json=request_data,
                timeout=60.0,
            )

            logger.info(f"📨 Response status: {response.status_code}")
            logger.info(f"📨 Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                response_json = response.json()
                logger.info(f"📨 Response JSON keys: {list(response_json.keys())}")

                if "choices" in response_json and response_json["choices"]:
                    choice = response_json["choices"][0]
                    message = choice.get("message", {})

                    logger.info(f"📨 Message keys: {list(message.keys())}")
                    logger.info(f"📨 Message content: '{message.get('content', '')}'")
                    logger.info(f"📨 Message role: {message.get('role', '')}")

                    # Check for tool_calls
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                        logger.info(
                            f"✅ Found {len(tool_calls)} tool calls in response!"
                        )
                        for i, tc in enumerate(tool_calls):
                            logger.info(f"🔧 Tool call {i}: {tc}")
                    else:
                        logger.warning("❌ No 'tool_calls' field in message")

                    # Log the full message for inspection
                    logger.info(f"📨 Full message: {json.dumps(message, indent=2)}")
                else:
                    logger.error("❌ No choices in response")

                # Log full response for debugging
                logger.info(f"📨 Full response: {json.dumps(response_json, indent=2)}")
            else:
                logger.error(f"❌ HTTP error {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"❌ Request failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_raw_http_response())
