#!/usr/bin/env python3
"""
Direct test of llama-cpp-python function calling - bypass our pipeline entirely
"""

import json


def test_llamacpp_direct():
    """Test llama-cpp-python function calling directly."""
    print("🔍 Testing llama-cpp-python function calling directly...")

    try:
        import llama_cpp

        print("✅ llama_cpp imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import llama_cpp: {e}")
        return

    # Create simple tool schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather like in Paris?"}]

    print(f"📤 Tools: {json.dumps(tools, indent=2)}")
    print(f"📤 Messages: {messages}")

    try:
        # Try to create llama instance with actual model path
        llama = llama_cpp.Llama(
            model_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
            n_ctx=4096,
            verbose=False,
        )

        print("✅ Llama instance created")

        # Test function calling
        response = llama.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
            temperature=0.1,
        )

        print(f"📥 Response type: {type(response)}")
        print(
            f"📥 Response keys: {response.keys() if isinstance(response, dict) else 'Not dict'}"
        )

        if isinstance(response, dict) and "choices" in response:
            message = response["choices"][0]["message"]
            print(f"📝 Message keys: {message.keys()}")
            print(f"📝 Content: {message.get('content', 'NO CONTENT')}")

            tool_calls = message.get("tool_calls")
            if tool_calls:
                print(f"✅ TOOL CALLS FOUND: {tool_calls}")
            else:
                print("❌ NO TOOL CALLS - checking content for manual parsing")
                content = message.get("content", "")
                if any(
                    marker in content
                    for marker in ["<function", "function_call", "tool_call"]
                ):
                    print("🎯 Found function call markup in content - needs parsing!")
                    print(f"Content sample: {content[:500]}")

    except Exception as e:
        print(f"💥 Error during direct test: {e}")
        print("This is expected if running outside container")


if __name__ == "__main__":
    test_llamacpp_direct()
