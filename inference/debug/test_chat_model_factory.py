#!/usr/bin/env python3
"""
Test script to verify the new BaseChatModel architecture works.
Tests Qwen3Moe as a BaseChatModel implementation.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Model, ModelProfile
from runner.chat_model_factory import chat_model_factory


def test_qwen3moe_chat_model():
    """Test creating and using Qwen3Moe as a BaseChatModel."""
    print("🧪 Testing Qwen3Moe as BaseChatModel...")

    # Use the chat model factory to get an existing model
    # First check what models are available
    from runner.pipeline_factory import pipeline_factory

    # Look for a Qwen model in the available models
    qwen_model = None
    for model_name, model in pipeline_factory.models.items():
        if model.pipeline == "Qwen3Pipe":
            qwen_model = model
            break

    if not qwen_model:
        print("❌ No Qwen3Pipe model found in configuration")
        return False

    print(f"Found Qwen model: {qwen_model.name}")

    # Create a test profile
    from models import ModelParameters

    test_profile = ModelProfile(
        model_name=qwen_model.name,
        user_id="test-user",
        name="test-profile",
        system_prompt="You are a helpful AI assistant.",
        parameters=ModelParameters(),
        type=1,  # Default profile type
    )

    # Try to create chat model
    try:
        chat_model = chat_model_factory.create_chat_model(qwen_model, test_profile)

        if chat_model is None:
            print("❌ Failed to create chat model")
            return False

        print(f"✅ Successfully created chat model: {type(chat_model).__name__}")
        print(f"   LLM Type: {chat_model._llm_type}")
        print(f"   Identifying Params: {chat_model._identifying_params}")

        return True

    except Exception as e:
        print(f"❌ Error creating chat model: {e}")
        return False


if __name__ == "__main__":
    success = test_qwen3moe_chat_model()
    if success:
        print("\n🎉 BaseChatModel architecture test passed!")
    else:
        print("\n💥 BaseChatModel architecture test failed!")
        sys.exit(1)
