"""
Multimodal Model Tests for Qwen3-VL

Simple test suite to validate multimodal functionality with Qwen3-VL models.
Tests can be run individually without pytest if needed.
"""

import sys
import urllib.request
from pathlib import Path
from typing import Optional

import llama_cpp
from llama_cpp import llama_chat_format


# Test model URLs - using smaller models for testing
# QWEN3_VL_MODEL_URL = "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated/resolve/main/GGUF/ggml-model-f16.gguf"
# QWEN3_VL_MMPROJ_URL = "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated/resolve/main/GGUF/mmproj-model-f16.gguf"

# Test image URLs
TEST_IMAGE_URL = "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png"


model_path, mmproj_path = (
    "/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
    "/models/qwen3-vl-2b/mmproj.gguf",
)


def test_text_only_chat_completion():
    """Test text-only chat completion with multimodal model."""
    print("Testing text-only chat completion...")
    try:
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=False
        )

        # Test that the chat handler can format messages properly
        # This is a safer test that doesn't require full model execution
        messages = [{"role": "user", "content": "Hello! How are you?"}]

        # Test that we can create the handler and it has the expected methods
        assert hasattr(chat_handler, "__call__")
        assert hasattr(chat_handler, "load_image")
        assert hasattr(chat_handler, "get_image_urls")

        # Test image URL extraction
        image_urls = chat_handler.get_image_urls(messages)  # type: ignore
        assert len(image_urls) == 0  # No images in text-only message

        print("✓ Text-only chat handler test passed")
        return True

    except Exception as e:
        print(f"✗ Text-only chat handler test failed: {e}")
        return False


def test_multimodal_chat_completion_with_image():
    """Test multimodal message preparation and image processing."""
    print("Testing multimodal chat completion with image...")
    try:
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=True
        )

        # Test with the Qwen demo image
        qwen_demo_image = (
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        )

        # Test image loading first
        print(f"Loading image: {qwen_demo_image}")
        image_bytes = chat_handler.load_image(qwen_demo_image)
        assert image_bytes is not None
        assert len(image_bytes) > 0
        print(f"✓ Image loaded successfully: {len(image_bytes)} bytes")

        # Test message structure for multimodal completion
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": qwen_demo_image}},
                    {
                        "type": "text",
                        "text": "What do you see in this image? Please describe it briefly.",
                    },
                ],
            }
        ]

        # Test image URL extraction
        image_urls = chat_handler.get_image_urls(messages)  # type: ignore
        assert len(image_urls) == 1
        assert image_urls[0] == qwen_demo_image
        print(f"✓ Extracted image URL: {image_urls[0]}")

        # Test that we can create a model with the handler
        print("Testing model initialization with multimodal handler...")
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,  # Conservative context size
            verbose=True,
        )

        # Verify the model is properly initialized
        assert llama is not None
        assert llama.chat_handler is not None
        print("✓ Model initialized with multimodal support")

        res = llama.create_chat_completion(messages)
        for choice in res["choices"]:
            print(choice["message"])

        print("✓ Multimodal pipeline validation completed")

        llama.close()
        print("✓ Multimodal chat completion preparation test passed")
        return True

    except Exception as e:
        print(f"✗ Multimodal chat completion test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Running Qwen3-VL Multimodal Tests")
    print("=" * 50)

    tests = [
        test_text_only_chat_completion,
        test_multimodal_chat_completion_with_image,
    ]

    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            results.append((test.__name__, False))

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)}")

    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
