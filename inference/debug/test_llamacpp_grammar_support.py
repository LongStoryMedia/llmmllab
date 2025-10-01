#!/usr/bin/env python3
"""
Test llamacpp grammar paramedef test_llama_cpp_python_version():
    \"\"\"Check llama-cpp-python version and features.\"\"\"
    try:
        import llama_cpp
        logger.info(f\"✅ llama-cpp-python version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'unknown'}\")

        # Check if Llama class supports grammar
        if hasattr(llama_cpp, 'Llama'):
            llama_init_signature = inspect.signature(llama_cpp.Llama.__init__)
            if 'grammar' in llama_init_signature.parameters:
                logger.info(\"✅ llama_cpp.Llama supports 'grammar' parameter\")
                return True
            else:
                logger.info(\"ℹ️ llama_cpp.Llama does not support 'grammar' parameter\")

        return False

    except Exception as e:
        logger.error(f\"❌ Error checking llama-cpp-python: {e}\")
        return Falsecript checks if ChatLlamaCpp supports grammar parameters.
"""

import logging
import sys
import os
import inspect

sys.path.insert(0, "/app")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llamacpp_grammar_parameter():
    """Test if ChatLlamaCpp supports grammar parameter."""
    try:
        from langchain_community.chat_models.llamacpp import ChatLlamaCpp
        import inspect

        # Get the __init__ method signature
        init_signature = inspect.signature(ChatLlamaCpp.__init__)

        # Check if 'grammar' parameter is supported
        if "grammar" in init_signature.parameters:
            logger.info("✅ ChatLlamaCpp supports 'grammar' parameter")
            param = init_signature.parameters["grammar"]
            logger.info(f"   Parameter details: {param}")
            return True

        # Check if 'grammar_path' parameter is supported
        if "grammar_path" in init_signature.parameters:
            logger.info("✅ ChatLlamaCpp supports 'grammar_path' parameter")
            param = init_signature.parameters["grammar_path"]
            logger.info(f"   Parameter details: {param}")
            return True

        logger.warning(
            "⚠️ ChatLlamaCpp does not support 'grammar' or 'grammar_path' parameters"
        )

        # List all parameters for debugging
        logger.info("Available parameters:")
        for name, param in init_signature.parameters.items():
            if name not in ["self", "args", "kwargs"]:
                logger.info(f"  - {name}: {param}")

        return False

    except Exception as e:
        logger.error(f"❌ Error checking ChatLlamaCpp parameters: {e}")
        return False


def test_llama_cpp_python_version():
    """Check llama-cpp-python version and features."""
    try:
        import llama_cpp

        logger.info(
            f"✅ llama-cpp-python version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'unknown'}"
        )

        # Check if Llama class supports grammar
        if hasattr(llama_cpp, "Llama"):
            llama_init_signature = inspect.signature(llama_cpp.Llama.__init__)
            if "grammar" in llama_init_signature.parameters:
                logger.info("✅ llama_cpp.Llama supports 'grammar' parameter")
                return True
            else:
                logger.info("ℹ️ llama_cpp.Llama does not support 'grammar' parameter")

        return False

    except Exception as e:
        logger.error(f"❌ Error checking llama-cpp-python: {e}")
        return False


def test_response_format_support():
    """Check if ChatLlamaCpp supports response_format parameter."""
    try:
        from langchain_community.chat_models.llamacpp import ChatLlamaCpp
        import inspect

        init_signature = inspect.signature(ChatLlamaCpp.__init__)

        if "response_format" in init_signature.parameters:
            logger.info("✅ ChatLlamaCpp supports 'response_format' parameter")
            param = init_signature.parameters["response_format"]
            logger.info(f"   Parameter details: {param}")
            return True

        logger.info("ℹ️ ChatLlamaCpp does not support 'response_format' parameter")
        return False

    except Exception as e:
        logger.error(f"❌ Error checking response_format support: {e}")
        return False


def main():
    """Run llamacpp grammar tests."""
    logger.info("🧪 Testing llamacpp grammar support...")

    # Test ChatLlamaCpp grammar parameter
    logger.info("\n📋 Test 1: ChatLlamaCpp Grammar Parameter")
    grammar_supported = test_llamacpp_grammar_parameter()

    # Test llama-cpp-python version and features
    logger.info("\n📋 Test 2: llama-cpp-python Features")
    llama_cpp_grammar = test_llama_cpp_python_version()

    # Test response_format support
    logger.info("\n📋 Test 3: Response Format Support")
    response_format_supported = test_response_format_support()

    # Summary
    logger.info("\n📊 Summary:")
    if grammar_supported or response_format_supported:
        logger.info("✅ Structured output is supported via grammar or response_format")
    elif llama_cpp_grammar:
        logger.info(
            "ℹ️ Grammar is supported at llama-cpp-python level but not exposed in ChatLlamaCpp"
        )
    else:
        logger.warning(
            "⚠️ No grammar support detected - may need to use alternative approach"
        )

    logger.info(
        "🔧 Recommendation: Use response_format parameter if available, otherwise extend ChatLlamaCpp"
    )


if __name__ == "__main__":
    main()
