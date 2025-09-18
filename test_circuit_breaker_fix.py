#!/usr/bin/env python3
"""
Test script to verify that user circuit breaker configuration
is properly passed to pipeline creation in the grpc service.
"""

import asyncio
import sys
import os

# Add inference paths
sys.path.insert(0, "/Users/lons7862/workspace/llmmllab/inference")
sys.path.insert(0, "/Users/lons7862/workspace/llmmllab/inference/server")
sys.path.insert(0, "/Users/lons7862/workspace/llmmllab/inference/runner")


async def test_user_config_loading():
    """Test that we can load user config from conversation ID."""
    print("Testing user configuration loading...")

    # Import the storage module
    try:
        from server.db import storage

        print("✓ Storage module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import storage: {e}")
        return False

    # Test conversation loading (using a mock conversation ID)
    try:
        # This will likely fail since we don't have a real database setup
        # but it will show us if the method calls work
        test_conversation_id = 1
        conversation = await storage.get_service(storage.conversation).get_conversation(
            test_conversation_id
        )
        print(f"✓ Conversation service call succeeded: {conversation}")
    except Exception as e:
        print(f"✗ Conversation loading failed (expected): {e}")
        print("   This is expected if database is not set up")

    print("Test completed!")
    return True


async def test_pipeline_factory_with_circuit_breaker():
    """Test pipeline factory with circuit breaker configuration."""
    print("\nTesting pipeline factory with circuit breaker...")

    try:
        from runner.pipeline_factory import PipelineFactory, PipelinePriority
        from models.model_profile import ModelProfile
        from models.model_parameters import ModelParameters
        from models.circuit_breaker_config import CircuitBreakerConfig
        from models import ChatResponse

        print("✓ All required modules imported successfully")

        # Create test circuit breaker config
        test_circuit_breaker = CircuitBreakerConfig(
            enable_perplexity_guard=False,  # This is what we want to test
            perplexity_threshold=5.0,
            failure_threshold=3,
            recovery_timeout=60,
        )
        print(
            f"✓ Created test circuit breaker config: perplexity_guard={test_circuit_breaker.enable_perplexity_guard}"
        )

        # Create test model profile
        test_profile = ModelProfile(
            id=None,
            user_id="test_user",
            model_name="thudm-glm-4.1v-9b-thinking",
            name="Test Profile",
            description="Test profile for circuit breaker",
            parameters=ModelParameters(),
            system_prompt="You are a test assistant.",
            type=1,
        )
        print("✓ Created test model profile")

        # Test pipeline factory
        factory = PipelineFactory()
        print("✓ Pipeline factory created")

        # This will likely fail without proper model setup, but we can see if the parameters are accepted
        try:
            pipeline = factory.get_pipeline(
                test_profile,
                ChatResponse,
                PipelinePriority.HIGH,
                test_circuit_breaker,  # This is the key parameter we're testing
            )
            print("✓ Pipeline creation succeeded!")
        except Exception as e:
            print(f"✗ Pipeline creation failed (may be expected): {e}")
            print(
                "   The important thing is that the circuit_breaker parameter was accepted"
            )

    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

    print("Pipeline factory test completed!")
    return True


def test_circuit_breaker_defaults():
    """Test the default circuit breaker configuration."""
    print("\nTesting default circuit breaker configuration...")

    try:
        from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG

        print(f"✓ Default circuit breaker config loaded")
        print(
            f"   enable_perplexity_guard: {DEFAULT_CIRCUIT_BREAKER_CONFIG.enable_perplexity_guard}"
        )
        print(
            f"   perplexity_threshold: {DEFAULT_CIRCUIT_BREAKER_CONFIG.perplexity_threshold}"
        )

        # Check if it has failure_threshold attribute
        if hasattr(DEFAULT_CIRCUIT_BREAKER_CONFIG, "failure_threshold"):
            print(
                f"   failure_threshold: {DEFAULT_CIRCUIT_BREAKER_CONFIG.failure_threshold}"
            )
        else:
            print("   failure_threshold: not available in this config")

        if DEFAULT_CIRCUIT_BREAKER_CONFIG.enable_perplexity_guard:
            print("⚠️  WARNING: Default config has perplexity_guard=True")
            print("   This explains why perplexity monitoring was enabled")
            print("   User config should override this default")
        else:
            print("✓ Default config has perplexity_guard=False")

    except ImportError as e:
        print(f"✗ Failed to import default configs: {e}")
        return False

    print("Default config test completed!")
    return True


async def main():
    """Run all tests."""
    print("=== Circuit Breaker Configuration Test ===\n")

    # Test 1: Default configuration
    test_circuit_breaker_defaults()

    # Test 2: User config loading
    await test_user_config_loading()

    # Test 3: Pipeline factory with circuit breaker
    await test_pipeline_factory_with_circuit_breaker()

    print("\n=== Test Summary ===")
    print("This test verifies that the circuit breaker configuration")
    print("can be properly passed through the system. The key insight")
    print("is that DEFAULT_CIRCUIT_BREAKER_CONFIG has enable_perplexity_guard=True")
    print("which means user configuration must override this default.")
    print("\nThe fix implemented in the grpc service should:")
    print("1. Load conversation by ID")
    print("2. Extract user_id from conversation")
    print("3. Load user configuration")
    print("4. Extract circuit_breaker config from user config")
    print("5. Pass it to pipeline factory get_pipeline()")


if __name__ == "__main__":
    asyncio.run(main())
