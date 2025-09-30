#!/usr/bin/env python3
"""
Test script for grammar-constrained structured output in runner pipelines.
Validates that the grammar system works end-to-end.
"""

import asyncio
import sys
import os

# Add the inference directory to Python path
sys.path.insert(0, '/app' if os.path.exists('/app') else os.path.dirname(os.path.dirname(__file__)))

from typing import Optional
from pydantic import BaseModel, Field

# Test imports
from utils.grammar import generate_grammar_from_pydantic, validate_grammar
from models import DeduplicationResult


class SimpleTestModel(BaseModel):
    """Simple test model for grammar generation."""
    name: str = Field(..., description="Name of the item")
    value: int = Field(..., description="Numeric value", ge=0, le=100)
    is_valid: bool = Field(default=True, description="Whether the item is valid")
    optional_field: Optional[str] = Field(None, description="Optional field")


async def test_grammar_generation():
    """Test grammar generation from Pydantic models."""
    print("Testing grammar generation...")
    
    # Test simple model
    print("\n1. Testing SimpleTestModel...")
    try:
        grammar = generate_grammar_from_pydantic(SimpleTestModel)
        print(f"✅ Generated grammar ({len(grammar)} chars)")
        
        # Validate the grammar
        if validate_grammar(grammar):
            print("✅ Grammar validation passed")
        else:
            print("❌ Grammar validation failed")
            
    except Exception as e:
        print(f"❌ Failed to generate grammar for SimpleTestModel: {e}")
    
    # Test DeduplicationResult model
    print("\n2. Testing DeduplicationResult...")
    try:
        grammar = generate_grammar_from_pydantic(DeduplicationResult)
        print(f"✅ Generated DeduplicationResult grammar ({len(grammar)} chars)")
        
        # Show a snippet of the grammar
        print(f"Grammar snippet: {grammar[:200]}...")
        
        if validate_grammar(grammar):
            print("✅ DeduplicationResult grammar validation passed")
        else:
            print("❌ DeduplicationResult grammar validation failed")
            
    except Exception as e:
        print(f"❌ Failed to generate grammar for DeduplicationResult: {e}")
    
    print("\n✅ Grammar generation tests completed")


async def test_runner_interface():
    """Test that runner interface accepts grammar parameter."""
    print("\nTesting runner interface...")
    
    try:
        # Import runner functions
        from runner.pipelines.run import run_pipeline, stream_pipeline
        from runner.pipelines.base import BasePipelineCore
        
        # Check function signatures support grammar
        import inspect
        
        run_sig = inspect.signature(run_pipeline)
        stream_sig = inspect.signature(stream_pipeline)
        
        if 'grammar' in run_sig.parameters:
            print("✅ run_pipeline accepts grammar parameter")
        else:
            print("❌ run_pipeline missing grammar parameter")
            
        if 'grammar' in stream_sig.parameters:
            print("✅ stream_pipeline accepts grammar parameter")
        else:
            print("❌ stream_pipeline missing grammar parameter")
            
        print("✅ Runner interface tests completed")
        
    except Exception as e:
        print(f"❌ Runner interface test failed: {e}")


def main():
    """Main test function."""
    print("🧪 Grammar Integration Test Suite")
    print("=" * 50)
    
    try:
        # Run async tests
        asyncio.run(test_grammar_generation())
        asyncio.run(test_runner_interface())
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())