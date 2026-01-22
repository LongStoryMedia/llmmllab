#!/usr/bin/env python3
"""Test the improved grammar generator implementation."""

import sys
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_grammar_generator():
    """Test the improved grammar generator with various Pydantic models."""

    try:
        from utils.grammar_generator import get_grammar_for_model, pydantic_to_grammar

        logger.info("✅ Successfully imported grammar generator functions")
    except Exception as e:
        logger.error(f"❌ Failed to import grammar generator: {e}")
        return False

    # Test model 1: Simple object
    class Person(BaseModel):
        name: str = Field(description="Person's name")
        age: int = Field(description="Person's age", ge=0, le=150)
        email: Optional[str] = Field(None, description="Email address")

    # Test model 2: Complex nested object
    class Address(BaseModel):
        street: str
        city: str
        country: str = "USA"
        zip_code: Optional[str] = None

    class Contact(BaseModel):
        person: Person
        address: Address
        tags: List[str] = Field(default_factory=list)
        is_active: bool = True

    # Test model 3: Model with enum-like constraints
    class TaskStatus(BaseModel):
        status: str = Field(pattern="^(pending|in_progress|completed|failed)$")
        priority: int = Field(ge=1, le=5)
        notes: Optional[str] = None

    test_models = [("Person", Person), ("Contact", Contact), ("TaskStatus", TaskStatus)]

    results = {}

    for model_name, model_class in test_models:
        logger.info(f"\n=== Testing {model_name} ===")

        try:
            # Test grammar generation
            grammar = pydantic_to_grammar(model_class)

            if grammar:
                logger.info(f"✅ {model_name}: Grammar generated successfully")
                logger.info(f"   Grammar length: {len(grammar)} characters")
                logger.info(f"   Rules count: {len(grammar.split('::='))}")

                # Show first few lines of grammar
                lines = grammar.split("\n")[:5]
                logger.info(f"   First rules: {lines}")

                results[model_name] = "SUCCESS - Grammar generated"
            else:
                logger.error(f"❌ {model_name}: Empty grammar generated")
                results[model_name] = "FAILED - Empty grammar"

        except Exception as e:
            logger.error(f"❌ {model_name}: Grammar generation failed - {e}")
            results[model_name] = f"FAILED - Exception: {e}"

    # Test the convenience function
    try:
        logger.info("\n=== Testing convenience function ===")
        grammar = get_grammar_for_model(Person, use_cache=False)
        if grammar:
            logger.info("✅ Convenience function works correctly")
            results["convenience_function"] = "SUCCESS"
        else:
            logger.error("❌ Convenience function returned empty grammar")
            results["convenience_function"] = "FAILED - Empty result"
    except Exception as e:
        logger.error(f"❌ Convenience function failed: {e}")
        results["convenience_function"] = f"FAILED - {e}"

    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    success_count = 0
    for test_name, result in results.items():
        status = "✅" if "SUCCESS" in result else "❌"
        logger.info(f"{status} {test_name}: {result}")
        if "SUCCESS" in result:
            success_count += 1

    logger.info(f"\nResults: {success_count}/{len(results)} tests successful")

    return success_count == len(results)


if __name__ == "__main__":
    success = test_grammar_generator()
    sys.exit(0 if success else 1)
