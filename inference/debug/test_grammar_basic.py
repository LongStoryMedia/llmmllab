#!/usr/bin/env python3
"""
Simplified test script for grammar generation utility.

This script tests only the grammar generation components without requiring
the full runner pipeline infrastructure.
"""

import logging
import sys
import os

# Ensure PYTHONPATH includes the app directory
sys.path.insert(0, '/app')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_grammar_utility_imports():
    """Test that grammar utility can be imported."""
    try:
        from utils.grammar_generator import (
            pydantic_to_grammar, 
            get_grammar_for_model, 
            parse_structured_output,
            StructuredOutputError
        )
        logger.info("✅ Grammar utility imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Grammar utility import failed: {e}")
        return False


def test_pydantic_model_import():
    """Test that we can import Pydantic models."""
    try:
        from models.deduplication_result import DeduplicationResult
        logger.info(f"✅ DeduplicationResult imported: {DeduplicationResult}")
        return True, DeduplicationResult
    except Exception as e:
        logger.error(f"❌ DeduplicationResult import failed: {e}")
        return False, None


def test_grammar_generation(model_class):
    """Test grammar generation for a Pydantic model."""
    try:
        from utils.grammar_generator import pydantic_to_grammar
        
        grammar = pydantic_to_grammar(model_class)
        logger.info(f"✅ Grammar generated: {len(grammar)} characters")
        logger.info(f"Grammar preview:\n{grammar[:300]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Grammar generation failed: {e}")
        return False


def test_structured_output_parsing(model_class):
    """Test parsing structured output."""
    try:
        from utils.grammar_generator import parse_structured_output
        
        # Sample valid JSON matching DeduplicationResult schema
        test_json = '''
        {
            "is_duplicate": false,
            "similarity_score": 0.35,
            "recommendation": "Tool has unique functionality, recommend creating new tool",
            "should_create_new": true,
            "merge_suggestion": null
        }
        '''
        
        result = parse_structured_output(test_json, model_class)
        logger.info(f"✅ Structured output parsed: {result}")
        logger.info(f"   - is_duplicate: {result.is_duplicate}")
        logger.info(f"   - similarity_score: {result.similarity_score}")
        logger.info(f"   - should_create_new: {result.should_create_new}")
        return True
    except Exception as e:
        logger.error(f"❌ Structured output parsing failed: {e}")
        return False


def main():
    """Run simplified grammar tests."""
    logger.info("🧪 Starting simplified grammar tests...")
    
    # Test 1: Import grammar utility
    logger.info("\n📋 Test 1: Grammar Utility Imports")
    if not test_grammar_utility_imports():
        logger.error("❌ Basic imports failed, aborting")
        return
    
    # Test 2: Import Pydantic model
    logger.info("\n📋 Test 2: Pydantic Model Import")
    success, model_class = test_pydantic_model_import()
    if not success:
        logger.error("❌ Model import failed, aborting")
        return
    
    # Test 3: Grammar generation
    logger.info("\n📋 Test 3: Grammar Generation")
    if not test_grammar_generation(model_class):
        logger.error("❌ Grammar generation failed")
        return
    
    # Test 4: Structured output parsing
    logger.info("\n📋 Test 4: Structured Output Parsing")
    if not test_structured_output_parsing(model_class):
        logger.error("❌ Structured output parsing failed")
        return
    
    logger.info("\n🎉 All simplified tests passed!")
    logger.info("✅ Grammar generation utility is working correctly")


if __name__ == "__main__":
    main()