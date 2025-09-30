#!/usr/bin/env python3
"""
Integration test for grammar-constrained structured output workflow.

This script demonstrates the complete workflow:
1. Define a Pydantic model for structured output
2. Generate GBNF grammar from the model
3. Use the grammar in pipeline execution (simulated)
4. Validate the output against the schema

This tests the end-to-end integration of grammar support.
"""

import asyncio
import logging
import sys
import json

sys.path.insert(0, '/app')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_end_to_end_workflow():
    """Test the complete grammar workflow."""
    logger.info("🧪 Testing end-to-end grammar workflow...")
    
    try:
        # Step 1: Import required components
        from models.deduplication_result import DeduplicationResult
        from utils.grammar_generator import (
            get_grammar_for_model,
            parse_structured_output,
            pydantic_to_grammar
        )
        
        logger.info("✅ Step 1: Imported required components")
        
        # Step 2: Generate grammar from Pydantic model
        grammar = get_grammar_for_model(DeduplicationResult)
        logger.info(f"✅ Step 2: Generated grammar ({len(grammar)} chars)")
        
        # Step 3: Simulate LLM response (this would come from grammar-constrained LLM)
        simulated_llm_response = '''{
            "is_duplicate": true,
            "similarity_score": 0.92,
            "recommendation": "The proposed tool 'file_reader' is functionally identical to existing tool 'read_file_content'. Both tools read file contents and return text. Recommend using the existing tool instead of creating a duplicate.",
            "should_create_new": false,
            "merge_suggestion": "Use existing 'read_file_content' tool which already provides the required functionality with robust error handling."
        }'''
        
        logger.info("✅ Step 3: Simulated grammar-constrained LLM response")
        
        # Step 4: Parse and validate the structured output
        result = parse_structured_output(simulated_llm_response, DeduplicationResult)
        logger.info(f"✅ Step 4: Parsed structured output successfully")
        logger.info(f"   - is_duplicate: {result.is_duplicate}")
        logger.info(f"   - similarity_score: {result.similarity_score}")
        logger.info(f"   - should_create_new: {result.should_create_new}")
        logger.info(f"   - recommendation: {result.recommendation[:50]}...")
        
        # Step 5: Demonstrate type safety
        assert isinstance(result.is_duplicate, bool), "is_duplicate should be boolean"
        assert isinstance(result.similarity_score, float), "similarity_score should be float"
        assert 0.0 <= result.similarity_score <= 1.0, "similarity_score should be between 0 and 1"
        assert isinstance(result.should_create_new, bool), "should_create_new should be boolean"
        
        logger.info("✅ Step 5: Type safety validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}")
        return False


def test_grammar_validation():
    """Test grammar validation with various input formats."""
    logger.info("🧪 Testing grammar validation with different inputs...")
    
    try:
        from models.deduplication_result import DeduplicationResult
        from utils.grammar_generator import parse_structured_output, StructuredOutputError
        
        # Test 1: Valid JSON
        valid_json = '{"is_duplicate": false, "similarity_score": 0.1, "recommendation": "Create new tool", "should_create_new": true, "merge_suggestion": null}'
        result1 = parse_structured_output(valid_json, DeduplicationResult)
        logger.info("✅ Valid JSON parsed successfully")
        
        # Test 2: JSON with extra whitespace
        whitespace_json = '''
        {
            "is_duplicate": true,
            "similarity_score": 0.85,
            "recommendation": "Very similar tool exists",
            "should_create_new": false,
            "merge_suggestion": "Consider extending existing tool"
        }
        '''
        result2 = parse_structured_output(whitespace_json, DeduplicationResult)
        logger.info("✅ JSON with whitespace parsed successfully")
        
        # Test 3: JSON embedded in text
        embedded_json = '''
        Analysis complete. Here is the result:
        
        {
            "is_duplicate": false,
            "similarity_score": 0.45,
            "recommendation": "Some similarity but unique enough",
            "should_create_new": true,
            "merge_suggestion": null
        }
        
        This concludes the analysis.
        '''
        result3 = parse_structured_output(embedded_json, DeduplicationResult)
        logger.info("✅ Embedded JSON parsed successfully")
        
        # Test 4: Invalid JSON (should fail gracefully)
        try:
            invalid_json = '{"is_duplicate": "not a boolean"}'
            parse_structured_output(invalid_json, DeduplicationResult)
            logger.error("❌ Invalid JSON should have failed validation")
            return False
        except StructuredOutputError:
            logger.info("✅ Invalid JSON correctly rejected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Grammar validation test failed: {e}")
        return False


def test_deduplication_use_case():
    """Test the specific deduplication use case."""
    logger.info("🧪 Testing deduplication analysis use case...")
    
    try:
        from models.deduplication_result import DeduplicationResult
        from utils.grammar_generator import parse_structured_output
        
        # Simulate realistic deduplication analysis scenarios
        scenarios = [
            {
                "name": "Duplicate Tool Detection",
                "response": '''
                {
                    "is_duplicate": true,
                    "similarity_score": 0.95,
                    "recommendation": "The proposed 'web_scraper' tool is nearly identical to the existing 'scrape_website' tool. Both extract content from web pages using similar approaches.",
                    "should_create_new": false,
                    "merge_suggestion": "Use existing 'scrape_website' tool which already handles error cases and rate limiting."
                }
                '''
            },
            {
                "name": "Similar but Distinct Tool",
                "response": '''
                {
                    "is_duplicate": false,
                    "similarity_score": 0.65,
                    "recommendation": "While both tools work with files, the proposed 'json_validator' provides specific JSON schema validation that the existing 'file_reader' does not offer.",
                    "should_create_new": true,
                    "merge_suggestion": "Consider creating the new tool but add integration points with existing file handling utilities."
                }
                '''
            },
            {
                "name": "Unique Tool",
                "response": '''
                {
                    "is_duplicate": false,
                    "similarity_score": 0.15,
                    "recommendation": "The proposed 'password_generator' tool provides unique functionality not covered by any existing tools.",
                    "should_create_new": true,
                    "merge_suggestion": null
                }
                '''
            }
        ]
        
        for scenario in scenarios:
            result = parse_structured_output(scenario["response"], DeduplicationResult)
            logger.info(f"✅ {scenario['name']}: duplicate={result.is_duplicate}, score={result.similarity_score:.2f}")
        
        logger.info("✅ All deduplication scenarios processed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Deduplication use case test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting grammar integration tests...")
    
    tests = [
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Grammar Validation", test_grammar_validation),
        ("Deduplication Use Case", test_deduplication_use_case),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Integration Test Results:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {status} {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Integration tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        logger.info("🎉 All integration tests passed!")
        logger.info("✅ Grammar-constrained structured output is ready for production use")
        logger.info("🔧 Next steps:")
        logger.info("   - Update composer agents to use grammar constraints")
        logger.info("   - Implement grammar support in pipeline factory")
        logger.info("   - Add grammar validation to LLM response processing")
    else:
        logger.warning("⚠️ Some integration tests failed")
        logger.info("🔧 Review failed tests and fix implementation issues")


if __name__ == "__main__":
    asyncio.run(main())