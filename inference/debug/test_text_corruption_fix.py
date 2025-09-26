#!/usr/bin/env python3
"""
Test script to verify that text corruption issues (spaces being removed by .strip()) are fixed.
This specifically tests the user's reported issue with function call tags and text formatting.
"""

import sys
import os
sys.path.append('/app')

from server.main_pipeline_orchestrator import MainPipelineOrchestrator
from models.generate_req import GenerateReq
from models.pipeline_config import PipelineConfig
from models.message import Message
from models.message_source import MessageSource
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_text_preservation():
    """Test that models preserve spaces and formatting properly - no unwanted .strip() effects"""
    
    test_cases = [
        {
            "name": "Function Call Format Test",
            "prompt": "Please use a web search tool to find information about 'Python > JavaScript comparison'. Make sure to include the > symbol in your search query.",
            "expected_checks": [
                "Python > JavaScript",  # Should preserve the > symbol
                "web_search",  # Should use web search tool
                "function_call"  # Should have proper function calling
            ]
        },
        {
            "name": "Spacing Preservation Test", 
            "prompt": "Tell me about the difference between 'machine learning' and 'deep learning'. Use proper spacing.",
            "expected_checks": [
                "machine learning",  # Should preserve spaces
                "deep learning",     # Should preserve spaces
                "difference"         # Should contain the word
            ]
        },
        {
            "name": "Complex Symbol Test",
            "prompt": "Search for 'React.js >= 18.0 vs Vue.js <= 3.0' comparison using web search.",
            "expected_checks": [
                ">=",  # Should preserve >= symbol
                "<=",  # Should preserve <= symbol  
                "18.0",  # Should preserve version numbers
                "3.0"    # Should preserve version numbers
            ]
        }
    ]
    
    models_to_test = [
        "qwen3-30b-a3b-q4-k-m",
        "openai-gpt-oss-20b-uncensored-q5_1", 
        "qwen2.5-vl-32b-instruct-q4-k-m"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING TEXT PRESERVATION: {model_name}")
        logger.info(f"{'='*60}")
        
        results[model_name] = {}
        
        try:
            # Initialize pipeline
            orchestrator = MainPipelineOrchestrator()
            
            for test_case in test_cases:
                logger.info(f"\nRunning: {test_case['name']}")
                logger.info(f"Prompt: {test_case['prompt']}")
                
                # Create request
                request = GenerateReq(
                    model=model_name,
                    messages=[
                        Message(
                            role="user",
                            content=test_case['prompt'],
                            source=MessageSource.USER
                        )
                    ],
                    stream=False,
                    config=PipelineConfig(
                        enable_web_search=True,
                        enable_tool_calling=True,
                        max_tokens=2000,
                        temperature=0.1
                    )
                )
                
                # Generate response
                try:
                    response = await orchestrator.generate(request)
                    
                    # Check response
                    if hasattr(response, 'choices') and response.choices:
                        content = response.choices[0].message.content
                        logger.info(f"Response length: {len(content)} chars")
                        logger.info(f"Response preview: {content[:200]}...")
                        
                        # Check for expected elements
                        checks_passed = 0
                        total_checks = len(test_case['expected_checks'])
                        
                        for check in test_case['expected_checks']:
                            if check.lower() in content.lower():
                                checks_passed += 1
                                logger.info(f"✅ Found: '{check}'")
                            else:
                                logger.warning(f"❌ Missing: '{check}'")
                        
                        success_rate = (checks_passed / total_checks) * 100
                        results[model_name][test_case['name']] = {
                            'success_rate': success_rate,
                            'checks_passed': checks_passed,
                            'total_checks': total_checks,
                            'content_length': len(content),
                            'status': 'PASS' if success_rate >= 50 else 'FAIL'
                        }
                        
                        logger.info(f"Result: {success_rate:.1f}% ({checks_passed}/{total_checks}) - {results[model_name][test_case['name']]['status']}")
                        
                    else:
                        logger.error(f"No response content generated")
                        results[model_name][test_case['name']] = {
                            'status': 'ERROR',
                            'error': 'No response generated'
                        }
                        
                except Exception as e:
                    logger.error(f"Error during generation: {e}")
                    results[model_name][test_case['name']] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to initialize pipeline for {model_name}: {e}")
            results[model_name] = {'initialization_error': str(e)}
    
    # Summary report
    logger.info(f"\n{'='*80}")
    logger.info("TEXT PRESERVATION TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    total_models = len(models_to_test)
    working_models = 0
    
    for model_name, model_results in results.items():
        logger.info(f"\n{model_name}:")
        
        if 'initialization_error' in model_results:
            logger.info(f"  ❌ INITIALIZATION FAILED: {model_results['initialization_error']}")
            continue
            
        model_working = True
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_result in model_results.items():
            total_tests += 1
            if test_result.get('status') == 'PASS':
                passed_tests += 1
                logger.info(f"  ✅ {test_name}: {test_result['success_rate']:.1f}%")
            elif test_result.get('status') == 'FAIL':
                logger.info(f"  ❌ {test_name}: {test_result['success_rate']:.1f}%")
                model_working = False
            else:
                logger.info(f"  ⚠️  {test_name}: ERROR - {test_result.get('error', 'Unknown')}")
                model_working = False
        
        if model_working and total_tests > 0:
            working_models += 1
            logger.info(f"  🎉 OVERALL: WORKING ({passed_tests}/{total_tests} tests passed)")
        else:
            logger.info(f"  💥 OVERALL: NEEDS DEBUGGING ({passed_tests}/{total_tests} tests passed)")
    
    logger.info(f"\nFINAL SUMMARY: {working_models}/{total_models} models working properly")
    logger.info("✅ Text corruption (.strip()) issue status: " + ("FIXED" if working_models >= 2 else "NEEDS MORE WORK"))
    
    return results

if __name__ == "__main__":
    asyncio.run(test_text_preservation())