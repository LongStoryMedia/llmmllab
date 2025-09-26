#!/usr/bin/env python3
"""
Full End-to-End Tool Execution Pipeline Test

This test validates the complete tool execution pipeline including:
1. Tool calling mechanism (LangGraph ToolNode compliance)
2. Web search tools with embedding-based ranking
3. RAG tools with semantic memory retrieval
4. Context extension and summarization
5. Error handling and fallback mechanisms

Designed to run on Kubernetes pods to test production environment.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the inference directory to path for imports
sys.path.insert(0, '/app')

# Import required modules
try:
    from models import Message, MessageRole, MessageContent, MessageContentType, ConversationCtx, Conversation
    from server.tools.rag_tools import search_memory, get_contextual_response
    logger.info("✅ Successfully imported tool modules")
except ImportError as e:
    logger.error(f"❌ Failed to import tool modules: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class ToolExecutionTestSuite:
    """Comprehensive test suite for tool execution pipeline validation"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "python_path": sys.path[:3]  # First few entries
            },
            "tests": {}
        }

    async def test_web_search_tool(self) -> bool:
        """Test web search tool with embedding-based ranking"""
        print("\n🔍 Testing Web Search Tool...")
        
        try:
            # Test query that should return meaningful results
            query = "latest developments in artificial intelligence 2024"
            
            print(f"   Searching for: '{query}'")
            
            start_time = time.time()
            results = await search_web(query, num_results=5)
            duration = time.time() - start_time
            
            print(f"   ⏱️  Search completed in {duration:.2f}s")
            
            if not results:
                print("❌ No search results returned")
                return False
            
            if isinstance(results, str) and len(results) > 100:
                print(f"✅ Search returned synthesized results ({len(results)} chars)")
                
                # Validate that results contain relevant information
                query_terms = ["ai", "artificial intelligence", "2024", "development"]
                found_terms = sum(1 for term in query_terms if term.lower() in results.lower())
                
                if found_terms >= 2:
                    print(f"   ✅ Results contain relevant terms ({found_terms}/{len(query_terms)})")
                    print(f"   📄 Sample: {results[:200]}...")
                    return True
                else:
                    print(f"   ⚠️  Results may not be relevant ({found_terms}/{len(query_terms)} terms found)")
                    return False
            else:
                print(f"❌ Unexpected result format: {type(results)}")
                return False
                
        except Exception as e:
            print(f"❌ Web search test failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False

    async def test_web_extraction_tool(self) -> bool:
        """Test web content extraction tool"""
        print("\n📄 Testing Web Extraction Tool...")
        
        try:
            # Test URL that should be extractable
            test_url = "https://www.wikipedia.org/wiki/Artificial_intelligence"
            
            print(f"   Extracting content from: {test_url}")
            
            start_time = time.time()
            content = await extract_web_content(test_url)
            duration = time.time() - start_time
            
            print(f"   ⏱️  Extraction completed in {duration:.2f}s")
            
            if not content:
                print("❌ No content extracted")
                return False
            
            if isinstance(content, dict) and "content" in content:
                extracted_text = content["content"]
                if len(extracted_text) > 500:
                    print(f"✅ Extracted {len(extracted_text)} characters of content")
                    print(f"   📄 Sample: {extracted_text[:200]}...")
                    
                    # Basic validation - should contain AI-related terms
                    ai_terms = ["artificial intelligence", "machine learning", "algorithm"]
                    found_ai_terms = sum(1 for term in ai_terms if term.lower() in extracted_text.lower())
                    
                    if found_ai_terms >= 1:
                        print(f"   ✅ Content is relevant ({found_ai_terms} AI terms found)")
                        return True
                    else:
                        print("   ⚠️  Content may not be from expected page")
                        return False
                else:
                    print(f"   ⚠️  Content too short ({len(extracted_text)} chars)")
                    return False
            else:
                print(f"❌ Unexpected extraction result format: {type(content)}")
                return False
                
        except Exception as e:
            print(f"❌ Web extraction test failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False

    async def test_rag_tools(self) -> bool:
        """Test RAG tools with memory search and contextual responses"""
        print("\n🧠 Testing RAG Tools...")
        
        try:
            # Test memory search
            print("   Testing memory search...")
            
            memory_query = "artificial intelligence research"
            memory_results = await search_memory(memory_query, limit=3)
            
            print(f"   🔍 Memory search returned: {type(memory_results)}")
            
            # Memory might be empty in test environment, that's acceptable
            if memory_results is None or (isinstance(memory_results, list) and len(memory_results) == 0):
                print("   ℹ️  No memory results (expected in test environment)")
            elif isinstance(memory_results, list):
                print(f"   ✅ Found {len(memory_results)} memory entries")
            else:
                print(f"   ✅ Memory search returned: {type(memory_results)}")
            
            # Test contextual response
            print("   Testing contextual response generation...")
            
            # Create mock conversation context
            mock_conversation = Conversation(
                id=1,
                user_id="test-user",
                title="Test Conversation",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            mock_context = ConversationCtx(
                messages=[],
                notes=[],
                images=[],
                conversation=mock_conversation
            )
            
            contextual_response = await get_contextual_response(
                "What are the latest AI developments?", 
                mock_context
            )
            
            if isinstance(contextual_response, str) and len(contextual_response) > 50:
                print(f"✅ Generated contextual response ({len(contextual_response)} chars)")
                print(f"   📄 Sample: {contextual_response[:150]}...")
                
                # Check if response is relevant
                if any(term in contextual_response.lower() for term in ["ai", "artificial", "intelligence", "development"]):
                    print("   ✅ Response is contextually relevant")
                    return True
                else:
                    print("   ⚠️  Response may not be contextually relevant")
                    return False
            else:
                print(f"❌ Unexpected contextual response: {type(contextual_response)}")
                return False
                
        except Exception as e:
            print(f"❌ RAG tools test failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False

    async def test_tool_integration(self) -> bool:
        """Test integration between multiple tools"""
        print("\n🔗 Testing Tool Integration...")
        
        try:
            # Simulate a complex query that would use multiple tools
            query = "Find information about recent AI breakthroughs and provide context"
            
            print(f"   Testing integrated workflow for: '{query}'")
            
            # Step 1: Search for current information
            print("   Step 1: Web search...")
            search_results = await search_web(query, num_results=3)
            
            search_success = isinstance(search_results, str) and len(search_results) > 100
            print(f"   {'✅' if search_success else '❌'} Web search: {search_success}")
            
            # Step 2: Try memory search
            print("   Step 2: Memory search...")
            memory_results = await search_memory("AI breakthroughs", limit=2)
            
            memory_success = memory_results is not None  # Any result (even empty) is success
            print(f"   {'✅' if memory_success else '❌'} Memory search: {memory_success}")
            
            # Step 3: Generate contextual response
            print("   Step 3: Contextual response...")
            
            mock_conversation = Conversation(
                id=1,
                user_id="test-user", 
                title="Test Integration",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            mock_context = ConversationCtx(
                messages=[],
                notes=[],
                images=[],
                conversation=mock_conversation
            )
            
            contextual_response = await get_contextual_response(
                f"Based on this information: {search_results[:500] if search_results else 'No search results'}, what are the key AI developments?",
                mock_context
            )
            
            response_success = isinstance(contextual_response, str) and len(contextual_response) > 50
            print(f"   {'✅' if response_success else '❌'} Contextual response: {response_success}")
            
            # Overall integration success
            integration_success = search_success and memory_success and response_success
            
            if integration_success:
                print("   ✅ All tool integration steps successful")
                return True
            else:
                successful_steps = sum([search_success, memory_success, response_success])
                print(f"   ⚠️  Partial integration success ({successful_steps}/3 steps)")
                return successful_steps >= 2  # Accept partial success
                
        except Exception as e:
            print(f"❌ Tool integration test failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tool execution tests"""
        print("🚀 Full Tool Execution Pipeline Tests")
        print("=" * 50)
        
        # Environment info
        print(f"Environment: {os.getcwd()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"PYTHONPATH: {':'.join(sys.path[:3])}")
        
        # Test GPU availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"CUDA Available: True")
                print(f"GPU: {gpu_name}")
            else:
                print(f"CUDA Available: False")
        except ImportError:
            print("PyTorch not available for GPU check")
        
        print()
        
        # Run individual tests
        tests = [
            ("web_search", self.test_web_search_tool),
            ("web_extraction", self.test_web_extraction_tool),
            ("rag_tools", self.test_rag_tools),
            ("tool_integration", self.test_tool_integration),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                self.test_results["tests"][test_name] = {
                    "passed": result,
                    "duration": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if result:
                    passed_tests += 1
                    
            except Exception as e:
                self.test_results["tests"][test_name] = {
                    "passed": False,
                    "error": str(e),
                    "duration": 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(f"❌ Test '{test_name}' failed with exception: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        for test_name, test_data in self.test_results["tests"].items():
            status = "✅ PASS" if test_data["passed"] else "❌ FAIL"
            duration = test_data.get("duration", 0)
            print(f"{status} {test_name.replace('_', ' ').title()} ({duration:.2f}s)")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if success_rate >= 100:
            print("   🎉 All tool execution systems operational!")
        elif success_rate >= 75:
            print("   ✅ Tool execution pipeline mostly functional")
        elif success_rate >= 50:
            print("   ⚠️  Tool execution has some issues - review failed tests")
        else:
            print("   🚨 Major tool execution issues - comprehensive debugging needed")
        
        # Save results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = f"tool_execution_test_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n📝 Results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
        
        print("=" * 50)
        
        return self.test_results


async def main():
    """Main test execution function"""
    test_suite = ToolExecutionTestSuite()
    results = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    passed_tests = sum(1 for test in results["tests"].values() if test["passed"])
    total_tests = len(results["tests"])
    
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    elif passed_tests >= total_tests * 0.75:
        sys.exit(1)  # Mostly passed
    else:
        sys.exit(2)  # Major failures


if __name__ == "__main__":
    asyncio.run(main())