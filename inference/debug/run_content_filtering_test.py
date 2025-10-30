#!/usr/bin/env python3
"""
Simple Content Filtering Test Runner

This script runs the full E2E test to validate content filtering fixes.
It focuses specifically on validating the streaming responses.
"""

import asyncio
import sys

async def run_content_filtering_test():
    """Run the E2E test to validate content filtering."""
    print("🧪 Running Content Filtering Validation via E2E Test")
    print("=" * 60)
    
    try:
        # Import the E2E tester
        from debug.test_composer_real_e2e import ChatCompletionE2ETester
        
        # Create tester with content validation focus
        tester = ChatCompletionE2ETester(
            target_model="qwen3-30b-a3b-q4-k-m",
            capture_llm_output=True,
            print_output=False,  # Keep output manageable
            server_url="http://localhost:8000"
        )
        
        # Run test with a query that should trigger content filtering validation
        query = "What are the latest AI developments in October 2025? Please search for recent news and give me a comprehensive analysis."
        
        print(f"📝 Query: {query[:80]}...")
        
        # Run the test
        results = await tester.run_full_test(query=query)
        
        # Extract content filtering specific results
        chat_completion_results = results["results"].get("chat_completion_execution", {})
        
        if chat_completion_results.get("success"):
            content_issues = chat_completion_results.get("content_issues", [])
            
            print("\n📊 Content Filtering Results:")
            print(f"   Streaming chunks: {chat_completion_results.get('streaming_chunks', 0)}")
            print(f"   Content length: {chat_completion_results.get('content_length', 0)}")
            print(f"   Tool calls: {chat_completion_results.get('tool_calls_count', 0)}")
            print(f"   Unknown tools: {chat_completion_results.get('unknown_tool_count', 0)}")
            
            if content_issues:
                print(f"\n❌ Content Filtering Issues Found ({len(content_issues)}):")
                for issue in content_issues:
                    issue_type = issue.get('issue', 'unknown')
                    print(f"   • {issue_type}")
                return False
            else:
                print("\n✅ Content Filtering Validation PASSED!")
                print("   • No intent analysis JSON leaked")
                print("   • No thoughts leaked into main content") 
                print("   • No unknown tool names")
                print("   • No serialized Pydantic objects")
                print("   • Correct date context")
                return True
        else:
            print(f"\n❌ Chat completion failed: {chat_completion_results.get('validation_errors', [])}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await run_content_filtering_test()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 CONTENT FILTERING VALIDATION PASSED!")
        print("Recent fixes are working correctly.")
    else:
        print("⚠️  CONTENT FILTERING VALIDATION FAILED!")
        print("Issues detected - fixes need investigation.")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)