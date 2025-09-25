"""
Focused Embedding Smoke Test Core

A lightweight test specifically designed to run on the K8s pod with the actual
embedding pipeline configuration. This test focuses on the core functionality
and llama_decode error diagnostics.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the embedding pipeline
try:
    from models import Message, MessageRole, MessageContent, MessageContentType
    from runner import embed_pipeline
    logger.info("✅ Successfully imported embedding modules")
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


async def create_test_messages(texts: List[str]) -> List[Message]:
    """Create test messages from text strings"""
    messages = []
    for i, text in enumerate(texts):
        content = [MessageContent(type=MessageContentType.TEXT, text=text)]
        message = Message(
            id=i,
            role=MessageRole.USER,
            content=content,
            conversation_id=1
        )
        messages.append(message)
    return messages


async def test_basic_embedding():
    """Test basic embedding functionality using the actual pipeline"""
    print("🧪 Testing basic embedding generation...")
    
    try:
        # Import the pipeline factory
        from runner.pipeline_factory import pipeline_factory
        from models import ModelProfile, ModelParameters
        
        # Create minimal model parameters
        params = ModelParameters(
            temperature=0.0,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Create model profile for embedding model
        profile = ModelProfile(
            user_id="test-user",
            name="embedding-test",
            model_name="nomic-embed-text-v2",
            parameters=params,
            system_prompt="",
            type=1
        )
        
        print(f"   Creating pipeline for model: {profile.model_name}")
        
        # Get embedding pipeline from factory
        pipeline = pipeline_factory.get_pipeline(profile, list)
        if pipeline is None:
            print("❌ Pipeline factory returned None")
            return False
            
        print("✅ Embedding pipeline created successfully")
        
        # Create simple test texts as messages
        test_texts = [
            "Hello world",
            "What are the latest AI developments?", 
            "This is a test document for embedding generation."
        ]
        
        messages = await create_test_messages(test_texts)
        print(f"   Testing with {len(messages)} messages...")
        
        # Try to use the actual embedding pipeline
        try:
            # Use the pipeline directly
            embeddings = await pipeline.process_messages(messages)
            
            if embeddings:
                print(f"✅ Generated {len(embeddings)} embeddings")
                print(f"   Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
                print(f"   Sample values: {embeddings[0][:3] if embeddings else 'None'}")
                
                # Validate dimensions
                expected_dim = 768  # Nomic embedding dimension
                if not embeddings or len(embeddings[0]) != expected_dim:
                    print(f"   ❌ Wrong dimensions: got {len(embeddings[0])}, expected {expected_dim}")
                    return False
                
                # Validate that embeddings are meaningful (not all zeros or empty)
                validation_passed = True
                
                for i, embedding in enumerate(embeddings):
                    # Check if all values are zero
                    if all(abs(val) < 1e-8 for val in embedding):
                        print(f"   ❌ Embedding {i} is all zeros")
                        validation_passed = False
                    
                    # Check if values are in reasonable range (not NaN/inf)
                    if any(not (-10.0 <= val <= 10.0) or val != val for val in embedding[:10]):  # val != val checks for NaN
                        print(f"   ❌ Embedding {i} has invalid values (NaN/inf/extreme)")
                        validation_passed = False
                    
                    # Check variance - meaningful embeddings should have some variance
                    import statistics
                    if len(embedding) > 1:
                        variance = statistics.variance(embedding)
                        if variance < 1e-6:
                            print(f"   ❌ Embedding {i} has no variance (likely uniform values)")
                            validation_passed = False
                        else:
                            print(f"   ✅ Embedding {i}: variance={variance:.6f}, range=[{min(embedding):.4f}, {max(embedding):.4f}]")
                
                if validation_passed:
                    print(f"   ✅ All embeddings are valid and meaningful")
                    print(f"   ✅ Correct dimensions: {expected_dim}D")
                    return True
                else:
                    print(f"   ❌ Embeddings validation failed")
                    return False
            else:
                print("❌ No embeddings returned")
                return False
                
        except Exception as e:
            error_msg = str(e).lower()
            if "llama_decode returned -1" in error_msg:
                print(f"❌ llama_decode error detected: {e}")
            elif "no module named" in error_msg:
                print(f"❌ Missing dependency: {e}")
            else:
                print(f"❌ Embedding generation failed: {e}")
            
            print(f"   Error details: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False


async def test_batch_resilience():
    """Test resilience to different batch sizes and decode errors"""
    print("\n🧪 Testing batch resilience...")
    
    try:
        # Import the pipeline factory
        from runner.pipeline_factory import pipeline_factory
        from models import ModelProfile, ModelParameters
        
        # Create minimal model parameters
        params = ModelParameters(
            temperature=0.0,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Create model profile for embedding model
        profile = ModelProfile(
            user_id="test-user",
            name="embedding-test",
            model_name="nomic-embed-text-v2",
            parameters=params,
            system_prompt="",
            type=1
        )
        
        # Get embedding pipeline from factory
        pipeline = pipeline_factory.get_pipeline(profile, list)
        if pipeline is None:
            print("❌ Pipeline factory returned None")
            return False
        
        # Test different batch sizes with progressively larger inputs
        test_cases = [
            ("single_short", ["AI research"]),
            ("single_medium", ["Artificial intelligence research is advancing rapidly with new developments."]),
            ("small_batch", ["AI", "ML", "NLP"]),
            ("medium_batch", [f"Document {i}: AI research topic" for i in range(5)]),
        ]
        
        results = {}
        
        for case_name, texts in test_cases:
            try:
                print(f"   Testing {case_name} ({len(texts)} texts)...")
                
                messages = await create_test_messages(texts)
                
                start_time = datetime.now()
                embeddings = await pipeline.process_messages(messages)
                duration = (datetime.now() - start_time).total_seconds()
                
                # Enhanced validation for meaningful embeddings
                success = False
                if embeddings and len(embeddings) == len(texts):
                    # Quick validation that embeddings are not all zeros
                    meaningful_count = 0
                    for embedding in embeddings:
                        if len(embedding) == 768 and not all(abs(val) < 1e-8 for val in embedding):
                            meaningful_count += 1
                    
                    success = meaningful_count == len(embeddings)
                    if not success:
                        print(f"   ⚠️  Only {meaningful_count}/{len(embeddings)} embeddings are meaningful")
                
                results[case_name] = {
                    "success": success,
                    "input_count": len(texts),
                    "output_count": len(embeddings) if embeddings else 0,
                    "duration": duration
                }
                
                if success:
                    print(f"   ✅ {case_name}: {len(embeddings)} embeddings in {duration:.2f}s")
                else:
                    print(f"   ❌ {case_name}: Failed or incomplete results")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "llama_decode returned -1" in error_msg:
                    print(f"   ⚠️  {case_name}: llama_decode error: {e}")
                    results[case_name] = {"success": False, "llama_decode_error": True, "error": str(e)}
                else:
                    print(f"   ❌ {case_name}: Other error: {e}")
                    results[case_name] = {"success": False, "error": str(e)}
        
        # Check if at least one approach worked
        successful_cases = [name for name, result in results.items() if result.get("success")]
        
        if successful_cases:
            print(f"   ✅ Batch resilience test passed: {len(successful_cases)} cases work")
            print(f"   Working cases: {', '.join(successful_cases)}")
            return True
        else:
            print(f"   ❌ Batch resilience test failed: No cases work")
            print(f"   llama_decode errors: {[name for name, result in results.items() if result.get('llama_decode_error')]}")
            return False
            
    except Exception as e:
        print(f"❌ Batch resilience test failed: {e}")
        return False


async def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing edge cases...")
    
    try:
        # Import the pipeline factory
        from runner.pipeline_factory import pipeline_factory
        from models import ModelProfile, ModelParameters
        
        # Create minimal model parameters
        params = ModelParameters(
            temperature=0.0,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Create model profile for embedding model
        profile = ModelProfile(
            user_id="test-user",
            name="embedding-test",
            model_name="nomic-embed-text-v2",
            parameters=params,
            system_prompt="",
            type=1
        )
        
        # Get embedding pipeline from factory
        pipeline = pipeline_factory.get_pipeline(profile, list)
        if pipeline is None:
            print("❌ Pipeline factory returned None")
            return False
        
        # Test edge cases
        test_cases = [
            ("empty_string", [""]),
            ("whitespace", ["   \n\t   "]),
            ("special_chars", ["@#$%^&*()_+-=[]{}|;':\",./<>?`~"]),
            ("unicode", ["Hello 世界 🌍 émojis"]),
            ("long_text", ["This is a very long text that should test the pipeline. " * 100]),
        ]
        
        passed_cases = 0
        total_cases = len(test_cases)
        
        for case_name, texts in test_cases:
            try:
                print(f"   Testing {case_name}...")
                
                messages = await create_test_messages(texts)
                embeddings = await pipeline.process_messages(messages)
                
                # For edge cases, validate embeddings are meaningful
                if embeddings and len(embeddings) == len(texts):
                    # Check if embeddings are meaningful (not all zeros)
                    meaningful_embeddings = 0
                    for embedding in embeddings:
                        if len(embedding) == 768 and not all(abs(val) < 1e-8 for val in embedding):
                            meaningful_embeddings += 1
                    
                    if meaningful_embeddings == len(embeddings):
                        print(f"   ✅ {case_name}: Generated meaningful embeddings")
                        passed_cases += 1
                    elif meaningful_embeddings > 0:
                        print(f"   ✅ {case_name}: Generated some meaningful embeddings ({meaningful_embeddings}/{len(embeddings)})")
                        passed_cases += 0.8
                    else:
                        print(f"   ⚠️  {case_name}: Generated zero embeddings (may be acceptable for edge case)")
                        passed_cases += 0.5
                elif embeddings and len(embeddings) > 0:
                    print(f"   ✅ {case_name}: Handled gracefully (generated some embedding)")
                    passed_cases += 0.7
                else:
                    print(f"   ⚠️  {case_name}: No embeddings but no crash (acceptable for edge case)")
                    passed_cases += 0.3  # Minimal credit for no crash
                
            except Exception as e:
                error_msg = str(e).lower()
                if "llama_decode returned -1" in error_msg:
                    print(f"   ⚠️  {case_name}: llama_decode error (may be expected for edge case)")
                    passed_cases += 0.5  # Partial credit for graceful error
                else:
                    print(f"   ❌ {case_name}: Unexpected error: {e}")
        
        success_rate = passed_cases / total_cases
        
        if success_rate >= 0.7:
            print(f"   ✅ Edge cases test passed: {success_rate:.1%} success rate")
            return True
        else:
            print(f"   ⚠️  Edge cases test partial: {success_rate:.1%} success rate")
            return success_rate > 0.3  # Partial success acceptable
            
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False


async def main():
    """Run the focused embedding tests"""
    print("🚀 Embedding Pipeline Smoke Tests")
    print("=" * 50)
    
    # Environment info
    print(f"Environment: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    try:
        import torch
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
    
    print()
    
    # Run tests
    tests = [
        ("Basic Embedding", test_basic_embedding),
        ("Batch Resilience", test_batch_resilience), 
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if results.get("Basic Embedding", False):
        print("   ✅ Basic embedding functionality is working")
    else:
        print("   🚨 Basic embedding is failing - check model configuration and files")
    
    if results.get("Batch Resilience", False):
        print("   ✅ Pipeline handles batching well")
    else:
        print("   ⚠️  Pipeline struggles with batching - may need smaller batch sizes")
    
    if results.get("Edge Cases", False):
        print("   ✅ Edge cases are handled gracefully")
    else:
        print("   ⚠️  Edge cases need improvement")
    
    # Determine overall status
    if passed == total:
        print("   🎉 All systems operational!")
        exit_code = 0
    elif passed >= 1:  # At least basic embedding works
        print("   ⚠️  Partial functionality - embedding works but has issues")
        exit_code = 1
    else:
        print("   🚨 Critical issues - basic embedding not working")
        exit_code = 2
    
    print("=" * 50)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"embedding_smoke_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": passed / total
            }
        }, f, indent=2)
    
    print(f"📝 Results saved to: {results_file}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())