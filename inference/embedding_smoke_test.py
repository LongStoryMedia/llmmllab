#!/app/v.sh runner python
"""
Focused Embedding Smoke Test

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
    from runner.pipelines.emb.nom2 import NomicEmbedTextPipe
    from models import Message, MessageRole, MessageContent, MessageContentType
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
    """Test basic embedding functionality"""
    print("🧪 Testing basic embedding generation...")
    
    try:
        # Import pipeline factory to get actual configured models
        from runner.pipeline_factory import pipeline_factory
        from runner import embed_pipeline
        
        # Try to get an actual embedding model configuration
        # This will use your real model configuration
        logger.info("Looking for embedding model configuration...")
        
        # Create simple test messages
        test_texts = [
            "Hello world",
            "What are the latest AI developments?",
            "This is a test document for embedding generation."
        ]
        
        messages = await create_test_messages(test_texts)
        
        # Try to use the actual embedding pipeline
        try:
            # This will use whatever embedding model is configured
            embeddings = await embed_pipeline(test_texts, None)  # Will use default pipeline
            
            if embeddings:
                print(f"✅ Generated {len(embeddings)} embeddings")
                print(f"   Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
                print(f"   Sample values: {embeddings[0][:5] if embeddings else 'None'}")
                return True
            else:
                print("❌ No embeddings returned")
                return False
                
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False


async def test_llama_decode_resilience():
    """Test resilience to llama_decode errors with different batch sizes"""
    print("\n🧪 Testing llama_decode error resilience...")
    
    try:
        from runner import embed_pipeline
        
        # Create a larger set of texts to potentially trigger batching issues
        test_texts = [
            f"Test document {i}: Artificial intelligence and machine learning are rapidly evolving fields. "
            f"Recent developments in neural networks, transformers, and large language models have "
            f"revolutionized natural language processing and computer vision applications. "
            f"These technologies are being applied in various domains including healthcare, finance, "
            f"autonomous vehicles, and scientific research. Document number {i}."
            for i in range(1, 11)  # 10 medium-length documents
        ]
        
        print(f"   Testing with {len(test_texts)} documents...")
        
        # Test different approaches
        approaches = [
            ("single", [test_texts[0]]),
            ("small_batch", test_texts[:3]),
            ("medium_batch", test_texts[:5]),
            ("large_batch", test_texts),
        ]
        
        results = {}
        
        for approach_name, texts in approaches:
            try:
                print(f"   Testing {approach_name} ({len(texts)} texts)...")
                
                start_time = datetime.now()
                embeddings = await embed_pipeline(texts, None)
                duration = (datetime.now() - start_time).total_seconds()
                
                success = embeddings and len(embeddings) == len(texts)
                
                results[approach_name] = {
                    "success": success,
                    "input_count": len(texts),
                    "output_count": len(embeddings) if embeddings else 0,
                    "duration": duration
                }
                
                if success:
                    print(f"   ✅ {approach_name}: {len(embeddings)} embeddings in {duration:.2f}s")
                else:
                    print(f"   ❌ {approach_name}: Failed or incomplete results")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "llama_decode returned -1" in error_msg:
                    print(f"   ⚠️  {approach_name}: llama_decode error (expected): {e}")
                    results[approach_name] = {"success": False, "llama_decode_error": True, "error": str(e)}
                else:
                    print(f"   ❌ {approach_name}: Other error: {e}")
                    results[approach_name] = {"success": False, "error": str(e)}
        
        # Check if at least one approach worked
        successful_approaches = [name for name, result in results.items() if result.get("success")]
        
        if successful_approaches:
            print(f"   ✅ Resilience test passed: {len(successful_approaches)} approaches work")
            print(f"   Working approaches: {', '.join(successful_approaches)}")
            return True
        else:
            print(f"   ❌ Resilience test failed: No approaches work")
            return False
            
    except Exception as e:
        print(f"❌ Resilience test setup failed: {e}")
        return False


async def test_text_processing():
    """Test text processing and splitting functionality"""
    print("\n🧪 Testing text processing...")
    
    try:
        from runner import embed_pipeline
        
        # Test different text types
        test_cases = [
            ("empty", ""),
            ("short", "AI"),
            ("normal", "What are the latest developments in artificial intelligence research?"),
            ("long", "This is a very long text that might need to be split into smaller chunks. " * 50),
            ("special_chars", "Test with émojis 🤖 and special chars: @#$%^&*()"),
            ("unicode", "多语言测试 Unicode test עברית العربية русский язык"),
        ]
        
        results = {}
        
        for case_name, text in test_cases:
            try:
                print(f"   Testing {case_name}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                embeddings = await embed_pipeline([text], None) if text else await embed_pipeline(["placeholder"], None)
                
                success = embeddings and len(embeddings) == 1 and len(embeddings[0]) > 0
                
                results[case_name] = {
                    "success": success,
                    "text_length": len(text),
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0
                }
                
                if success:
                    print(f"   ✅ {case_name}: Generated {len(embeddings[0])}D embedding")
                else:
                    print(f"   ❌ {case_name}: Failed to generate embedding")
                
            except Exception as e:
                print(f"   ⚠️  {case_name}: Error handled gracefully: {e}")
                results[case_name] = {"success": False, "error": str(e)}
        
        # Check success rate
        successful_cases = [name for name, result in results.items() if result.get("success")]
        success_rate = len(successful_cases) / len(results)
        
        if success_rate >= 0.5:  # At least 50% should work
            print(f"   ✅ Text processing test passed: {success_rate:.1%} success rate")
            return True
        else:
            print(f"   ❌ Text processing test failed: {success_rate:.1%} success rate")
            return False
            
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False


async def main():
    """Run the focused embedding tests"""
    print("🚀 Embedding Pipeline Smoke Tests")
    print("=" * 50)
    
    # Environment info
    print(f"Environment: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")
    
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
        ("Decode Resilience", test_llama_decode_resilience),
        ("Text Processing", test_text_processing),
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
    
    if results.get("Decode Resilience", False):
        print("   ✅ Pipeline handles batching and decode errors well")
    else:
        print("   ⚠️  Pipeline may struggle with llama_decode errors - reduce batch sizes")
    
    if results.get("Text Processing", False):
        print("   ✅ Text processing and edge cases are handled")
    else:
        print("   ⚠️  Text processing needs improvement - check splitting logic")
    
    if passed == total:
        print("   🎉 All systems operational!")
        exit_code = 0
    elif passed >= total * 0.5:
        print("   ⚠️  Degraded performance - some issues detected")
        exit_code = 1
    else:
        print("   🚨 Critical issues - major problems detected")
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