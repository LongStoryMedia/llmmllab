#!/usr/bin/env python3
"""
Comprehensive Embedding Pipeline Test Suite

Tests for the Nomic embedding pipeline including smoke tests, similarity tests,
and text splitting validation. Designed to run on Kubernetes pods to diagnose
embedding pipeline issues.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Setup logging for detailed diagnostics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the inference directory to path for imports
sys.path.insert(0, '/app')

# Import required modules
try:
    from models import Model, ModelProfile, Message, MessageRole, MessageContent, MessageContentType
    from runner.pipelines.emb.nom2 import NomicEmbedTextPipe
    from runner.pipeline_factory import pipeline_factory
    logger.info("✅ Successfully imported required modules")
except ImportError as e:
    logger.error(f"❌ Failed to import required modules: {e}")
    sys.exit(1)


class EmbeddingTestSuite:
    """Comprehensive test suite for embedding pipeline validation"""
    
    def __init__(self):
        self.pipeline: Optional[NomicEmbedTextPipe] = None
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "cuda_available": False
            },
            "tests": {}
        }
        
        # Test data for various scenarios
        self.test_texts = {
            "simple": "Hello world",
            "query": "What are the latest AI breakthroughs in 2024?",
            "document": "Artificial intelligence has made significant advances in 2024, particularly in the areas of large language models, computer vision, and robotics. These developments have implications for various industries.",
            "multilingual": "Hello world. Bonjour le monde. Hola mundo. 你好世界. مرحبا بالعالم",
            "long": "This is a very long document that should test the text splitting functionality of the embedding pipeline. " * 100,
            "empty": "",
            "special_chars": "Test with special characters: @#$%^&*()_+-=[]{}|;':\",./<>?`~",
            "unicode": "Unicode test: 🌍🚀🔬💡🎯📊🔥✨⭐🎨 émojis and spëcial chars",
            "code": "def hello_world():\n    print('Hello, World!')\n    return 42",
            "numbers": "1234567890 + 9876543210 = 11111111100",
        }

    def _setup_environment_info(self):
        """Gather environment information for diagnostics"""
        try:
            import torch
            self.test_results["environment"]["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.test_results["environment"]["cuda_device_count"] = torch.cuda.device_count()
                self.test_results["environment"]["cuda_device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            logger.warning("PyTorch not available for CUDA detection")
        
        # Check for model files
        model_paths = ["/app/.models.json", "/models", "/app/models"]
        self.test_results["environment"]["model_paths_checked"] = model_paths
        self.test_results["environment"]["model_paths_exist"] = {
            path: os.path.exists(path) for path in model_paths
        }

    async def setup_pipeline(self) -> bool:
        """Initialize the embedding pipeline for testing"""
        try:
            logger.info("🔧 Setting up embedding pipeline...")
            
            # Create a minimal model configuration for testing
            # This should match your actual Nomic model configuration
            test_model = Model(
                id="test-nomic-embed",
                name="Nomic Embed Text v1.5",
                model="/app/models/nomic-embed-text-v1.5.f16.gguf",  # Adjust path as needed
                details=type('Details', (), {
                    'gguf_file': "/app/models/nomic-embed-text-v1.5.f16.gguf"
                })()
            )
            
            test_profile = ModelProfile(
                id="test-profile",
                name="Test Embedding Profile",
                user_id="test-user",
                parameters=type('Parameters', (), {
                    'num_ctx': 512,
                    'seed': 42
                })()
            )
            
            # Initialize the pipeline
            self.pipeline = NomicEmbedTextPipe(test_model, test_profile)
            logger.info("✅ Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup pipeline: {e}")
            self.test_results["setup_error"] = str(e)
            return False

    async def test_smoke_test(self) -> Dict[str, Any]:
        """Basic smoke test - can we generate embeddings at all?"""
        logger.info("🧪 Running smoke test...")
        
        test_result = {
            "name": "smoke_test",
            "description": "Basic embedding generation test",
            "status": "FAILED",
            "details": {}
        }
        
        try:
            # Test simple text embedding
            test_text = self.test_texts["simple"]
            
            start_time = time.time()
            embeddings = await self.pipeline.embed_texts([test_text])
            duration = time.time() - start_time
            
            # Validate results
            if embeddings and len(embeddings) == 1 and len(embeddings[0]) == 768:
                test_result["status"] = "PASSED"
                test_result["details"] = {
                    "input_text": test_text,
                    "embedding_dimension": len(embeddings[0]),
                    "duration_seconds": duration,
                    "first_few_values": embeddings[0][:5],
                    "embedding_norm": float(np.linalg.norm(embeddings[0]))
                }
                logger.info(f"✅ Smoke test passed - {len(embeddings[0])}D embedding in {duration:.3f}s")
            else:
                test_result["details"]["error"] = f"Invalid embedding format: {len(embeddings) if embeddings else 0} embeddings"
                logger.error(f"❌ Smoke test failed - invalid embedding format")
                
        except Exception as e:
            test_result["details"]["error"] = str(e)
            logger.error(f"❌ Smoke test failed with exception: {e}")
        
        return test_result

    async def test_cosine_similarity(self) -> Dict[str, Any]:
        """Test cosine similarity functionality"""
        logger.info("🧪 Running cosine similarity test...")
        
        test_result = {
            "name": "cosine_similarity_test",
            "description": "Test semantic similarity between related and unrelated texts",
            "status": "FAILED",
            "details": {}
        }
        
        try:
            # Test pairs: (text1, text2, expected_high_similarity)
            test_pairs = [
                ("AI breakthrough", "artificial intelligence advancement", True),
                ("machine learning", "deep learning neural networks", True),
                ("Python programming", "software development", True),
                ("cat", "elephant", False),
                ("mathematics", "cooking recipes", False),
                ("hello world", "goodbye moon", False),
            ]
            
            similarity_results = []
            
            for text1, text2, expected_high in test_pairs:
                try:
                    # Generate embeddings
                    embeddings = await self.pipeline.embed_texts([text1, text2])
                    
                    if len(embeddings) == 2:
                        # Calculate cosine similarity
                        emb1 = np.array(embeddings[0])
                        emb2 = np.array(embeddings[1])
                        
                        # Normalize vectors
                        emb1_norm = emb1 / np.linalg.norm(emb1)
                        emb2_norm = emb2 / np.linalg.norm(emb2)
                        
                        # Cosine similarity
                        similarity = float(np.dot(emb1_norm, emb2_norm))
                        
                        # Check if similarity matches expectation
                        is_correct = (similarity > 0.7) == expected_high
                        
                        similarity_results.append({
                            "text1": text1,
                            "text2": text2,
                            "similarity": similarity,
                            "expected_high": expected_high,
                            "is_correct": is_correct
                        })
                        
                        logger.info(f"  Similarity({text1[:20]}..., {text2[:20]}...): {similarity:.3f} {'✅' if is_correct else '❌'}")
                    
                except Exception as e:
                    similarity_results.append({
                        "text1": text1,
                        "text2": text2,
                        "error": str(e)
                    })
            
            # Calculate success rate
            correct_count = sum(1 for r in similarity_results if r.get("is_correct", False))
            total_count = len([r for r in similarity_results if "error" not in r])
            success_rate = correct_count / total_count if total_count > 0 else 0
            
            test_result["details"] = {
                "similarity_results": similarity_results,
                "correct_predictions": correct_count,
                "total_predictions": total_count,
                "success_rate": success_rate
            }
            
            # Pass if success rate > 70%
            if success_rate > 0.7:
                test_result["status"] = "PASSED"
                logger.info(f"✅ Cosine similarity test passed - {success_rate:.1%} accuracy")
            else:
                logger.error(f"❌ Cosine similarity test failed - {success_rate:.1%} accuracy")
                
        except Exception as e:
            test_result["details"]["error"] = str(e)
            logger.error(f"❌ Cosine similarity test failed with exception: {e}")
        
        return test_result

    async def test_text_splitting(self) -> Dict[str, Any]:
        """Test automatic text splitting for long documents"""
        logger.info("🧪 Running text splitting test...")
        
        test_result = {
            "name": "text_splitting_test",
            "description": "Test automatic splitting of long texts that exceed token limits",
            "status": "FAILED",
            "details": {}
        }
        
        try:
            splitting_tests = []
            
            # Test different text lengths
            test_cases = [
                ("short", "Short text", False),
                ("medium", "Medium text. " * 50, False),  # ~100 tokens
                ("long", "This is a long text that should be split. " * 200, True),  # ~1600 tokens
                ("very_long", self.test_texts["long"], True)  # Very long
            ]
            
            for case_name, text, should_split in test_cases:
                try:
                    # Check if text will be split
                    will_split = self.pipeline.will_text_be_split(text)
                    token_estimate = self.pipeline.get_token_count_estimate(text)
                    
                    # Generate embedding
                    start_time = time.time()
                    embeddings = await self.pipeline.embed_texts([text])
                    duration = time.time() - start_time
                    
                    # Validate results
                    success = (
                        embeddings and 
                        len(embeddings) == 1 and 
                        len(embeddings[0]) == 768 and
                        will_split == should_split
                    )
                    
                    splitting_tests.append({
                        "case": case_name,
                        "text_length": len(text),
                        "estimated_tokens": token_estimate,
                        "will_split": will_split,
                        "should_split": should_split,
                        "split_prediction_correct": will_split == should_split,
                        "embedding_generated": embeddings is not None,
                        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                        "duration_seconds": duration,
                        "success": success
                    })
                    
                    logger.info(f"  {case_name}: {len(text)} chars, {token_estimate} tokens, split={will_split} {'✅' if success else '❌'}")
                    
                except Exception as e:
                    splitting_tests.append({
                        "case": case_name,
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate success rate
            successful_tests = [t for t in splitting_tests if t.get("success", False)]
            success_rate = len(successful_tests) / len(splitting_tests)
            
            test_result["details"] = {
                "splitting_tests": splitting_tests,
                "successful_tests": len(successful_tests),
                "total_tests": len(splitting_tests),
                "success_rate": success_rate
            }
            
            # Pass if success rate > 75%
            if success_rate > 0.75:
                test_result["status"] = "PASSED"
                logger.info(f"✅ Text splitting test passed - {success_rate:.1%} success rate")
            else:
                logger.error(f"❌ Text splitting test failed - {success_rate:.1%} success rate")
                
        except Exception as e:
            test_result["details"]["error"] = str(e)
            logger.error(f"❌ Text splitting test failed with exception: {e}")
        
        return test_result

    async def test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing and retry logic for llama_decode errors"""
        logger.info("🧪 Running batch processing test...")
        
        test_result = {
            "name": "batch_processing_test",
            "description": "Test batching and retry logic to handle llama_decode errors",
            "status": "FAILED",
            "details": {}
        }
        
        try:
            # Test different batch sizes
            test_texts = [
                f"Test document {i}: {self.test_texts['document']} Additional content {i}." 
                for i in range(1, 16)  # 15 texts
            ]
            
            batch_tests = []
            
            # Test with different batch configurations
            batch_configs = [
                {"batch_size": 1, "description": "Single item processing"},
                {"batch_size": 4, "description": "Small batch processing"},
                {"batch_size": 8, "description": "Default batch processing"},
                {"batch_size": 15, "description": "Large batch processing"},
            ]
            
            for config in batch_configs:
                try:
                    start_time = time.time()
                    
                    # Generate embeddings with specific batch size
                    embeddings = await self.pipeline._generate_embeddings_with_batching(
                        test_texts, 
                        max_batch_size=config["batch_size"]
                    )
                    
                    duration = time.time() - start_time
                    
                    # Validate results
                    success = (
                        embeddings and 
                        len(embeddings) == len(test_texts) and
                        all(len(emb) == 768 for emb in embeddings)
                    )
                    
                    batch_tests.append({
                        "batch_size": config["batch_size"],
                        "description": config["description"],
                        "input_count": len(test_texts),
                        "output_count": len(embeddings) if embeddings else 0,
                        "all_correct_dimension": all(len(emb) == 768 for emb in embeddings) if embeddings else False,
                        "duration_seconds": duration,
                        "success": success
                    })
                    
                    logger.info(f"  Batch size {config['batch_size']}: {len(embeddings) if embeddings else 0}/{len(test_texts)} embeddings in {duration:.2f}s {'✅' if success else '❌'}")
                    
                except Exception as e:
                    batch_tests.append({
                        "batch_size": config["batch_size"],
                        "description": config["description"],
                        "error": str(e),
                        "success": False
                    })
                    logger.warning(f"  Batch size {config['batch_size']}: Failed with {e}")
            
            # Calculate results
            successful_tests = [t for t in batch_tests if t.get("success", False)]
            success_rate = len(successful_tests) / len(batch_tests)
            
            test_result["details"] = {
                "batch_tests": batch_tests,
                "successful_configs": len(successful_tests),
                "total_configs": len(batch_tests),
                "success_rate": success_rate
            }
            
            # Pass if at least one batch size works
            if success_rate > 0:
                test_result["status"] = "PASSED"
                logger.info(f"✅ Batch processing test passed - {len(successful_tests)}/{len(batch_tests)} configurations work")
            else:
                logger.error(f"❌ Batch processing test failed - no configurations work")
                
        except Exception as e:
            test_result["details"]["error"] = str(e)
            logger.error(f"❌ Batch processing test failed with exception: {e}")
        
        return test_result

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and fallback mechanisms"""
        logger.info("🧪 Running error handling test...")
        
        test_result = {
            "name": "error_handling_test",
            "description": "Test pipeline behavior with problematic inputs",
            "status": "FAILED",
            "details": {}
        }
        
        try:
            error_tests = []
            
            # Test problematic inputs
            problematic_inputs = [
                ("empty_string", ""),
                ("whitespace_only", "   \n\t\r   "),
                ("very_short", "a"),
                ("special_chars", self.test_texts["special_chars"]),
                ("unicode", self.test_texts["unicode"]),
                ("extreme_length", "x" * 50000),  # Very long text
                ("none_input", None),
            ]
            
            for case_name, input_text in problematic_inputs:
                try:
                    if input_text is None:
                        # Test None handling (should gracefully fail)
                        try:
                            embeddings = await self.pipeline.embed_texts([input_text])
                            success = False  # Should have failed
                        except Exception:
                            success = True  # Expected to fail
                            embeddings = None
                    else:
                        embeddings = await self.pipeline.embed_texts([input_text])
                        success = (
                            embeddings is not None and 
                            len(embeddings) == 1 and
                            len(embeddings[0]) == 768
                        )
                    
                    error_tests.append({
                        "case": case_name,
                        "input": input_text[:100] if input_text else str(input_text),
                        "embedding_generated": embeddings is not None,
                        "success": success
                    })
                    
                    logger.info(f"  {case_name}: {'✅' if success else '❌'}")
                    
                except Exception as e:
                    # For problematic inputs, graceful handling (zero embeddings) is acceptable
                    error_tests.append({
                        "case": case_name,
                        "input": input_text[:100] if input_text else str(input_text),
                        "error": str(e),
                        "success": True  # Graceful error handling is success
                    })
                    logger.info(f"  {case_name}: ✅ (graceful error handling)")
            
            # Calculate results
            successful_tests = [t for t in error_tests if t.get("success", False)]
            success_rate = len(successful_tests) / len(error_tests)
            
            test_result["details"] = {
                "error_tests": error_tests,
                "handled_gracefully": len(successful_tests),
                "total_tests": len(error_tests),
                "success_rate": success_rate
            }
            
            # Pass if most cases are handled gracefully (>70%)
            if success_rate > 0.7:
                test_result["status"] = "PASSED"
                logger.info(f"✅ Error handling test passed - {success_rate:.1%} cases handled gracefully")
            else:
                logger.error(f"❌ Error handling test failed - {success_rate:.1%} cases handled gracefully")
                
        except Exception as e:
            test_result["details"]["error"] = str(e)
            logger.error(f"❌ Error handling test failed with exception: {e}")
        
        return test_result

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🚀 Starting Embedding Pipeline Test Suite")
        
        # Setup environment info
        self._setup_environment_info()
        
        # Initialize pipeline
        if not await self.setup_pipeline():
            logger.error("❌ Failed to setup pipeline, aborting tests")
            return self.test_results
        
        # Run all tests
        tests_to_run = [
            self.test_smoke_test,
            self.test_cosine_similarity,
            self.test_text_splitting,
            self.test_batch_processing,
            self.test_error_handling,
        ]
        
        for test_func in tests_to_run:
            try:
                test_result = await test_func()
                self.test_results["tests"][test_result["name"]] = test_result
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} crashed: {e}")
                self.test_results["tests"][test_func.__name__] = {
                    "name": test_func.__name__,
                    "status": "CRASHED",
                    "error": str(e)
                }
        
        # Generate summary
        self._generate_summary()
        
        return self.test_results

    def _generate_summary(self):
        """Generate test summary and recommendations"""
        tests = self.test_results["tests"]
        
        passed = len([t for t in tests.values() if t.get("status") == "PASSED"])
        failed = len([t for t in tests.values() if t.get("status") == "FAILED"])
        crashed = len([t for t in tests.values() if t.get("status") == "CRASHED"])
        total = len(tests)
        
        self.test_results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "crashed": crashed,
            "success_rate": passed / total if total > 0 else 0,
            "overall_status": "HEALTHY" if passed >= total * 0.8 else "DEGRADED" if passed >= total * 0.5 else "CRITICAL"
        }
        
        # Generate recommendations
        recommendations = []
        
        if failed > 0 or crashed > 0:
            recommendations.append("🔧 Review failed tests for specific embedding pipeline issues")
        
        if tests.get("smoke_test", {}).get("status") != "PASSED":
            recommendations.append("🚨 Basic embedding generation is failing - check model file and configuration")
        
        if tests.get("batch_processing_test", {}).get("status") != "PASSED":
            recommendations.append("🔧 Batch processing issues detected - review llama_decode error handling")
        
        if tests.get("cosine_similarity_test", {}).get("status") != "PASSED":
            recommendations.append("⚠️ Semantic similarity not working as expected - check model quality")
        
        if not recommendations:
            recommendations.append("✨ All embedding pipeline tests passing - system is healthy")
        
        self.test_results["recommendations"] = recommendations


async def main():
    """Main test execution function"""
    try:
        # Run the test suite
        test_suite = EmbeddingTestSuite()
        results = await test_suite.run_all_tests()
        
        # Print summary
        print("\n" + "="*70)
        print("🧪 EMBEDDING PIPELINE TEST RESULTS")
        print("="*70)
        
        summary = results.get("summary", {})
        print(f"📊 Tests: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed")
        print(f"🎯 Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"🔍 Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        
        print(f"\n📋 Test Details:")
        for test_name, test_result in results.get("tests", {}).items():
            status_icon = {"PASSED": "✅", "FAILED": "❌", "CRASHED": "💥"}.get(test_result.get("status"), "❓")
            print(f"  {status_icon} {test_name}: {test_result.get('status', 'UNKNOWN')}")
        
        print(f"\n💡 Recommendations:")
        for rec in results.get("recommendations", []):
            print(f"  {rec}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"embedding_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📝 Detailed results saved to: {results_file}")
        
        # Environment info
        env_info = results.get("environment", {})
        print(f"\n🖥️  Environment Info:")
        print(f"  • CUDA Available: {env_info.get('cuda_available', 'Unknown')}")
        print(f"  • Working Directory: {env_info.get('working_directory', 'Unknown')}")
        
        print("="*70)
        
        # Exit code based on results
        if summary.get('overall_status') == 'CRITICAL':
            sys.exit(1)
        elif summary.get('overall_status') == 'DEGRADED':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())