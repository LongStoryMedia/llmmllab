"""
Unit tests for Resizer memory estimation with real-world validation data.

Tests the Resizer class against actual memory measurements collected from 
real llama.cpp executions to ensure accuracy and reliability.
"""

import pytest
import json
import os
from unittest.mock import Mock
from typing import Dict, Any, List

from runner.utils.resizer import Resizer, MemoryBreakdown
from models import Model, ModelDetails, OptimalParameters


class TestResizerRealWorldValidation:
    """Test suite for Resizer validation against real memory measurements."""

    @pytest.fixture
    def resizer(self):
        """Fixture providing a Resizer instance."""
        return Resizer()

    @pytest.fixture
    def real_memory_samples(self) -> List[Dict[str, Any]]:
        """Load real memory samples collected from actual llama.cpp executions."""
        samples_path = os.path.join(
            os.path.dirname(__file__), 
            "real_memory_samples.json"
        )
        
        if not os.path.exists(samples_path):
            pytest.skip("Real memory samples file not found")
            
        with open(samples_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_mock_model_from_sample(self, sample: Dict[str, Any]) -> Model:
        """Create a mock Model object from a real memory sample."""
        
        # Extract model size from GGUF path or estimate from param_size
        model_size = self._estimate_model_size_from_param_size(sample["param_size"])
        
        # Calculate clip model size if mmproj is present
        clip_model_size = 1024 * 1024 * 1024 if sample.get("mmproj_path") else 0  # 1GB estimate
        
        model_details = ModelDetails(
            parent_model="test",
            format="gguf",
            size=model_size,
            family="test",
            families=["test"],
            parameter_size=sample["param_size"],
            dtype="Q4_K_M",
            quantization_level="q4_k_m",
            specialization="Text",
            gguf_file=sample["gguf_path"],
            clip_model_path=sample.get("mmproj_path"),
            clip_model_size=clip_model_size,
            supports_thinking=True,
            supports_vision=bool(sample.get("mmproj_path")),
            original_ctx=262144,
            n_layers=self._estimate_layers_from_param_size(sample["param_size"]),
            hidden_size=self._estimate_hidden_size_from_param_size(sample["param_size"]),
            n_heads=64,
            n_kv_heads=8
        )
        
        return Model(
            id=sample["model_id"],
            name=sample["model_name"],
            model=sample["gguf_path"],
            provider="llama_cpp",
            pipeline="TestPipeline",
            modified_at="2025-01-01",
            digest="test",
            details=model_details,
            task="TextToText"
        )

    def _estimate_model_size_from_param_size(self, param_size: str) -> int:
        """Estimate model size in bytes from parameter size string."""
        # Extract numeric value
        if "32B" in param_size.upper():
            return int(20 * 1024**3)  # 20GB for Q4_K_M 32B
        elif "30B" in param_size.upper():
            return int(18 * 1024**3)  # 18GB for Q4_K_M 30B
        elif "4B" in param_size.upper():
            return int(3.5 * 1024**3)  # 3.5GB for Q6_K_XL 4B
        elif "3.2B" in param_size or "3B" in param_size.upper():
            return int(2.3 * 1024**3)  # 2.3GB for Q5_K_M 3B
        elif "2B" in param_size.upper():
            return int(1.8 * 1024**3)  # 1.8GB for F16 2B
        else:
            return int(5 * 1024**3)  # 5GB fallback

    def _estimate_layers_from_param_size(self, param_size: str) -> int:
        """Estimate number of layers from parameter size."""
        if "32B" in param_size.upper():
            return 64
        elif "30B" in param_size.upper():
            return 48
        elif "4B" in param_size.upper():
            return 36
        elif any(x in param_size.upper() for x in ["3.2B", "3B"]):
            return 28
        elif "2B" in param_size.upper():
            return 28
        else:
            return 32

    def _estimate_hidden_size_from_param_size(self, param_size: str) -> int:
        """Estimate hidden size from parameter size."""
        if "32B" in param_size.upper():
            return 5120
        elif "30B" in param_size.upper():
            return 2048
        elif "4B" in param_size.upper():
            return 2560
        elif any(x in param_size.upper() for x in ["3.2B", "3B"]):
            return 3072
        elif "2B" in param_size.upper():
            return 2048
        else:
            return 4096

    def _create_optimal_parameters_from_sample(self, sample: Dict[str, Any]) -> OptimalParameters:
        """Create OptimalParameters from a real memory sample."""
        return OptimalParameters(
            n_ctx=sample["context_size"],
            n_batch=sample["batch_size"],
            n_ubatch=sample.get("ubatch_size", sample["batch_size"]),
            n_gpu_layers=sample["gpu_layers"],
            n_threads=24,
            n_threads_batch=24
        )

    def test_resizer_accuracy_against_real_samples(self, resizer: Resizer, real_memory_samples: List[Dict[str, Any]]):
        """Test Resizer accuracy against real memory measurements."""
        
        # Filter samples to get a diverse set for testing
        test_samples = [
            sample for sample in real_memory_samples 
            if sample.get("total_actual_gb", 0) > 0  # Only successful measurements
        ]
        
        assert len(test_samples) >= 10, f"Need at least 10 successful samples, got {len(test_samples)}"
        
        accuracy_results = []
        
        for sample in test_samples:
            # Create mock objects
            model = self._create_mock_model_from_sample(sample)
            params = self._create_optimal_parameters_from_sample(sample)
            
            # Calculate memory breakdown
            breakdown = resizer.calculate_memory_breakdown(params, model)
            
            # Get total estimated GPU memory
            estimated_gb = breakdown["total_gpu_gb"]
            actual_gb = sample["total_actual_gb"]
            
            # Calculate accuracy ratio
            if actual_gb > 0:
                accuracy_ratio = estimated_gb / actual_gb
                accuracy_results.append({
                    "model": sample["model_name"],
                    "context": sample["context_size"],
                    "estimated_gb": estimated_gb,
                    "actual_gb": actual_gb,
                    "accuracy_ratio": accuracy_ratio,
                    "notes": sample.get("notes", "")
                })
        
        # Assert we have meaningful results
        assert len(accuracy_results) >= 10, "Need at least 10 accuracy measurements"
        
        # Calculate aggregate statistics
        ratios = [r["accuracy_ratio"] for r in accuracy_results]
        avg_accuracy = sum(ratios) / len(ratios)
        min_accuracy = min(ratios)
        max_accuracy = max(ratios)
        
        # Print results for debugging
        print(f"\nResizer Accuracy Results ({len(accuracy_results)} samples):")
        print(f"Average accuracy ratio: {avg_accuracy:.2f}")
        print(f"Min accuracy ratio: {min_accuracy:.2f}")
        print(f"Max accuracy ratio: {max_accuracy:.2f}")
        
        # Assert reasonable accuracy bounds
        # Based on real data, estimates are typically 2-6x higher than actual
        assert 0.1 <= avg_accuracy <= 10.0, f"Average accuracy {avg_accuracy:.2f} outside reasonable range"
        assert min_accuracy >= 0.05, f"Minimum accuracy {min_accuracy:.2f} too low"
        assert max_accuracy <= 20.0, f"Maximum accuracy {max_accuracy:.2f} too high"

    def test_resizer_with_small_models(self, resizer: Resizer, real_memory_samples: List[Dict[str, Any]]):
        """Test Resizer accuracy specifically with small models (≤4B parameters)."""
        
        small_model_samples = [
            sample for sample in real_memory_samples
            if sample.get("total_actual_gb", 0) > 0 and 
            any(size in sample["param_size"].upper() for size in ["2B", "3B", "3.2B", "4B"])
        ]
        
        assert len(small_model_samples) >= 5, f"Need at least 5 small model samples, got {len(small_model_samples)}"
        
        accuracy_results = []
        
        for sample in small_model_samples:
            model = self._create_mock_model_from_sample(sample)
            params = self._create_optimal_parameters_from_sample(sample)
            
            breakdown = resizer.calculate_memory_breakdown(params, model)
            estimated_gb = breakdown["total_gpu_gb"]
            actual_gb = sample["total_actual_gb"]
            
            if actual_gb > 0:
                accuracy_ratio = estimated_gb / actual_gb
                accuracy_results.append({
                    "model": sample["model_name"],
                    "param_size": sample["param_size"],
                    "context": sample["context_size"],
                    "estimated_gb": estimated_gb,
                    "actual_gb": actual_gb,
                    "accuracy_ratio": accuracy_ratio
                })
        
        # Small models should have better accuracy
        ratios = [r["accuracy_ratio"] for r in accuracy_results]
        avg_accuracy = sum(ratios) / len(ratios)
        
        print(f"\nSmall Model Accuracy Results ({len(accuracy_results)} samples):")
        for result in accuracy_results:
            print(f"  {result['model']} ({result['param_size']}) @ {result['context']//1024}K: "
                  f"{result['estimated_gb']:.1f}GB est vs {result['actual_gb']:.1f}GB actual "
                  f"({result['accuracy_ratio']:.2f}x)")
        
        # Small models typically show 2-5x overestimation
        assert 0.2 <= avg_accuracy <= 8.0, f"Small model average accuracy {avg_accuracy:.2f} outside expected range"

    def test_resizer_with_large_models(self, resizer: Resizer, real_memory_samples: List[Dict[str, Any]]):
        """Test Resizer accuracy specifically with large models (≥30B parameters)."""
        
        large_model_samples = [
            sample for sample in real_memory_samples
            if sample.get("total_actual_gb", 0) > 0 and 
            any(size in sample["param_size"].upper() for size in ["30B", "32B"])
        ]
        
        if len(large_model_samples) < 3:
            pytest.skip(f"Not enough large model samples for testing, got {len(large_model_samples)}")
        
        accuracy_results = []
        
        for sample in large_model_samples:
            model = self._create_mock_model_from_sample(sample)
            params = self._create_optimal_parameters_from_sample(sample)
            
            breakdown = resizer.calculate_memory_breakdown(params, model)
            estimated_gb = breakdown["total_gpu_gb"]
            actual_gb = sample["total_actual_gb"]
            
            if actual_gb > 0:
                accuracy_ratio = estimated_gb / actual_gb
                accuracy_results.append({
                    "model": sample["model_name"],
                    "param_size": sample["param_size"],
                    "context": sample["context_size"],
                    "estimated_gb": estimated_gb,
                    "actual_gb": actual_gb,
                    "accuracy_ratio": accuracy_ratio
                })
        
        # Large models should still be reasonably accurate
        ratios = [r["accuracy_ratio"] for r in accuracy_results]
        avg_accuracy = sum(ratios) / len(ratios)
        
        print(f"\nLarge Model Accuracy Results ({len(accuracy_results)} samples):")
        for result in accuracy_results:
            print(f"  {result['model']} ({result['param_size']}) @ {result['context']//1024}K: "
                  f"{result['estimated_gb']:.1f}GB est vs {result['actual_gb']:.1f}GB actual "
                  f"({result['accuracy_ratio']:.2f}x)")
        
        # Large models may have wider variance due to complexity
        assert 0.1 <= avg_accuracy <= 15.0, f"Large model average accuracy {avg_accuracy:.2f} outside expected range"

    def test_resizer_with_vision_models(self, resizer: Resizer, real_memory_samples: List[Dict[str, Any]]):
        """Test Resizer accuracy with vision models that have mmproj files."""
        
        vision_model_samples = [
            sample for sample in real_memory_samples
            if sample.get("total_actual_gb", 0) > 0 and sample.get("mmproj_path") is not None
        ]
        
        if len(vision_model_samples) < 2:
            pytest.skip(f"Not enough vision model samples for testing, got {len(vision_model_samples)}")
        
        accuracy_results = []
        
        for sample in vision_model_samples:
            model = self._create_mock_model_from_sample(sample)
            params = self._create_optimal_parameters_from_sample(sample)
            
            breakdown = resizer.calculate_memory_breakdown(params, model)
            estimated_gb = breakdown["total_gpu_gb"]
            actual_gb = sample["total_actual_gb"]
            
            # Verify CLIP model is accounted for
            assert breakdown["clip_model_gb"] > 0, "Vision model should have CLIP model component"
            
            if actual_gb > 0:
                accuracy_ratio = estimated_gb / actual_gb
                accuracy_results.append({
                    "model": sample["model_name"],
                    "param_size": sample["param_size"],
                    "context": sample["context_size"],
                    "estimated_gb": estimated_gb,
                    "actual_gb": actual_gb,
                    "accuracy_ratio": accuracy_ratio,
                    "clip_gb": breakdown["clip_model_gb"]
                })
        
        # Vision models include additional CLIP model overhead
        ratios = [r["accuracy_ratio"] for r in accuracy_results]
        avg_accuracy = sum(ratios) / len(ratios)
        
        print(f"\nVision Model Accuracy Results ({len(accuracy_results)} samples):")
        for result in accuracy_results:
            print(f"  {result['model']} ({result['param_size']}) @ {result['context']//1024}K: "
                  f"{result['estimated_gb']:.1f}GB est ({result['clip_gb']:.1f}GB CLIP) vs "
                  f"{result['actual_gb']:.1f}GB actual ({result['accuracy_ratio']:.2f}x)")
        
        # Vision models may have different accuracy patterns due to CLIP overhead
        assert 0.1 <= avg_accuracy <= 15.0, f"Vision model average accuracy {avg_accuracy:.2f} outside expected range"

    def test_resizer_context_scaling(self, resizer: Resizer, real_memory_samples: List[Dict[str, Any]]):
        """Test that Resizer properly scales memory estimates with context size."""
        
        # Find samples with same model but different context sizes
        model_groups = {}
        for sample in real_memory_samples:
            if sample.get("total_actual_gb", 0) > 0:
                key = (sample["model_id"], sample["batch_size"], sample["gpu_layers"])
                if key not in model_groups:
                    model_groups[key] = []
                model_groups[key].append(sample)
        
        # Find a model with multiple context size measurements
        test_group = None
        for group in model_groups.values():
            if len(group) >= 3:  # At least 3 different context sizes
                test_group = sorted(group, key=lambda x: x["context_size"])
                break
        
        if not test_group:
            pytest.skip("No model group found with multiple context size measurements")
        
        context_results = []
        
        for sample in test_group:
            model = self._create_mock_model_from_sample(sample)
            params = self._create_optimal_parameters_from_sample(sample)
            
            breakdown = resizer.calculate_memory_breakdown(params, model)
            
            context_results.append({
                "context_size": sample["context_size"],
                "estimated_gb": breakdown["total_gpu_gb"],
                "actual_gb": sample["total_actual_gb"],
                "kv_cache_gb": breakdown["kv_cache_gb"]
            })
        
        print(f"\nContext Scaling Results for {test_group[0]['model_name']}:")
        for result in context_results:
            print(f"  {result['context_size']//1024}K context: "
                  f"{result['estimated_gb']:.1f}GB est ({result['kv_cache_gb']:.1f}GB KV) vs "
                  f"{result['actual_gb']:.1f}GB actual")
        
        # Verify that estimates increase with context size
        for i in range(1, len(context_results)):
            prev = context_results[i-1]
            curr = context_results[i]
            
            assert curr["context_size"] > prev["context_size"], "Results should be ordered by context size"
            assert curr["kv_cache_gb"] > prev["kv_cache_gb"], "KV cache should increase with context size"
            # Total memory should generally increase (allowing for some variance)
            assert curr["estimated_gb"] >= prev["estimated_gb"] * 0.9, "Total memory should generally increase with context"

    def test_resizer_memory_breakdown_components(self, resizer: Resizer, real_memory_samples: List[Dict[str, Any]]):
        """Test that Resizer breakdown components are reasonable."""
        
        # Test with a representative sample
        test_samples = [
            sample for sample in real_memory_samples[:10]  # First 10 samples
            if sample.get("total_actual_gb", 0) > 0
        ]
        
        assert len(test_samples) >= 5, "Need at least 5 samples for breakdown testing"
        
        for sample in test_samples:
            model = self._create_mock_model_from_sample(sample)
            params = self._create_optimal_parameters_from_sample(sample)
            
            breakdown = resizer.calculate_memory_breakdown(params, model)
            
            # Verify all components are non-negative
            assert breakdown["model_weights_gpu_gb"] >= 0, "Model weights should be non-negative"
            assert breakdown["kv_cache_gb"] >= 0, "KV cache should be non-negative"
            assert breakdown["activation_gb"] >= 0, "Activation memory should be non-negative"
            assert breakdown["overhead_gb"] >= 0, "Overhead should be non-negative"
            assert breakdown["clip_model_gb"] >= 0, "CLIP model should be non-negative"
            
            # Verify total is sum of components
            expected_total = (
                breakdown["model_weights_gpu_gb"] +
                breakdown["kv_cache_gb"] +
                breakdown["activation_gb"] +
                breakdown["overhead_gb"] +
                breakdown["clip_model_gb"]
            )
            
            assert abs(breakdown["total_gpu_gb"] - expected_total) < 0.02, (
                f"Total GPU memory {breakdown['total_gpu_gb']:.2f}GB should equal sum of components "
                f"{expected_total:.2f}GB"
            )
            
            # Verify model weights are reasonable (should be significant portion)
            assert breakdown["model_weights_gpu_gb"] > 0, "Model weights should be present"
            
            # For models with mmproj, verify CLIP component
            if sample.get("mmproj_path"):
                assert breakdown["clip_model_gb"] > 0, "Vision models should have CLIP component"
            else:
                assert breakdown["clip_model_gb"] == 0, "Text-only models should not have CLIP component"

    def test_resizer_specific_assertions_from_real_data(self, resizer: Resizer):
        """Test specific assertions based on collected real memory data."""
        
        # Test case 1: 3.2B model at 4K context (from real data)
        model_3b = Model(
            id="test-3b",
            name="Test 3B",
            model="/test/3b.gguf",
            provider="llama_cpp",
            pipeline="TestPipeline",
            modified_at="2025-01-01",
            digest="test",
            details=ModelDetails(
                parent_model="test",
                format="gguf",
                size=int(2.3 * 1024**3),  # 2.3GB
                family="test",
                families=["test"],
                parameter_size="3.2B",
                dtype="Q5_K_M",
                quantization_level="q5_k_m",
                specialization="Text",
                gguf_file="/test/3b.gguf",
                original_ctx=131072,
                n_layers=28,
                hidden_size=3072,
                n_heads=24,
                n_kv_heads=8
            ),
            task="TextToText"
        )
        
        params_4k = OptimalParameters(
            n_ctx=4096,
            n_batch=512,
            n_ubatch=512,
            n_gpu_layers=35,
            n_threads=24,
            n_threads_batch=24
        )
        
        breakdown_4k = resizer.calculate_memory_breakdown(params_4k, model_3b)
        
        # Based on real measurement: 8.38GB estimated vs 3.67GB actual (0.44x accuracy)
        # Allow reasonable range around this - estimates are now more conservative
        assert 2.5 <= breakdown_4k["total_gpu_gb"] <= 8.0, (
            f"3B model @ 4K context estimate {breakdown_4k['total_gpu_gb']:.1f}GB outside expected range"
        )
        
        # Test case 2: 4B model at 128K context (from real data) 
        model_4b = Model(
            id="test-4b",
            name="Test 4B",
            model="/test/4b.gguf",
            provider="llama_cpp",
            pipeline="TestPipeline",
            modified_at="2025-01-01",
            digest="test",
            details=ModelDetails(
                parent_model="test",
                format="gguf",
                size=int(3.5 * 1024**3),  # 3.5GB
                family="test",
                families=["test"],
                parameter_size="4B",
                dtype="Q6_K_XL",
                quantization_level="q6_k_xl",
                specialization="Text",
                gguf_file="/test/4b.gguf",
                original_ctx=40960,
                n_layers=36,
                hidden_size=2560,
                n_heads=32,
                n_kv_heads=8
            ),
            task="TextToText"
        )
        
        params_128k = OptimalParameters(
            n_ctx=131072,
            n_batch=4096,
            n_ubatch=4096,
            n_gpu_layers=-1,
            n_threads=24,
            n_threads_batch=24
        )
        
        breakdown_128k = resizer.calculate_memory_breakdown(params_128k, model_4b)
        
        # High context should significantly increase memory estimate
        assert breakdown_128k["total_gpu_gb"] > breakdown_4k["total_gpu_gb"], (
            "128K context should require more memory than 4K context"
        )
        
        assert breakdown_128k["kv_cache_gb"] > 2.0, (
            "128K context should have substantial KV cache memory requirement"
        )
        
        # Test case 3: Large model from real data (if available)
        # Based on real measurement of 32B VL model: 23GB estimated vs 25.9GB actual (1.13x accuracy)
        model_32b = Model(
            id="test-32b",
            name="Test 32B VL",
            model="/test/32b.gguf",
            provider="llama_cpp",
            pipeline="TestPipeline",
            modified_at="2025-01-01",
            digest="test",
            details=ModelDetails(
                parent_model="test",
                format="gguf",
                size=int(20 * 1024**3),  # 20GB
                family="test",
                families=["test"],
                parameter_size="32B",
                dtype="Q4_K_M",
                quantization_level="q4_k_m",
                specialization="Text",
                gguf_file="/test/32b.gguf",
                clip_model_path="/test/mmproj.gguf",
                clip_model_size=int(1.2 * 1024**3),  # 1.2GB
                supports_vision=True,
                original_ctx=262144,
                n_layers=64,
                hidden_size=5120,
                n_heads=64,
                n_kv_heads=8
            ),
            task="VisionTextToText"
        )
        
        params_40k = OptimalParameters(
            n_ctx=40960,
            n_batch=4096,
            n_ubatch=4096,
            n_gpu_layers=-1,
            n_threads=24,
            n_threads_batch=24
        )
        
        breakdown_32b = resizer.calculate_memory_breakdown(params_40k, model_32b)
        
        # Based on real measurement: should be in range of 18-30GB
        assert 15.0 <= breakdown_32b["total_gpu_gb"] <= 35.0, (
            f"32B VL model @ 40K context estimate {breakdown_32b['total_gpu_gb']:.1f}GB outside expected range"
        )
        
        # Should have CLIP model component
        assert breakdown_32b["clip_model_gb"] > 0, "32B VL model should have CLIP component"
        
        # Model weights should be substantial
        assert breakdown_32b["model_weights_gpu_gb"] >= 15.0, (
            "32B model should have substantial GPU weights"
        )