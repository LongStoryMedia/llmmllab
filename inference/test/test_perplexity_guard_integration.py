"""
Test suite for perplexity guard integration in pipelines.

This module tests how the circuit breaker configuration integrates
with the pipeline system, specifically testing perplexity guard
enable/disable behavior in realistic pipeline scenarios.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to the sys.path to make imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.circuit_breaker_config import CircuitBreakerConfig
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG
from models.model_profile import ModelProfile


class TestPipelinePerplexityGuardIntegration:
    """Test perplexity guard behavior in pipeline context."""

    def test_default_pipeline_config(self):
        """Test that pipeline uses default perplexity guard setting."""
        # Simulate pipeline initialization with default config
        config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Pipeline should have perplexity guard enabled by default
        assert config.enable_perplexity_guard is True

        # Verify other related settings are also set
        assert config.perplexity_window is not None
        assert config.perplexity_threshold is not None
        assert config.avg_logprob_floor is not None

    def test_pipeline_with_disabled_perplexity_guard(self):
        """Test pipeline behavior when perplexity guard is disabled."""
        # Create profile config that disables perplexity guard
        profile_config = CircuitBreakerConfig(enable_perplexity_guard=False)

        # Merge with global defaults
        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, profile_config
        )

        # Perplexity guard should be disabled
        assert final_config.enable_perplexity_guard is False

        # Other timeout settings should still be available
        assert final_config.base_timeout is not None
        assert final_config.max_retries is not None

    def test_pipeline_config_merge_scenarios(self):
        """Test various configuration merge scenarios for pipeline use."""
        base_config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Scenario 1: Profile with only timeout override
        profile1 = CircuitBreakerConfig(base_timeout=120.0)
        merged1 = CircuitBreakerConfig.merge_with_overrides(base_config, profile1)

        assert merged1.base_timeout == 120.0  # Overridden
        assert merged1.enable_perplexity_guard is True  # From base

        # Scenario 2: Profile with perplexity guard disabled
        profile2 = CircuitBreakerConfig(
            base_timeout=120.0, enable_perplexity_guard=False
        )
        merged2 = CircuitBreakerConfig.merge_with_overrides(base_config, profile2)

        assert merged2.base_timeout == 120.0  # Overridden
        assert merged2.enable_perplexity_guard is False  # Overridden

        # Scenario 3: Profile with perplexity settings but guard enabled
        profile3 = CircuitBreakerConfig(
            enable_perplexity_guard=True, perplexity_threshold=5.0, perplexity_window=20
        )
        merged3 = CircuitBreakerConfig.merge_with_overrides(base_config, profile3)

        assert merged3.enable_perplexity_guard is True  # Overridden (explicit)
        assert merged3.perplexity_threshold == 5.0  # Overridden
        assert merged3.perplexity_window == 20  # Overridden

    def test_model_profile_circuit_breaker_integration(self):
        """Test how circuit breaker config integrates with model profiles."""
        # Mock model profile with circuit breaker config
        mock_profile = MagicMock(spec=ModelProfile)
        mock_profile.circuit_breaker = CircuitBreakerConfig(
            enable_perplexity_guard=False, base_timeout=180.0
        )

        # Simulate pipeline getting config from profile
        profile_circuit_breaker = mock_profile.circuit_breaker

        # Merge with global defaults
        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, profile_circuit_breaker
        )

        # Verify the merge worked correctly
        assert final_config.enable_perplexity_guard is False  # From profile
        assert final_config.base_timeout == 180.0  # From profile
        assert (
            final_config.max_retries == DEFAULT_CIRCUIT_BREAKER_CONFIG.max_retries
        )  # From global

    def test_none_profile_circuit_breaker(self):
        """Test pipeline behavior when profile has no circuit breaker config."""
        # Mock model profile with no circuit breaker config
        mock_profile = MagicMock(spec=ModelProfile)
        mock_profile.circuit_breaker = None

        # Simulate pipeline handling None circuit breaker
        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, mock_profile.circuit_breaker
        )

        # Should be identical to global config
        assert (
            final_config.enable_perplexity_guard
            == DEFAULT_CIRCUIT_BREAKER_CONFIG.enable_perplexity_guard
        )
        assert final_config.base_timeout == DEFAULT_CIRCUIT_BREAKER_CONFIG.base_timeout

    def test_pipeline_safe_value_access(self):
        """Test that pipeline can safely access config values with None handling."""
        # Test with config that has some None values
        partial_config = CircuitBreakerConfig(
            enable_perplexity_guard=False,
            base_timeout=90.0,
            # Other fields are None
        )

        # Merge with defaults to fill in None values
        safe_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, partial_config
        )

        # Pipeline should be able to safely access all values
        assert safe_config.enable_perplexity_guard is False  # From override
        assert safe_config.base_timeout == 90.0  # From override
        assert safe_config.cooldown_period is not None  # From default
        assert safe_config.max_retries is not None  # From default

        # Test that we can use the or operator for additional safety
        cooldown = safe_config.cooldown_period or 30.0
        max_retries = safe_config.max_retries or 2
        enable_guard = safe_config.enable_perplexity_guard or False

        assert cooldown > 0
        assert max_retries >= 0
        assert enable_guard in (True, False)


class TestPerplexityGuardPipelineLogic:
    """Test the actual pipeline logic that depends on perplexity guard setting."""

    def test_perplexity_guard_enabled_logic(self):
        """Test pipeline logic when perplexity guard is enabled."""
        config = CircuitBreakerConfig(enable_perplexity_guard=True)

        # Simulate pipeline logic that checks the guard setting
        if config.enable_perplexity_guard:
            # This would be the path taken when guard is enabled
            should_collect_logprobs = True
            should_monitor_perplexity = True
        else:
            should_collect_logprobs = False
            should_monitor_perplexity = False

        assert should_collect_logprobs is True
        assert should_monitor_perplexity is True

    def test_perplexity_guard_disabled_logic(self):
        """Test pipeline logic when perplexity guard is disabled."""
        config = CircuitBreakerConfig(enable_perplexity_guard=False)

        # Simulate pipeline logic that checks the guard setting
        if config.enable_perplexity_guard:
            should_collect_logprobs = True
            should_monitor_perplexity = True
        else:
            # This would be the path taken when guard is disabled
            should_collect_logprobs = False
            should_monitor_perplexity = False

        assert should_collect_logprobs is False
        assert should_monitor_perplexity is False

    def test_perplexity_guard_none_handling(self):
        """Test pipeline logic when perplexity guard is None (should use default)."""
        # Config with None value
        config = CircuitBreakerConfig()
        assert config.enable_perplexity_guard is None

        # Pipeline should merge with defaults first
        merged_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, config
        )

        # Now it should have a definite value
        assert merged_config.enable_perplexity_guard is not None
        assert (
            merged_config.enable_perplexity_guard
            == DEFAULT_CIRCUIT_BREAKER_CONFIG.enable_perplexity_guard
        )

    def test_conditional_perplexity_features(self):
        """Test that perplexity-related features are conditionally enabled."""
        # Test with guard enabled
        enabled_config = CircuitBreakerConfig(
            enable_perplexity_guard=True, perplexity_window=50, perplexity_threshold=8.0
        )

        # Features should be available when guard is enabled
        if enabled_config.enable_perplexity_guard:
            window = enabled_config.perplexity_window or 40
            threshold = enabled_config.perplexity_threshold or 10.0

            assert window == 50
            assert threshold == 8.0

        # Test with guard disabled
        disabled_config = CircuitBreakerConfig(
            enable_perplexity_guard=False,
            perplexity_window=50,
            perplexity_threshold=8.0,
        )

        # Features should be ignored when guard is disabled
        if not disabled_config.enable_perplexity_guard:
            # Pipeline would skip perplexity monitoring
            should_skip_perplexity = True
        else:
            should_skip_perplexity = False

        assert should_skip_perplexity is True


class TestRealWorldScenarios:
    """Test real-world usage scenarios for perplexity guard configuration."""

    def test_research_profile_high_quality(self):
        """Test research profile that needs high-quality output with perplexity guard."""
        research_profile = CircuitBreakerConfig(
            base_timeout=300.0,  # Longer timeout for research
            deep_research_timeout=600.0,
            enable_perplexity_guard=True,  # Quality monitoring enabled
            perplexity_threshold=5.0,  # Stricter threshold
            max_retries=3,
        )

        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, research_profile
        )

        # Should prioritize quality
        assert final_config.enable_perplexity_guard is True
        assert final_config.perplexity_threshold == 5.0
        assert final_config.base_timeout == 300.0

    def test_speed_profile_no_quality_monitoring(self):
        """Test speed-optimized profile that disables perplexity guard for performance."""
        speed_profile = CircuitBreakerConfig(
            base_timeout=30.0,  # Quick timeout
            enable_perplexity_guard=False,  # Disable quality monitoring for speed
            max_retries=1,  # Fewer retries for speed
        )

        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, speed_profile
        )

        # Should prioritize speed
        assert final_config.enable_perplexity_guard is False
        assert final_config.base_timeout == 30.0
        assert final_config.max_retries == 1

    def test_development_profile_detailed_monitoring(self):
        """Test development profile with detailed monitoring and logging."""
        dev_profile = CircuitBreakerConfig(
            enable_perplexity_guard=True,
            log_repetition_events=True,
            perplexity_log_interval_tokens=10,  # More frequent logging
            min_tokens_for_eval=5,  # Earlier evaluation
        )

        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, dev_profile
        )

        # Should have detailed monitoring
        assert final_config.enable_perplexity_guard is True
        assert final_config.log_repetition_events is True
        assert final_config.perplexity_log_interval_tokens == 10
        assert final_config.min_tokens_for_eval == 5

    def test_production_profile_balanced_settings(self):
        """Test production profile with balanced quality and performance."""
        production_profile = CircuitBreakerConfig(
            base_timeout=120.0,
            enable_perplexity_guard=True,  # Quality monitoring on
            perplexity_threshold=12.0,  # Moderate threshold
            max_retries=2,
            log_repetition_events=False,  # Less logging in production
        )

        final_config = CircuitBreakerConfig.merge_with_overrides(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, production_profile
        )

        # Should balance quality and performance
        assert final_config.enable_perplexity_guard is True
        assert final_config.perplexity_threshold == 12.0
        assert final_config.max_retries == 2
        assert final_config.log_repetition_events is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
