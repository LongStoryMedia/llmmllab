"""
Test suite for CircuitBreakerConfig functionality.

This module tests the circuit breaker configuration system, including:
- Default configuration values
- Profile-specific overrides
- Merge functionality
- Perplexity guard enable/disable behavior
"""

import pytest
import sys
import os

# Add the parent directory to the sys.path to make imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.circuit_breaker_config import CircuitBreakerConfig
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG


class TestCircuitBreakerConfig:
    """Test cases for CircuitBreakerConfig model and functionality."""

    def test_default_config_values(self):
        """Test that the default configuration has expected values."""
        config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Test core timeout values
        assert config.base_timeout == 60.0
        assert config.deep_research_timeout == 120.0
        assert config.max_retries == 2
        assert config.cooldown_period == 30.0

        # Test perplexity guard settings
        assert config.enable_perplexity_guard is True
        assert config.perplexity_window == 40
        assert config.perplexity_threshold == 10.0
        assert config.avg_logprob_floor == -6.0

        # Test repetition detection settings
        assert config.repetition_ngram == 6
        assert config.repetition_threshold == 6
        assert config.min_tokens_for_eval == 20
        assert config.perplexity_log_interval_tokens == 20
        assert config.log_repetition_events is True

        # Test tool generation settings
        assert config.tool_gen_repetition_ngram == 4
        assert config.tool_gen_repetition_threshold == 3

    def test_optional_field_creation(self):
        """Test creating CircuitBreakerConfig with optional fields."""
        # Empty config (all None)
        empty_config = CircuitBreakerConfig()
        assert empty_config.base_timeout is None
        assert empty_config.enable_perplexity_guard is None
        assert empty_config.max_retries is None

        # Partial config
        partial_config = CircuitBreakerConfig(
            base_timeout=90.0, enable_perplexity_guard=False
        )
        assert partial_config.base_timeout == 90.0
        assert partial_config.enable_perplexity_guard is False
        assert partial_config.max_retries is None

    def test_perplexity_guard_disable(self):
        """Test that perplexity guard can be explicitly disabled."""
        # Create config with perplexity guard disabled
        disabled_config = CircuitBreakerConfig(enable_perplexity_guard=False)
        assert disabled_config.enable_perplexity_guard is False

        # Create config with perplexity guard enabled
        enabled_config = CircuitBreakerConfig(enable_perplexity_guard=True)
        assert enabled_config.enable_perplexity_guard is True

        # Create config with perplexity guard unset (None)
        unset_config = CircuitBreakerConfig(base_timeout=60.0)
        assert unset_config.enable_perplexity_guard is None

    def test_merge_with_overrides_none(self):
        """Test merging with None overrides returns base config."""
        base_config = DEFAULT_CIRCUIT_BREAKER_CONFIG
        merged = CircuitBreakerConfig.merge_with_overrides(base_config, None)

        # Should return the same config
        assert merged.base_timeout == base_config.base_timeout
        assert merged.enable_perplexity_guard == base_config.enable_perplexity_guard
        assert merged.max_retries == base_config.max_retries

    def test_merge_with_overrides_partial(self):
        """Test merging with partial overrides."""
        base_config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Override only timeout, keep other defaults
        override_config = CircuitBreakerConfig(base_timeout=180.0)
        merged = CircuitBreakerConfig.merge_with_overrides(base_config, override_config)

        # Timeout should be overridden
        assert merged.base_timeout == 180.0

        # Other values should remain from base config
        assert merged.enable_perplexity_guard == base_config.enable_perplexity_guard
        assert merged.max_retries == base_config.max_retries
        assert merged.cooldown_period == base_config.cooldown_period

    def test_merge_perplexity_guard_disable(self):
        """Test merging when override disables perplexity guard."""
        base_config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Base config has perplexity guard enabled
        assert base_config.enable_perplexity_guard is True

        # Override to disable perplexity guard
        override_config = CircuitBreakerConfig(enable_perplexity_guard=False)
        merged = CircuitBreakerConfig.merge_with_overrides(base_config, override_config)

        # Perplexity guard should be disabled in merged config
        assert merged.enable_perplexity_guard is False

        # Other values should remain from base config
        assert merged.base_timeout == base_config.base_timeout
        assert merged.max_retries == base_config.max_retries

    def test_merge_perplexity_guard_enable(self):
        """Test merging when override enables perplexity guard."""
        # Create a base config with perplexity guard disabled
        base_config = CircuitBreakerConfig(
            base_timeout=60.0, enable_perplexity_guard=False, max_retries=2
        )

        # Override to enable perplexity guard
        override_config = CircuitBreakerConfig(enable_perplexity_guard=True)
        merged = CircuitBreakerConfig.merge_with_overrides(base_config, override_config)

        # Perplexity guard should be enabled in merged config
        assert merged.enable_perplexity_guard is True

        # Other values should remain from base config
        assert merged.base_timeout == base_config.base_timeout
        assert merged.max_retries == base_config.max_retries

    def test_merge_perplexity_guard_unset(self):
        """Test merging when override doesn't specify perplexity guard."""
        base_config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Base config has perplexity guard enabled
        assert base_config.enable_perplexity_guard is True

        # Override only timeout, don't touch perplexity guard
        override_config = CircuitBreakerConfig(base_timeout=180.0)
        merged = CircuitBreakerConfig.merge_with_overrides(base_config, override_config)

        # Perplexity guard should remain enabled (from base)
        assert merged.enable_perplexity_guard is True

        # Timeout should be overridden
        assert merged.base_timeout == 180.0

    def test_merge_multiple_overrides(self):
        """Test merging with multiple field overrides."""
        base_config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # Override multiple fields including perplexity guard
        override_config = CircuitBreakerConfig(
            base_timeout=300.0,
            max_retries=5,
            enable_perplexity_guard=False,
            perplexity_threshold=15.0,
        )

        merged = CircuitBreakerConfig.merge_with_overrides(base_config, override_config)

        # All overridden fields should have new values
        assert merged.base_timeout == 300.0
        assert merged.max_retries == 5
        assert merged.enable_perplexity_guard is False
        assert merged.perplexity_threshold == 15.0

        # Non-overridden fields should retain base values
        assert merged.cooldown_period == base_config.cooldown_period
        assert merged.deep_research_timeout == base_config.deep_research_timeout

    def test_validation_constraints(self):
        """Test that validation constraints are enforced."""
        # Test valid values pass
        valid_config = CircuitBreakerConfig(
            base_timeout=30.0,  # Within 1.0-600.0 range
            max_retries=3,  # Within 0-10 range
            perplexity_threshold=5.0,  # Within 1.0-50.0 range
        )
        assert valid_config.base_timeout == 30.0
        assert valid_config.max_retries == 3
        assert valid_config.perplexity_threshold == 5.0

        # Test invalid values raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            CircuitBreakerConfig(base_timeout=0.5)  # Below minimum

        with pytest.raises(Exception):  # Pydantic ValidationError
            CircuitBreakerConfig(base_timeout=700.0)  # Above maximum

        with pytest.raises(Exception):  # Pydantic ValidationError
            CircuitBreakerConfig(max_retries=-1)  # Below minimum

        with pytest.raises(Exception):  # Pydantic ValidationError
            CircuitBreakerConfig(max_retries=15)  # Above maximum

    def test_model_dump_exclude_none(self):
        """Test that model_dump(exclude_none=True) works correctly."""
        config = CircuitBreakerConfig(
            base_timeout=90.0,
            enable_perplexity_guard=False,
            # max_retries intentionally not set (None)
        )

        data = config.model_dump(exclude_none=True)

        # Should include non-None values
        assert "base_timeout" in data
        assert "enable_perplexity_guard" in data
        assert data["base_timeout"] == 90.0
        assert data["enable_perplexity_guard"] is False

        # Should exclude None values
        assert "max_retries" not in data
        assert "cooldown_period" not in data


class TestPerplexityGuardScenarios:
    """Test specific scenarios for perplexity guard functionality."""

    def test_scenario_global_enabled_profile_disabled(self):
        """
        Scenario: Global setting has perplexity guard enabled,
        but a specific model profile disables it.
        """
        # Global config (enabled by default)
        global_config = DEFAULT_CIRCUIT_BREAKER_CONFIG
        assert global_config.enable_perplexity_guard is True

        # Profile specifically disables perplexity guard
        profile_override = CircuitBreakerConfig(enable_perplexity_guard=False)

        # Merge should result in disabled perplexity guard
        final_config = CircuitBreakerConfig.merge_with_overrides(
            global_config, profile_override
        )
        assert final_config.enable_perplexity_guard is False

    def test_scenario_global_disabled_profile_enabled(self):
        """
        Scenario: Global setting has perplexity guard disabled,
        but a specific model profile enables it.
        """
        # Global config with disabled perplexity guard
        global_config = CircuitBreakerConfig(
            base_timeout=60.0, enable_perplexity_guard=False, max_retries=2
        )

        # Profile specifically enables perplexity guard
        profile_override = CircuitBreakerConfig(enable_perplexity_guard=True)

        # Merge should result in enabled perplexity guard
        final_config = CircuitBreakerConfig.merge_with_overrides(
            global_config, profile_override
        )
        assert final_config.enable_perplexity_guard is True

    def test_scenario_global_enabled_profile_unset(self):
        """
        Scenario: Global setting has perplexity guard enabled,
        profile doesn't specify it (should use global).
        """
        # Global config (enabled by default)
        global_config = DEFAULT_CIRCUIT_BREAKER_CONFIG
        assert global_config.enable_perplexity_guard is True

        # Profile overrides other settings but not perplexity guard
        profile_override = CircuitBreakerConfig(base_timeout=180.0, max_retries=5)

        # Merge should keep global perplexity guard setting
        final_config = CircuitBreakerConfig.merge_with_overrides(
            global_config, profile_override
        )
        assert final_config.enable_perplexity_guard is True
        assert final_config.base_timeout == 180.0  # Override applied
        assert final_config.max_retries == 5  # Override applied

    def test_scenario_no_profile_overrides(self):
        """
        Scenario: No profile overrides at all (None).
        Should use global settings completely.
        """
        # Global config
        global_config = DEFAULT_CIRCUIT_BREAKER_CONFIG

        # No profile overrides
        final_config = CircuitBreakerConfig.merge_with_overrides(global_config, None)

        # Should be identical to global config
        assert (
            final_config.enable_perplexity_guard
            == global_config.enable_perplexity_guard
        )
        assert final_config.base_timeout == global_config.base_timeout
        assert final_config.max_retries == global_config.max_retries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
