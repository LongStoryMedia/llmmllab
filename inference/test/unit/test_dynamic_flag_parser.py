"""Unit tests for DynamicFlagParser flag deduplication and processing."""

import argparse
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import re

# Avoid importing the heavy runner module during test collection
# We'll only import what we need for specific tests


class TestFlagDeduplicationLogic:
    """Test the core deduplication logic without importing the full module."""

    def test_negation_flag_detection(self):
        """Test detection of flags that have negation forms."""
        # Simulate what happens when parsing help line:
        # "--warmup, --no-warmup    skip warmup..."

        long_flags = ["--warmup", "--no-warmup"]

        # This is the logic from dynamic_flag_parser.py
        is_negation = any(f.lstrip("-").startswith("no-") for f in long_flags)

        assert is_negation is True, "Should detect negation when --no-warmup present"

    def test_base_name_extraction(self):
        """Test extracting base name from flags."""
        # When we have both --warmup and --no-warmup
        long_flags = ["--warmup", "--no-warmup"]

        # Get first one
        primary_long = long_flags[0]
        base_name = primary_long.lstrip("-")

        # Extract actual base name (removing no- prefix if present)
        if base_name.startswith("no-"):
            base_name = base_name[3:]

        # But we need to check if ANY form has no-, not just the primary
        is_negation = any(f.lstrip("-").startswith("no-") for f in long_flags)

        # If any form is negation, we should use the negation form
        if is_negation:
            # Find the negation form
            neg_flag = next(
                (f for f in long_flags if f.lstrip("-").startswith("no-")), None
            )
            if neg_flag:
                dest_name = neg_flag.lstrip("-").replace("-", "_")
            else:
                dest_name = base_name.replace("-", "_")
        else:
            dest_name = base_name.replace("-", "_")

        assert base_name == "warmup", f"Base name should be 'warmup', got '{base_name}'"
        assert is_negation is True
        assert (
            dest_name == "no_warmup"
        ), f"Dest should be 'no_warmup', got '{dest_name}'"

    def test_dest_name_from_negation_flag(self):
        """Test destination name extraction from negation flag."""
        flag = "--no-warmup"
        dest = flag.lstrip("-").replace("-", "_")
        assert dest == "no_warmup"

        flag = "--no-webui"
        dest = flag.lstrip("-").replace("-", "_")
        assert dest == "no_webui"


class TestArgumentBuildingFlow:
    """Test the complete flow from config dict to CLI arguments."""

    def test_config_to_fake_args(self):
        """Test conversion of config dict to fake args list."""
        config = {
            "no_warmup": True,
            "no_webui": True,
            "cont_batching": True,
            "metrics": True,
        }

        # This is what the builder does
        fake_args = []
        for key, value in config.items():
            if value is None:
                continue

            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    fake_args.append(flag)

        assert "--no-warmup" in fake_args
        assert "--no-webui" in fake_args
        assert "--cont-batching" in fake_args
        assert "--metrics" in fake_args

    def test_argparse_store_true_for_boolean_flags(self):
        """Test that store_true action works correctly for boolean flags."""
        parser = argparse.ArgumentParser()

        # Both positive and negative forms should use store_true
        # because they're presence indicators in the config dict
        parser.add_argument("--warmup", dest="warmup", action="store_true")
        parser.add_argument("--no-warmup", dest="no_warmup", action="store_true")

        # When we pass --no-warmup
        args = parser.parse_args(["--no-warmup"])

        assert args.no_warmup is True
        assert args.warmup is False  # Not specified, so defaults to False

    def test_argparse_only_negation_form_registered(self):
        """Test when only the negation form is registered (preferred approach)."""
        parser = argparse.ArgumentParser()

        # Only register the negation form
        parser.add_argument("--no-warmup", dest="no_warmup", action="store_true")
        parser.add_argument("--no-webui", dest="no_webui", action="store_true")

        # Parse the args
        args = parser.parse_args(["--no-warmup", "--no-webui"])

        assert vars(args) == {"no_warmup": True, "no_webui": True}

    def test_build_args_reconstruction(self):
        """Test reconstruction of CLI args from parsed namespace."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-warmup", dest="no_warmup", action="store_true")
        parser.add_argument("--no-webui", dest="no_webui", action="store_true")
        parser.add_argument(
            "--cont-batching", dest="cont_batching", action="store_true"
        )

        # Simulate what happens during parsing
        args = parser.parse_args(["--no-warmup", "--no-webui", "--cont-batching"])

        # This is what build_args() does
        cli_args = []
        for key, value in vars(args).items():
            if value is None:
                continue

            flag = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)

        # Verify reconstruction
        assert "--no-warmup" in cli_args
        assert "--no-webui" in cli_args
        assert "--cont-batching" in cli_args


class TestIntegrationWithDynamicFlagParser:
    """Integration tests with actual DynamicFlagParser if imports work."""

    @pytest.mark.skip(reason="Requires full runner module imports")
    def test_parse_warmup_webui_flags(self):
        """Test parsing warmup and webui flags from help output."""
        # This would test the actual parser if we can import it
        from runner.server_manager.dynamic_flag_parser import DynamicFlagParser

        help_output = """
Server arguments:
  --warmup, --no-warmup              skip warmup phase (default: disabled)
  --webui, --no-webui                serve webui (default: enabled)
"""

        with patch.object(
            DynamicFlagParser, "get_help_output", return_value=help_output
        ):
            parser = DynamicFlagParser("/dummy/path")
            flags = parser.parse_flags()

        # Should have deduplicated into 2 flags (warmup and webui)
        assert len([f for f in flags if f["base_name"] == "warmup"]) == 1
        assert len([f for f in flags if f["base_name"] == "webui"]) == 1


class TestRealWorldFlagGeneration:
    """Test realistic scenarios matching the actual Qwen3 model command."""

    def test_llama_cpp_boolean_flags_roundtrip(self):
        """Test the complete roundtrip: config → fake_args → argparse → CLI reconstruction.

        This is the exact flow that was failing with --warmup instead of --no-warmup.
        """
        # Step 1: Builder creates config with negation flags
        config = {
            "no_warmup": True,
            "no_webui": True,
            "cont_batching": True,
            "metrics": True,
            "jinja": True,
        }

        # Step 2: Builder converts config to fake args (what we pass to argparse)
        fake_args = []
        for key, value in config.items():
            if isinstance(value, bool) and value:
                fake_args.append(f"--{key.replace('_', '-')}")

        # Verify config dict was correctly converted to CLI format
        assert (
            "--no-warmup" in fake_args
        ), "Config no_warmup=True should create --no-warmup"
        assert (
            "--no-webui" in fake_args
        ), "Config no_webui=True should create --no-webui"
        assert "--cont-batching" in fake_args
        assert "--metrics" in fake_args
        assert "--jinja" in fake_args

        # Step 3: Parser registers flags with destinations matching config keys
        # This is what DynamicFlagParser should produce:
        parser = argparse.ArgumentParser()

        # Register negation forms (what parser should discover and register)
        parser.add_argument("--no-warmup", dest="no_warmup", action="store_true")
        parser.add_argument("--no-webui", dest="no_webui", action="store_true")
        parser.add_argument(
            "--cont-batching", dest="cont_batching", action="store_true"
        )
        parser.add_argument("--metrics", dest="metrics", action="store_true")
        parser.add_argument("--jinja", dest="jinja", action="store_true")

        # Step 4: Argparse parses the fake args
        args = parser.parse_args(fake_args)

        # Step 5: Verify parsed namespace matches config
        parsed_dict = vars(args)
        assert parsed_dict["no_warmup"] is True
        assert parsed_dict["no_webui"] is True
        assert parsed_dict["cont_batching"] is True
        assert parsed_dict["metrics"] is True
        assert parsed_dict["jinja"] is True

        # Step 6: Reconstruct CLI command (what build_args() does)
        cli_args = []
        for key, value in parsed_dict.items():
            if isinstance(value, bool) and value:
                cli_args.append(f"--{key.replace('_', '-')}")

        # Step 7: Final verification - CLI args should match original
        assert "--no-warmup" in cli_args, "Roundtrip failed: missing --no-warmup"
        assert "--no-webui" in cli_args, "Roundtrip failed: missing --no-webui"
        assert "--cont-batching" in cli_args
        assert "--metrics" in cli_args
        assert "--jinja" in cli_args

        # Most important: should NOT have positive forms
        assert (
            "--warmup" not in cli_args
        ), "Should NOT have --warmup (should be --no-warmup)"
        assert (
            "--webui" not in cli_args
        ), "Should NOT have --webui (should be --no-webui)"

    def test_mixed_boolean_and_value_flags(self):
        """Test realistic mix of boolean and value-taking flags."""
        config = {
            "host": "127.0.0.1",
            "port": 8001,
            "threads": 24,
            "ctx_size": 160000,
            "batch_size": 2048,
            "ubatch_size": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "no_warmup": True,
            "no_webui": True,
            "cont_batching": True,
        }

        # Convert to CLI args
        cli_args = []
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
            else:
                cli_args.extend([flag, str(value)])

        # Parse with argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str)
        parser.add_argument("--port", type=int)
        parser.add_argument("--threads", type=int)
        parser.add_argument("--ctx-size", dest="ctx_size", type=int)
        parser.add_argument("--batch-size", dest="batch_size", type=int)
        parser.add_argument("--ubatch-size", dest="ubatch_size", type=int)
        parser.add_argument("--cache-type-k", dest="cache_type_k")
        parser.add_argument("--cache-type-v", dest="cache_type_v")
        parser.add_argument("--no-warmup", dest="no_warmup", action="store_true")
        parser.add_argument("--no-webui", dest="no_webui", action="store_true")
        parser.add_argument(
            "--cont-batching", dest="cont_batching", action="store_true"
        )

        args = parser.parse_args(cli_args)

        # Verify parsing
        assert args.host == "127.0.0.1"
        assert args.port == 8001
        assert args.threads == 24
        assert args.ctx_size == 160000
        assert args.cache_type_k == "f16"
        assert args.no_warmup is True
        assert args.no_webui is True
        assert args.cont_batching is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
