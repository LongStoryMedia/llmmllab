"""
Unit tests for DynamicFlagParser - Dynamic llama.cpp flag discovery system
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import subprocess

from runner.server_manager.argument_builder import DynamicFlagParser


class TestDynamicFlagParser(unittest.TestCase):
    """Test suite for DynamicFlagParser functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.executable_path = "/test/path/llama-server"
        self.parser = DynamicFlagParser(self.executable_path)

    def test_init(self):
        """Test parser initialization."""
        self.assertEqual(self.parser.executable_path, self.executable_path)
        self.assertIsNone(self.parser.parsed_flags)

    @patch('runner.server_manager.dynamic_flag_parser.subprocess.run')
    def test_get_help_output_success(self, mock_run):
        """Test successful help output retrieval."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test help output"
        mock_run.return_value = mock_result

        result = self.parser.get_help_output()
        
        self.assertEqual(result, "test help output")
        mock_run.assert_called_once_with(
            [self.executable_path, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )

    @patch('runner.server_manager.dynamic_flag_parser.subprocess.run')
    def test_get_help_output_failure(self, mock_run):
        """Test help output retrieval failure."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
        
        result = self.parser.get_help_output()
        
        self.assertEqual(result, "")

    def test_parse_flags_with_sample_help(self):
        """Test flag parsing with realistic help output."""
        sample_help = """Usage: llama-server [options]

----- common params -----
-h,    --help, --usage                  print usage and exit
--version                               show version and build info
-t,    --threads N                      number of CPU threads to use during generation (default: -1)
--verbose-prompt                        print a verbose prompt before generation (default: false)
-c,    --ctx-size N                     size of the prompt context (default: 4096)
--temp N                                temperature (default: 0.8)
--override-tensor, -ot <tensor name pattern>=<buffer type>,...
--no-mmproj

----- sampling params -----
--top-k N                               top-k sampling (default: 40, 0 = disabled)
"""

        # Mock the help output
        with patch.object(self.parser, 'get_help_output', return_value=sample_help):
            flags = self.parser.parse_flags()

        # Should find all valid flags
        self.assertGreater(len(flags), 0)
        
        # Check specific flags
        flag_names = []
        for flag in flags:
            flag_names.extend(flag.get('short_flags', []))
            flag_names.extend(flag.get('long_flags', []))

        # Verify key flags are found
        self.assertIn('-h', flag_names)
        self.assertIn('--help', flag_names)
        self.assertIn('--version', flag_names)
        self.assertIn('-t', flag_names)
        self.assertIn('--threads', flag_names)
        self.assertIn('--verbose-prompt', flag_names)
        self.assertIn('--override-tensor', flag_names)
        self.assertIn('-ot', flag_names)

    def test_flag_type_inference(self):
        """Test type inference for different flag types."""
        test_cases = [
            # Integer flag with N value type
            ("-t, --threads N                      number of CPU threads to use", int),
            # Float flag based on description keywords
            ("--temp                                temperature value for sampling", float),
            # String flag with FNAME value type  
            ("-f, --file FNAME                     path to model file", str),
            # Boolean flag (no description splitting)
            ("--verbose", None),
        ]

        for sample_help, expected_type in test_cases:
            with self.subTest(sample_help=sample_help):
                with patch.object(self.parser, 'get_help_output', return_value=sample_help):
                    self.parser.parsed_flags = None  # Reset cache
                    flags = self.parser.parse_flags()
                    if flags and len(flags) > 0:
                        actual_type = flags[0]['type'] 
                        self.assertEqual(actual_type, expected_type, 
                                       f"Expected {expected_type}, got {actual_type} for flag: {sample_help}")

    def test_build_parser_integration(self):
        """Test building argparse parser with discovered flags."""
        sample_flags = [
            {
                'short_flags': ['-t'],
                'long_flags': ['--threads'],
                'type': int,
                'action': 'store',
                'help': 'Number of threads',
                'takes_value': True,
                'value_type': 'N'
            },
            {
                'short_flags': [],
                'long_flags': ['--verbose'],
                'type': None,
                'action': 'store_true',
                'help': 'Verbose output',
                'takes_value': False,
                'value_type': None
            }
        ]

        # Mock the parse_flags method
        with patch.object(self.parser, 'parse_flags', return_value=sample_flags):
            base_parser = argparse.ArgumentParser()
            self.parser.build_parser(base_parser)

            # Test that arguments can be parsed
            args = base_parser.parse_args(['-t', '8', '--verbose'])
            self.assertEqual(args.threads, 8)
            self.assertTrue(args.verbose)

    def test_caching_behavior(self):
        """Test that flags are cached after first parse."""
        with patch.object(self.parser, 'get_help_output', return_value="-t, --threads N\n") as mock_help:
            # First call should fetch help output
            flags1 = self.parser.parse_flags()
            self.assertEqual(mock_help.call_count, 1)

            # Second call should use cached results
            flags2 = self.parser.parse_flags()
            self.assertEqual(mock_help.call_count, 1)  # Should not call again
            
            # Results should be identical
            self.assertEqual(flags1, flags2)

    def test_empty_help_output(self):
        """Test handling of empty help output."""
        with patch.object(self.parser, 'get_help_output', return_value=""):
            flags = self.parser.parse_flags()
            self.assertEqual(flags, [])

    def test_malformed_flag_lines(self):
        """Test handling of malformed flag lines."""
        malformed_help = """
----- section header -----
- not a real flag
--good-flag                             good flag description
malformed line without dashes
--another-good-flag N                   another good flag
"""
        
        with patch.object(self.parser, 'get_help_output', return_value=malformed_help):
            flags = self.parser.parse_flags()
            
            # Should only find the good flags
            flag_names = []
            for flag in flags:
                flag_names.extend(flag.get('long_flags', []))
            
            self.assertIn('--good-flag', flag_names)
            self.assertIn('--another-good-flag', flag_names)
            self.assertEqual(len(flags), 2)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)