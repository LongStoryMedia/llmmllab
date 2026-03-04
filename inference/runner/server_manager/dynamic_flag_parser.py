"""
Dynamic Flag Parser - Discovers llama.cpp server flags from help output.

This module provides dynamic flag discovery by parsing the help output from
llama.cpp executables, allowing automatic adaptation to new flags without
manual updates.
"""

import argparse
import re
import subprocess
from typing import Any, Dict, List, Optional
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="DynamicFlagParser")


class DynamicFlagParser:
    """Dynamically parse llama.cpp server flags from help output."""

    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.parsed_flags = None

    def get_help_output(self) -> str:
        """Get the help output from the llama-server executable."""
        try:
            result = subprocess.run(
                [self.executable_path, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            if result.returncode == 0:
                return result.stdout
            else:
                logger.warning(
                    f"Help command failed with code {result.returncode}: {result.stderr}"
                )
                return ""
        except Exception as e:
            logger.error(f"Failed to get help output from {self.executable_path}: {e}")
            return ""

    def parse_flags(self) -> List[Dict[str, Any]]:
        """Parse flag definitions from help output."""
        if self.parsed_flags is not None:
            return self.parsed_flags

        help_output = self.get_help_output()
        if not help_output:
            logger.warning("No help output available, falling back to static flags")
            return []

        flags = []
        flag_map = {}  # Track flags by their base name for deduplication

        # Parse line by line looking for flag definitions
        lines = help_output.split("\n")

        for line in lines:
            original_line = line
            line = line.strip()

            # Skip empty lines, section headers, and non-flag lines
            # Only accept lines that start with flags at the beginning (no indentation)
            # and have proper flag format: -x or --xxx followed by space, comma, or end
            if (
                not line
                or line.startswith("-----")
                or original_line.startswith(" ")  # Skip indented lines (descriptions)
                or original_line.startswith("\t")  # Skip tab-indented lines
                or not re.match(r"^-{1,2}[a-zA-Z]", line)
            ):
                continue

            # Parse flag line - format is typically:
            # "-t, --threads N                      number of CPU threads..."
            # "--help                               print usage and exit"
            # "-ngl, --gpu-layers, --n-gpu-layers N  max. number of layers..."

            # Split to separate flags from description - look for pattern where
            # flags end and description begins (description doesn't start with -)
            parts = None

            # Try to split on large whitespace gaps, but ensure the description part
            # doesn't start with a flag (-)
            for potential_split in re.finditer(r"\s{2,}", line):
                start = potential_split.start()
                potential_desc = line[potential_split.end() :].strip()

                # Description shouldn't start with a flag marker
                if potential_desc and not potential_desc.startswith("-"):
                    flag_spec = line[:start].strip()
                    description = potential_desc
                    parts = [flag_spec, description]
                    break

            # If no description found on same line, treat entire line as flag spec
            if not parts or len(parts) < 2:
                # This line contains only flag specification, no description
                flag_spec = line.strip()
                description = ""  # Empty description
                parts = [flag_spec, description]

            flag_spec = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""

            # Parse the flag specification with improved logic
            short_flags = []
            long_flags = []
            value_type = None

            # Split on comma but respect braces - don't split commas inside {}
            flag_parts = []
            current_part = ""
            brace_depth = 0

            for char in flag_spec:
                if char == "{":
                    brace_depth += 1
                    current_part += char
                elif char == "}":
                    brace_depth -= 1
                    current_part += char
                elif char == "," and brace_depth == 0:
                    # Only split on comma if we're not inside braces
                    if current_part.strip():
                        flag_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char

            # Add the last part
            if current_part.strip():
                flag_parts.append(current_part.strip())

            for part in flag_parts:
                part = part.strip()

                # Split the part to separate flag name from value type
                tokens = part.split()
                if not tokens:
                    continue

                flag_name = tokens[0]

                # Validate flag name format - must be proper flag format
                # Short flags: -x (single letter after dash)
                # Long flags: --word (word characters, hyphens, underscores, dots after --)
                if flag_name.startswith("--"):
                    if not re.match(r"^--[a-zA-Z][a-zA-Z0-9_.-]*$", flag_name):
                        continue  # Skip malformed long flags like --threads)
                elif flag_name.startswith("-"):
                    if not re.match(r"^-[a-zA-Z][a-zA-Z0-9]*$", flag_name):
                        continue  # Skip malformed short flags
                else:
                    continue  # Skip anything that doesn't start with - or --

                # Check for value type after flag name
                if len(tokens) > 1:
                    potential_value_type = tokens[1]
                    # Check for choice patterns like {none,layer,row} or [on|off|auto]
                    if potential_value_type.startswith(
                        "{"
                    ) and potential_value_type.endswith("}"):
                        value_type = potential_value_type
                    elif (
                        potential_value_type.startswith("[")
                        and potential_value_type.endswith("]")
                        and "|" in potential_value_type
                    ):
                        value_type = potential_value_type
                    elif potential_value_type.upper() in [
                        "N",
                        "TYPE",
                        "SEED",
                        "FNAME",
                        "HOST",
                        "PORT",
                        "PATH",
                        "PREFIX",
                        "KEY",
                        "TOKEN",
                        "STRING",
                        "SCHEMA",
                        "FILE",
                        "URL",
                        "SCALE",
                        "INDEX",
                        "SIMILARITY",
                        "FORMAT",
                        "SEQUENCE",
                        "SAMPLERS",
                        "PROMPT",
                        "GRAMMAR",
                        "BIAS",
                        "JINJA_TEMPLATE",
                        "JINJA_TEMPLATE_FILE",
                        "M",
                        "P",
                        "<0|1>",
                        "<0...100>",
                        "LO-HI",
                    ]:
                        value_type = potential_value_type

                # Categorize flag
                if flag_name.startswith("--"):
                    long_flags.append(flag_name)
                elif flag_name.startswith("-") and len(flag_name) > 1:
                    short_flags.append(flag_name)

            # Determine if this flag takes a value
            takes_value = value_type is not None

            # Enhance value type detection from description
            if not takes_value:
                desc_lower = description.lower()
                # Check for patterns that indicate boolean flags (overrides other patterns)
                if any(
                    pattern in desc_lower
                    for pattern in [
                        "(default: disabled)",
                        "(default: enabled)",
                        "enable ",
                        "disable ",
                        "restrict to only",
                    ]
                ):
                    takes_value = False
                elif any(
                    pattern in desc_lower
                    for pattern in [
                        "number of",
                        "size of",
                        "path to",
                        "url",
                        "file",
                        "directory",
                        "timeout",
                        "port",
                        "host",
                        "value",
                        "factor",
                        "probability",
                        "temperature",
                        "scale",
                        "rate",
                        "threshold",
                    ]
                ):
                    takes_value = True

            if long_flags or short_flags:
                # Extract base flag name for deduplication
                # For --no-XXX flags, base is XXX; for --xxx flags, base is xxx
                primary_long = (
                    long_flags[0]
                    if long_flags
                    else (short_flags[0] if short_flags else "")
                )
                base_name = primary_long.lstrip("-")

                # Check if ANY form is a negation flag (not just the primary)
                # This handles cases like "--warmup, --no-warmup" on the same help line
                is_negation = any(
                    f.lstrip("-").startswith("no-") for f in (long_flags + short_flags)
                )

                # Extract actual base name (without no- prefix if present)
                if base_name.startswith("no-"):
                    base_name = base_name[3:]

                # Determine argument type (now that we know if it's a negation flag)
                arg_type = self._infer_argument_type(
                    description, takes_value, value_type, is_negation
                )

                flag_info = {
                    "base_name": base_name,
                    "short_flags": short_flags,
                    "long_flags": long_flags,
                    "type": arg_type["type"],
                    "action": arg_type["action"],
                    "help": description,
                    "takes_value": takes_value,
                    "value_type": value_type,
                    "is_negation": is_negation,
                }

                # Store or update in map (negation variants override positive forms)
                if base_name not in flag_map or flag_info["is_negation"]:
                    flag_map[base_name] = flag_info

        # Convert map to list
        flags = list(flag_map.values())

        self.parsed_flags = flags
        logger.info(f"Parsed {len(flags)} flags from llama.cpp help output")
        return flags

    def _infer_argument_type(
        self,
        description: str,
        takes_value: bool,
        value_type: Optional[str] = None,
        is_negation: bool = False,
    ) -> Dict[str, Any]:
        """Infer argument type from description and value requirement.

        CRITICAL: For boolean flags, ALWAYS use store_true action!
        The is_negation parameter is accepted but NOT used for action determination.

        Reason: The config dict uses semantics like config["no_warmup"] = True to mean
        "include the --no-warmup flag in the command". This is a PRESENCE indicator.
        - store_true action: if flag present in argv, set destination to True
        - This correctly represents: flag present (True) or absent (False default)
        - store_false is NOT used because flag absence is already default (False)

        The destination name (e.g., "no_warmup") handles the negation semantics;
        the action only indicates presence.
        """
        desc_lower = description.lower()

        # Boolean flags (no value) - only for flags that actually don't take values
        if not takes_value:
            # ALWAYS use store_true for boolean flags, regardless of is_negation
            return {"type": None, "action": "store_true"}

        # Choice patterns like {none,layer,row} or [on|off|auto] - treat as string
        if value_type and (
            (value_type.startswith("{") and value_type.endswith("}"))
            or (
                value_type.startswith("[")
                and value_type.endswith("]")
                and "|" in value_type
            )
        ):
            return {"type": str, "action": "store"}

        # String flags based on value type indicators (highest priority)
        if value_type and value_type.upper() in [
            "FNAME",
            "FILE",
            "PATH",
            "URL",
            "HOST",
            "TOKEN",
            "KEY",
            "STRING",
            "SCHEMA",
            "SAMPLERS",
            "PROMPT",
            "GRAMMAR",
            "BIAS",
            "SEQUENCE",
            "JINJA_TEMPLATE",
            "JINJA_TEMPLATE_FILE",
            "FORMAT",
            "PREFIX",
            "TYPE",
            "SEED",
            "SCALE",
            "INDEX",
            "SIMILARITY",
            "M",
            "LO-HI",
            "<0|1>",
            "<0...100>",
            "<DEV1",  # Special patterns
        ]:
            return {"type": str, "action": "store"}

        # Integer flags based on value type indicator
        if value_type == "N" or any(
            word in desc_lower
            for word in [
                "number",
                "size",
                "count",
                "threads",
                "layers",
                "index",
                "port",
                "timeout",
            ]
        ):
            return {"type": int, "action": "store"}

        # Float flags - be more specific to avoid false positives but include sampling parameters
        if value_type == "P" or any(
            word in desc_lower
            for word in [
                "temperature",
                "probability",
                "factor",
                "threshold",
                "penalty",
                "learning rate",
                "ratio",
                "sampling",
                "typical",
                "multiplier",
            ]
        ):
            return {"type": float, "action": "store"}

        # String flags (default for value-taking flags)
        return {"type": str, "action": "store"}

    def _get_dest_name(self, flag_info: Dict[str, Any]) -> str:
        """Get the destination name for a flag.

        Uses the FIRST flag form from the help output as the basis for the destination.
        This ensures consistency: when registering both --cont-batching and --no-cont-batching,
        they both map to destination "cont_batching".

        The builder will use config keys that match this destination name.
        """
        # Prefer long flags over short flags
        if flag_info.get("long_flags"):
            primary_flag = flag_info["long_flags"][0]
        elif flag_info.get("short_flags"):
            primary_flag = flag_info["short_flags"][0]
        else:
            raise ValueError(f"No valid flags found in flag_info: {flag_info}")

        # Strip leading dashes and convert hyphens to underscores for destination name
        dest = primary_flag.lstrip("-").replace("-", "_")
        return dest

    def build_parser(self, base_parser: argparse.ArgumentParser) -> None:
        """Add dynamically discovered flags to the argument parser.

        For flags with both positive and negative forms (e.g., --warmup, --no-warmup),
        we register them separately with individual destinations so that:
        - --warmup sets destination "warmup"
        - --no-warmup sets destination "no_warmup"

        This allows the builder to use whichever form it chooses via the config dict.
        """
        flags = self.parse_flags()

        added_count = 0
        for flag_info in flags:
            try:
                flag_names = flag_info["short_flags"] + flag_info["long_flags"]

                if not flag_names:
                    continue

                # For boolean flags with multiple forms, register each form separately
                # with its own destination
                if flag_info.get("action") == "store_true" and len(flag_names) > 1:
                    for flag_name in flag_names:
                        try:
                            # Each flag gets its own destination based on its form
                            dest = flag_name.lstrip("-").replace("-", "_")
                            kwargs = {
                                "help": flag_info["help"],
                                "dest": dest,
                                "action": "store_true",
                            }
                            base_parser.add_argument(flag_name, **kwargs)
                            added_count += 1
                        except argparse.ArgumentError as e:
                            logger.debug(f"Skipping flag {flag_name}: {e}")
                            continue
                else:
                    # For non-boolean flags or single-form boolean flags, use default logic
                    dest = self._get_dest_name(flag_info)

                    kwargs = {"help": flag_info["help"]}
                    kwargs["dest"] = dest

                    # Handle actions
                    action = flag_info.get("action", "store")
                    if action in ["store_true", "store_false"]:
                        kwargs["action"] = action
                    elif flag_info.get("type"):
                        kwargs["type"] = flag_info["type"]
                        kwargs["action"] = "store"
                    else:
                        kwargs["action"] = action

                    # Add the argument
                    base_parser.add_argument(*flag_names, **kwargs)
                    added_count += 1

            except argparse.ArgumentError as e:
                # Skip conflicting arguments (probably already defined)
                logger.debug(
                    f"Skipping conflicting flag {flag_info.get('long_flags', flag_info.get('short_flags'))}: {e}"
                )
                continue
            except Exception as e:
                logger.warning(f"Failed to add flag {flag_info}: {e}")
                continue

        logger.info(f"Added {added_count} dynamic flags to argument parser")
