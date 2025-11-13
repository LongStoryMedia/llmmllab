#!/usr/bin/env python3
"""
Debug script for Dynamic Flag Parser - Clean display of discovered llama.cpp flags

This script provides a clean, formatted view of all flags discovered by the
DynamicFlagParser from the llama-server --help output.
"""

import sys
from pathlib import Path

from runner.server_manager.dynamic_flag_parser import DynamicFlagParser


# Add the inference directory to Python path
inference_dir = Path(__file__).parent.parent
sys.path.insert(0, str(inference_dir))


def format_flag_name(flag_info):
    """Format flag names for display."""
    short_flags = flag_info.get("short_flags", [])
    long_flags = flag_info.get("long_flags", [])
    all_flags = short_flags + long_flags

    if not all_flags:
        return "NO_FLAGS"

    return ", ".join(all_flags)


def format_flag_type(flag_info):
    """Format flag type for display."""
    flag_type = flag_info.get("type")
    action = flag_info.get("action", "store")
    value_type = flag_info.get("value_type")

    if flag_type is None and action == "store_true":
        return "bool"
    elif flag_type:
        type_name = flag_type.__name__
        if value_type:
            return f"{type_name} ({value_type})"
        return type_name
    else:
        return "unknown"


def print_header():
    """Print the script header."""
    print("=" * 80)
    print("🔧 LLAMA.CPP DYNAMIC FLAG DISCOVERY DEBUG")
    print("=" * 80)
    print()


def print_summary(flags, executable_path):
    """Print summary statistics."""
    total_flags = len(flags)
    bool_flags = sum(
        1 for f in flags if f.get("type") is None and f.get("action") == "store_true"
    )
    int_flags = sum(1 for f in flags if f.get("type") is int)
    float_flags = sum(1 for f in flags if f.get("type") is float)
    str_flags = sum(1 for f in flags if f.get("type") is str)

    short_flags = sum(len(f.get("short_flags", [])) for f in flags)
    long_flags = sum(len(f.get("long_flags", [])) for f in flags)

    print("📊 SUMMARY")
    print(f"   Executable: {executable_path}")
    print(f"   Total flags discovered: {total_flags}")
    print(
        f"   Flag types: {bool_flags} bool, {int_flags} int, {float_flags} float, {str_flags} str"
    )
    print(f"   Flag forms: {short_flags} short (-x), {long_flags} long (--xxx)")
    print()


def print_flags_by_type(flags):
    """Print flags organized by type."""
    # Group flags by type
    flag_groups = {
        "Boolean Flags": [],
        "Integer Flags": [],
        "Float Flags": [],
        "String Flags": [],
        "Other Flags": [],
    }

    for flag in flags:
        flag_type = flag.get("type")
        action = flag.get("action", "store")

        if flag_type is None and action == "store_true":
            flag_groups["Boolean Flags"].append(flag)
        elif flag_type is int:
            flag_groups["Integer Flags"].append(flag)
        elif flag_type is float:
            flag_groups["Float Flags"].append(flag)
        elif flag_type is str:
            flag_groups["String Flags"].append(flag)
        else:
            flag_groups["Other Flags"].append(flag)

    # Print each group
    for group_name, group_flags in flag_groups.items():
        if not group_flags:
            continue

        print(f"📁 {group_name.upper()} ({len(group_flags)} flags)")
        print("-" * 60)

        for flag in group_flags:
            flag_names = format_flag_name(flag)
            flag_type = format_flag_type(flag)
            description = flag.get("help", "No description")

            # Truncate long descriptions
            if len(description) > 50:
                description = description[:47] + "..."

            print(f"  {flag_names:<25} {flag_type:<12} {description}")

        print()


def print_flags_alphabetically(flags):
    """Print all flags in alphabetical order."""
    print("📝 ALL FLAGS (ALPHABETICAL)")
    print("-" * 80)

    # Sort flags by their first flag name
    sorted_flags = sorted(flags, key=lambda f: format_flag_name(f).lower())

    for i, flag in enumerate(sorted_flags, 1):
        flag_names = format_flag_name(flag)
        flag_type = format_flag_type(flag)
        description = flag.get("help", "No description")

        # Truncate long descriptions
        if len(description) > 40:
            description = description[:37] + "..."

        print(f"{i:3}. {flag_names:<30} {flag_type:<12} {description}")

    print()


def main():
    """Main debug function."""
    print_header()

    # Default executable path - can be overridden via command line
    executable_path = "/llama.cpp/build/bin/llama-server"

    if len(sys.argv) > 1:
        executable_path = sys.argv[1]

    print(f"🔍 Discovering flags from: {executable_path}")
    print()

    try:
        # Create parser and discover flags
        parser = DynamicFlagParser(executable_path)
        flags = parser.parse_flags()

        if not flags:
            print(
                "❌ No flags discovered. Check if the executable exists and provides --help output."
            )
            return 1

        # Print summary
        print_summary(flags, executable_path)

        # Print flags by type
        print_flags_by_type(flags)

        # Print all flags alphabetically
        print_flags_alphabetically(flags)

        print("✅ Dynamic flag discovery completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error during flag discovery: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
