#!/usr/bin/env python3
"""
Kubernetes deployment environment variables validation script.
Tests that all required composer environment variables are set correctly.
"""
import os
import sys


def test_composer_env_vars():
    """Test composer service environment variables."""
    print("🧪 Testing Composer Environment Variables...")

    # System configuration (required for service startup)
    system_vars = {
        "COMPOSER_HOST": "0.0.0.0",
        "COMPOSER_PORT": "8001",
        "COMPOSER_DEBUG": ["true", "false"],
        "COMPOSER_RELOAD": ["true", "false"],
        "COMPOSER_LOG_LEVEL": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "COMPOSER_ENABLE_CORS": ["true", "false"],
        "COMPOSER_VENV": "/opt/venv/composer",
    }

    # Workflow defaults (user-customizable via UI)
    workflow_vars = {
        "COMPOSER_ENABLE_CACHE": ["true", "false"],
        "COMPOSER_CACHE_TTL": lambda x: x.isdigit() and 60 <= int(x) <= 86400,
        "COMPOSER_MAX_PARALLEL_TOOLS": lambda x: x.isdigit() and 1 <= int(x) <= 20,
        "COMPOSER_ENABLE_MULTI_AGENT": ["true", "false"],
        "COMPOSER_DEFAULT_TIMEOUT": lambda x: is_float(x) and 5.0 <= float(x) <= 300.0,
        "COMPOSER_MAX_CONTEXT_LENGTH": lambda x: x.isdigit()
        and 1000 <= int(x) <= 1000000,
        "COMPOSER_CONTEXT_TRIM_THRESHOLD": lambda x: is_float(x)
        and 0.1 <= float(x) <= 1.0,
        "COMPOSER_ENABLE_STREAMING": ["true", "false"],
        "COMPOSER_STREAM_BUFFER_SIZE": lambda x: x.isdigit() and 256 <= int(x) <= 8192,
    }

    # Tool defaults (user-customizable via UI)
    tool_vars = {
        "COMPOSER_TOOL_SIMILARITY_THRESHOLD": lambda x: is_float(x)
        and 0.1 <= float(x) <= 1.0,
        "COMPOSER_TOOL_MODIFICATION_THRESHOLD": lambda x: is_float(x)
        and 0.1 <= float(x) <= 1.0,
        "COMPOSER_ENABLE_TOOL_GENERATION": ["true", "false"],
        "COMPOSER_MAX_TOOL_RETRIES": lambda x: x.isdigit() and 0 <= int(x) <= 10,
        "COMPOSER_TOOL_TIMEOUT": lambda x: is_float(x) and 1.0 <= float(x) <= 120.0,
        "COMPOSER_ENABLE_TOOL_CACHING": ["true", "false"],
        "COMPOSER_TOOL_CACHE_TTL": lambda x: x.isdigit() and 60 <= int(x) <= 7200,
        "COMPOSER_ENABLE_SEMANTIC_SEARCH": ["true", "false"],
        "COMPOSER_SEARCH_TOP_K": lambda x: x.isdigit() and 1 <= int(x) <= 50,
    }

    all_vars = {**system_vars, **workflow_vars, **tool_vars}
    errors = []

    for var_name, expected in all_vars.items():
        value = os.getenv(var_name)

        if value is None:
            errors.append(f"❌ {var_name}: Missing environment variable")
            continue

        # Validate value
        if isinstance(expected, list):
            if value not in expected:
                errors.append(f"❌ {var_name}={value}: Must be one of {expected}")
            else:
                print(f"✅ {var_name}={value}")
        elif callable(expected):
            if not expected(value):
                errors.append(f"❌ {var_name}={value}: Invalid value format or range")
            else:
                print(f"✅ {var_name}={value}")
        else:
            if value != expected:
                errors.append(f"❌ {var_name}={value}: Expected {expected}")
            else:
                print(f"✅ {var_name}={value}")

    return errors


def is_float(value):
    """Check if string can be converted to float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def test_database_connectivity():
    """Test database environment variables for composer."""
    print("\n🧪 Testing Database Environment Variables...")

    db_vars = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    redis_vars = ["REDIS_HOST", "REDIS_PORT", "REDIS_DB"]

    errors = []

    for var in db_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={'*' * len(value) if 'PASSWORD' in var else value}")
        else:
            errors.append(f"❌ {var}: Missing database environment variable")

    for var in redis_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            errors.append(f"❌ {var}: Missing Redis environment variable")

    return errors


def test_virtual_environment():
    """Test virtual environment configuration."""
    print("\n🧪 Testing Virtual Environment Configuration...")

    composer_venv = os.getenv("COMPOSER_VENV", "/opt/venv/composer")
    pythonpath = os.getenv("PYTHONPATH", "")

    errors = []

    print(f"✅ COMPOSER_VENV={composer_venv}")

    if "/app" not in pythonpath:
        errors.append("❌ PYTHONPATH: Should include /app for cross-module imports")
    else:
        print(f"✅ PYTHONPATH={pythonpath}")

    return errors


def main():
    """Run all environment variable tests."""
    print("🚀 Kubernetes Deployment Environment Variable Validation\n")

    all_errors = []

    # Test composer environment variables
    all_errors.extend(test_composer_env_vars())

    # Test database connectivity
    all_errors.extend(test_database_connectivity())

    # Test virtual environment
    all_errors.extend(test_virtual_environment())

    # Summary
    print(f"\n📊 Validation Summary:")
    if all_errors:
        print(f"❌ {len(all_errors)} errors found:")
        for error in all_errors:
            print(f"   {error}")
        sys.exit(1)
    else:
        print("✅ All environment variables configured correctly!")
        print("\n🎉 Composer service ready for Kubernetes deployment!")


if __name__ == "__main__":
    main()
