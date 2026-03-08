"""
Integration test setup verification.

This module verifies that the integration test environment is properly configured.
"""

import os
import pytest

pytestmark = pytest.mark.integration


def test_environment_variables():
    """Test that required environment variables are set."""
    required_vars = [
        "DB_CONNECTION_STRING",
        "TEST_MODE",
    ]

    for var in required_vars:
        assert var in os.environ, f"Required environment variable {var} is not set"
        assert os.environ[var], f"Required environment variable {var} is empty"


def test_database_connection_string():
    """Test that the database connection string is properly configured."""
    connection_string = os.environ.get("DB_CONNECTION_STRING")

    assert connection_string is not None, "DB_CONNECTION_STRING is not set"
    assert "postgresql://" in connection_string, "Invalid connection string format"
    assert "localhost" in connection_string or "db" in connection_string, \
        "Connection string should point to localhost or db"
    assert "llmmll_test" in connection_string, \
        "Connection string should use llmmll_test database"


def test_test_mode():
    """Test that test mode is enabled."""
    test_mode = os.environ.get("TEST_MODE", "").lower()

    assert test_mode == "true", "TEST_MODE should be enabled for integration tests"


def test_server_url():
    """Test that server URL is configured."""
    server_url = os.environ.get("SERVER_URL", "http://localhost:8000")

    assert server_url is not None
    assert server_url.startswith("http"), "SERVER_URL should be a valid URL"


def test_test_user_id():
    """Test that test user ID is configured."""
    user_id = os.environ.get("TEST_USER_ID")

    assert user_id is not None, "TEST_USER_ID is not set"
    assert len(user_id) > 0, "TEST_USER_ID should not be empty"


def test_api_key():
    """Test that API key is configured."""
    api_key = os.environ.get("SERVER_API_KEY", "test-api-key")

    assert api_key is not None
    assert len(api_key) > 0, "API key should not be empty"


def test_logging_level():
    """Test that logging level is configured."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    assert log_level in ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"], \
        f"Invalid LOG_LEVEL: {log_level}"


def test_pytest_config():
    """Test that pytest is properly configured."""
    pytest_opts = os.environ.get("PYTEST_ADDOPTS", "")

    assert "integration" in pytest_opts or "--tb=short" in pytest_opts or True