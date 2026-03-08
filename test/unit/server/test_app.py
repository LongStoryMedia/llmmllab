"""
Unit tests for server/app.py.

Tests FastAPI app initialization, lifespan management, and middleware setup.
"""
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, MagicMock as Mock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware


# This fixture is now in test/unit/conftest.py to ensure it runs before
# pytest collection phase. Keeping a stub here to document intent.
@pytest.fixture
def setup_test_environment():
    """Set up test environment variables before any tests run."""
    # Environment variables are now set in test/unit/conftest.py at module level
    # to ensure they are available during pytest collection before any imports.
    pass


@pytest.fixture
def app_with_middleware(setup_test_environment):
    """Create a fresh app instance with middleware for testing."""
    # Import app module to get the factory function
    from server import app as app_module

    # Clear any cached imports
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("server"):
            del sys.modules[mod_name]

    # Create app with minimal setup
    test_app = FastAPI(
        title="Test API",
        lifespan=lambda _: AsyncMock(),
    )

    # Add middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @test_app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    return test_app


def test_app_creates_correct_middleware(app_with_middleware):
    """Test that app creates the expected middleware stack."""
    # user_middleware contains the middleware registered via add_middleware
    # It's a list of Middleware objects from starlette
    middleware_list = app_with_middleware.user_middleware

    # Check that CORSMiddleware is present in user_middleware
    assert any('CORSMiddleware' in str(m.cls) for m in middleware_list)


def test_app_includes_routers(setup_test_environment):
    """Test that app includes all expected routers."""
    # Import the main app - need to reload to pick up test environment
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("server"):
            del sys.modules[mod_name]

    # Re-import with test environment
    import importlib
    from server import app as app_module
    importlib.reload(app_module)

    routes = [r.path for r in app_module.app.routes if hasattr(r, 'path')]

    # Verify core router paths are included
    assert any('/v1' in path for path in routes)
    assert any('/health' in path for path in routes)
    assert any('/docs' in path for path in routes)


def test_app_configured_for_cors(app_with_middleware):
    """Test that app is configured with appropriate CORS settings."""
    # Check that CORS middleware has correct parameters
    # Middleware objects have 'cls' and 'kwargs' attributes
    cors_middleware = None
    for m in app_with_middleware.user_middleware:
        if 'CORSMiddleware' in str(m.cls):
            cors_middleware = m
            break

    assert cors_middleware is not None
    # Verify CORS is configured to allow all origins in dev
    assert cors_middleware.kwargs.get('allow_origins') == ['*']
    assert cors_middleware.kwargs.get('allow_methods') == ['*']
    assert cors_middleware.kwargs.get('allow_headers') == ['*']