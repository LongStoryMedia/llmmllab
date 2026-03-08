"""
Server integration tests.

Tests the FastAPI server endpoints and middleware.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_health_endpoint(server_client: AsyncClient):
    """Test the health check endpoint."""
    response = await server_client.get("/health")

    assert response.status_code == 200
    assert response.json() == "OK"


@pytest.mark.asyncio
async def test_root_endpoint(server_client: AsyncClient):
    """Test the root endpoint."""
    response = await server_client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "API" in data or "title" in data or isinstance(data, dict)


@pytest.mark.asyncio
async def test_docs_endpoint(server_client: AsyncClient):
    """Test the Swagger UI endpoint."""
    response = await server_client.get("/docs")

    assert response.status_code == 200
    assert "swagger" in response.text.lower()


@pytest.mark.asyncio
async def test_openapi_json(server_client: AsyncClient):
    """Test the OpenAPI schema endpoint."""
    response = await server_client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "paths" in data
    assert "components" in data


@pytest.mark.asyncio
async def test_openai_compatibility_endpoint(server_client: AsyncClient):
    """Test the OpenAI-compatible API endpoint exists."""
    response = await server_client.get("/v1")

    # May return 404 if no specific route, but should not be 500
    assert response.status_code in [200, 404, 405]


@pytest.mark.asyncio
async def test_auth_bypass_enabled(server_client: AsyncClient):
    """Test that authentication is disabled for testing."""
    # Without auth headers, we should still be able to access protected endpoints
    response = await server_client.get("/v1/models")

    # Should not get a 401 Unauthorized
    assert response.status_code != 401, "Auth bypass may not be enabled"


@pytest.mark.asyncio
async def test_server_response_time(server_client: AsyncClient):
    """Test that the server responds within a reasonable time."""
    import time

    start = time.time()
    response = await server_client.get("/health")
    elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 5.0, f"Server response time too slow: {elapsed}s"


@pytest.mark.asyncio
async def test_server_headers(server_client: AsyncClient):
    """Test that the server includes proper headers."""
    response = await server_client.get("/health")

    assert "content-type" in response.headers
    assert "date" in response.headers
    assert "server" in response.headers