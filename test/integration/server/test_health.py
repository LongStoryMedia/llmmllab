"""
Server health check integration tests.
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
    assert isinstance(data, dict)
    assert "API" in data or "title" in data