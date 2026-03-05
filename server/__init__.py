"""
Server Application.

The server is the DB gateway and API endpoint provider for the llmmllab system.
It handles:
- Database access (PostgreSQL, Redis, in-memory storage)
- API endpoints (OpenAI-compatible, Anthropic-compatible)
- Communication with runner and composer services
- User authentication and authorization
"""

from typing import Optional

from .db.storage import Storage
from .config import settings

# Global storage instance
_storage: Optional[Storage] = None


def get_storage() -> Storage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        raise RuntimeError("Storage not initialized. Call initialize_storage() first.")
    return _storage


async def initialize_storage() -> Storage:
    """Initialize the global storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
        await _storage.initialize()
    return _storage


async def shutdown_storage() -> None:
    """Shutdown the global storage instance."""
    global _storage
    if _storage is not None:
        await _storage.close()
        _storage = None


# Composer interface - can be ported to gRPC later
class ComposerClient:
    """
    Client interface for communicating with the composer service.

    Currently uses HTTP, but can be ported to gRPC for better performance.
    """

    def __init__(self, base_url: str = "http://composer:8000"):
        self.base_url = base_url

    async def compose_workflow(
        self,
        user_id: str,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """Compose a workflow for the given user."""
        # TODO: Implement HTTP call to composer service
        pass

    async def execute_workflow(self, initial_state: dict, workflow: dict):
        """Execute a compiled workflow."""
        # TODO: Implement HTTP call to composer service
        pass

    async def health_check(self) -> bool:
        """Check if the composer service is healthy."""
        # TODO: Implement health check endpoint
        return True


# Runner interface - can be ported to gRPC later
class RunnerClient:
    """
    Client interface for communicating with the runner service.

    Currently uses HTTP, but can be ported to gRPC for better performance.
    """

    def __init__(self, base_url: str = "http://runner:8000"):
        self.base_url = base_url

    async def execute_pipeline(
        self,
        pipeline_name: str,
        model_name: str,
        input_data: dict,
        **kwargs,
    ):
        """Execute a pipeline using the runner service."""
        # TODO: Implement HTTP call to runner service
        pass

    async def get_model_info(self, model_id: str):
        """Get information about a specific model."""
        # TODO: Implement HTTP call to runner service
        pass

    async def health_check(self) -> bool:
        """Check if the runner service is healthy."""
        # TODO: Implement health check endpoint
        return True