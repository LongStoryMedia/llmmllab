"""Common API routers - Overlapping endpoints between OpenAI and Anthropic

This package contains FastAPI routers for endpoints that are shared between
the OpenAI and Anthropic API specifications.

Usage in app.py:
    from server.routers.common import ROUTERS

    for router in ROUTERS:
        app.include_router(router, prefix="/v1")
"""

from .import_routers import ROUTERS

__all__ = ["ROUTERS"]
