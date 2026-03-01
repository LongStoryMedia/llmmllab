"""Anthropic API routers - Auto-generated from claude.generated.yaml

This package contains FastAPI routers for all Anthropic API endpoints.
Each root path (messages, models, files, etc.) has its own router file.

Usage in app.py:
    from server.routers.anthropic import ROUTERS

    for router in ROUTERS:
        app.include_router(router, prefix="/v1")
"""

from .import_routers import ROUTERS

__all__ = ["ROUTERS"]
