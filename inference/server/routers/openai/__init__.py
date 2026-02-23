"""OpenAI API routers - Auto-generated from openai.documented.yml

This package contains FastAPI routers for all OpenAI API endpoints.
Each root path (chat, audio, models, etc.) has its own router file.

Usage in app.py:
    from server.routers.openai import ROUTERS
    
    for router in ROUTERS:
        app.include_router(router, prefix="/v1")
"""

from .import_routers import ROUTERS

__all__ = ["ROUTERS"]
