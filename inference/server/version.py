"""
API versioning utilities for the inference server.

This module provides utilities to work with versioned API endpoints,
supporting backward compatibility as the API evolves.
"""

import os
from typing import Dict, List, Optional, Union
from fastapi import APIRouter, FastAPI

# Import API version from config
try:
    from server.config import API_VERSION
except ImportError:
    # Fallback if server.config can't be imported (prevents circular import)
    API_VERSION = os.environ.get("API_VERSION", "v1")

# Current API version
CURRENT_API_VERSION = API_VERSION


def get_versioned_prefix(router_prefix: str, version: str = CURRENT_API_VERSION) -> str:
    """
    Generate a versioned API prefix for routers.

    Args:
        router_prefix: The router-specific prefix (e.g., "/chat")
        version: The API version (default: CURRENT_API_VERSION)

    Returns:
        str: Versioned prefix (e.g., "/v1/chat")
    """
    # Clean up any existing leading/trailing slashes for consistent formatting
    clean_prefix = router_prefix.strip("/")
    return f"/{version}/{clean_prefix}"


def create_versioned_router(prefix: str, **kwargs) -> APIRouter:
    """
    Create an APIRouter with a versioned prefix.

    Args:
        prefix: The router-specific prefix (e.g., "/chat")
        **kwargs: Additional arguments to pass to APIRouter

    Returns:
        APIRouter: A router with a versioned prefix
    """
    versioned_prefix = get_versioned_prefix(prefix)
    return APIRouter(prefix=versioned_prefix, **kwargs)


def include_versioned_routers(
    app: FastAPI, routers: dict, supported_versions: Optional[List[str]] = None
) -> None:
    """
    Include versioned routers in a FastAPI application.

    Args:
        app: The FastAPI application
        routers: A dictionary of {prefix: router_module} pairs
        supported_versions: List of supported API versions (default: [CURRENT_API_VERSION])
    """
    if supported_versions is None:
        supported_versions = [CURRENT_API_VERSION]

    for version in supported_versions:
        for prefix, router_module in routers.items():
            versioned_prefix = get_versioned_prefix(prefix, version)
            # Create a copy of the router with the versioned prefix
            versioned_router = APIRouter(
                prefix=versioned_prefix, tags=router_module.router.tags
            )

            # Copy all routes from the original router to the versioned one
            for route in router_module.router.routes:
                versioned_router.routes.append(route)

            app.include_router(versioned_router)
