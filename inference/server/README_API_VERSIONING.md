# API Versioning in Inference Server

This document explains how API versioning is implemented in the inference server.

## Overview

The API versioning system allows both non-versioned (legacy) endpoints and versioned endpoints to coexist, ensuring backward compatibility while enabling future API evolution.

## Current Implementation

- The current API version is defined as a constant `CURRENT_API_VERSION` in `version.py` (currently set to `"v1"`)
- All routes are available under both non-versioned paths (e.g., `/chat/conversations`) and versioned paths (e.g., `/v1/chat/conversations`)
- The root endpoint (`/`) provides information about available endpoints, including their versioned paths

## Utility Functions

The `version.py` module provides several utility functions to help with API versioning:

1. `get_versioned_prefix(base_prefix, version)`: Generates a versioned path prefix
2. `create_versioned_router(base_router, version)`: Creates a router with versioned routes
3. `include_versioned_routers(app, routers, versions)`: Includes routers with multiple versions

## Usage

### Including Versioned Routes

In `app.py`, all routers are included with both non-versioned and versioned paths:

```python
# Include routers with both non-versioned and versioned paths
app.include_router(chat_router)  # Non-versioned (legacy)
app.include_router(chat_router, prefix=f"/{CURRENT_API_VERSION}")  # Versioned
```

### Creating New Endpoints

When creating new endpoints:

1. Add them to the appropriate router
2. They will automatically be available under both non-versioned and versioned paths
3. Consider future compatibility when designing new endpoints

### Updating Existing Endpoints

When updating existing endpoints:

1. For minor changes, update the endpoint directly
2. For breaking changes, consider implementing a new version in a separate router

## Future Versioning

To add a new API version in the future:

1. Update the `CURRENT_API_VERSION` constant in `version.py`
2. Use the `include_versioned_routers` function to include routers under multiple versions
3. Implement version-specific routers for endpoints with breaking changes

## Client Integration

The UI API layer includes corresponding versioning logic to interact with the versioned endpoints. See the UI API documentation for details.
