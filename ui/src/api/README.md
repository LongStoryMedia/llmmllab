# API Layer with Versioning

This directory contains the API layer for the UI, which has been updated to include versioning support for better compatibility management.

## API Versioning

All API requests now include a version prefix (e.g., `/v1/models`, `/v1/config`) to ensure backward compatibility as APIs evolve.

### Configuration

The API version is configured in `src/config/index.ts`:

```typescript
export default {
  server: {
    baseUrl: import.meta.env.VITE_BASE_URL,
    apiVersion: 'v1'  // Current API version
  },
  // ...other config
}
```

### Usage Examples

#### Basic API Requests

API requests automatically include the configured API version:

```typescript
// This will call `/v1/models`
const models = await getModels(accessToken);
```

#### Targeting Specific API Versions

If you need to target a specific API version for compatibility:

```typescript
import { req, getHeaders } from './api';

// Call a specific API version endpoint
const result = await req({
  method: 'GET',
  path: 'models',
  apiVersion: 'v2',  // This will call `/v2/models`
  headers: getHeaders(accessToken)
});
```

#### Version Utilities

The `version.ts` module provides utilities for working with versioned APIs:

```typescript
import { getVersionedPath, isVersionCompatible } from './api';

// Get a versioned path (e.g., "v1/models")
const path = getVersionedPath('models');

// Check if the current API version supports a feature
if (isVersionCompatible('v2')) {
  // Use features that require API v2 or higher
}
```

#### WebSocket Connections

WebSocket connections also support versioning:

```typescript
const ws = new ChatWebSocketClient(
  SocketConnectionTypeValues.CHAT, 
  handleMessage,
  '/conversation/123',
  'v2'  // Optional custom API version
);
```

## Migration

When the backend API evolves and introduces breaking changes, follow these steps:

1. Update the API version in the backend (e.g., from `/v1/...` to `/v2/...`)
2. Maintain backward compatibility in the backend for older clients
3. Update the UI to use the new API version when ready:

   ```typescript
   // Update global default in config
   config.server.apiVersion = 'v2';
   ```
   
4. For gradual migration, you can target specific endpoints with specific versions
