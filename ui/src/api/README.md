# API Layer

This directory contains the API layer for the UI.

## Structure

The API layer is organized by resource type:

- `base.ts` - Core request functions (`req`, `gen`) and authentication utilities
- `types.ts` - TypeScript types for API requests and responses
- `message.ts` - Message and chat operations
- `conversation.ts` - Conversation management
- `config.ts` - User configuration
- `model.ts` - Model and model profile management
- `image.ts` - Image generation and management
- `usrmgr.ts` - User management and authentication
- `resources.ts` - System resource monitoring

## Usage Examples

### Basic API Requests

API requests use the centralized `req` function:

```typescript
import { req, getHeaders } from './api';

// Make a GET request
const models = await req({
  method: 'GET',
  path: 'models',
  headers: getHeaders(accessToken)
});

// Make a POST request with body
const result = await req({
  method: 'POST',
  path: 'chat/completions',
  headers: getHeaders(accessToken),
  body: JSON.stringify(data)
});
```

### Streaming Responses

For streaming endpoints, use the `gen` function:

```typescript
import { gen } from './api';

for await (const chunk of gen({
  method: 'POST',
  path: 'chat/completions',
  headers: getHeaders(accessToken),
  body: JSON.stringify(data)
})) {
  // Process streaming chunks
  console.log(chunk);
}
```

### WebSocket Connections

WebSocket connections are managed through the `ChatWebSocketClient`:

```typescript
const ws = new ChatWebSocketClient(
  SocketConnectionTypeValues.CHAT, 
  handleMessage,
  '/conversation/123'
);
```
