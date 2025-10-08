# Configuration Architecture

## Overview

The LLM ML Lab platform uses a hierarchical configuration system that separates system-level settings from user-configurable preferences. This ensures proper separation of concerns while maintaining flexibility for user customization.

## Configuration Hierarchy

### 1. System Configuration (Non-User Configurable)

These settings are managed at the infrastructure level and cannot be modified by end users:

- **Service Settings**: Host, port, debug mode, CORS configuration
- **Database Configuration**: Connection strings, connection pools
- **Infrastructure Settings**: Rate limiting, health checks, logging levels
- **Circuit Breaker Settings**: Failure thresholds, recovery timeouts

**Configuration Files:**
- `schemas/server_config.yaml` - Server configuration  
- `schemas/database_config.yaml` - Database configuration
- `inference/composer/config.py` - Environment variable loading

### 2. User Configuration (User Customizable)

These settings can be customized per user and are stored in the UserConfig model:

#### Workflow Settings
- **Caching**: Enable/disable workflow caching, TTL settings
- **Execution**: Parallel tool limits, timeouts, context length
- **Streaming**: Enable/disable response streaming, buffer sizes
- **Multi-Agent**: Enable/disable multi-agent capabilities

#### Tool Settings  
- **Selection**: Similarity thresholds for tool matching
- **Generation**: Enable/disable dynamic tool creation
- **Execution**: Retry limits, timeouts, caching preferences
- **Search**: Semantic search preferences, top-k limits

#### Memory & Context
- **Memory Configuration**: Retrieval settings, storage preferences
- **Circuit Breaker**: User-specific failure thresholds
- **Model Profiles**: Primary/formatting model preferences
- **Preferences**: UI preferences, notification settings

**Configuration Files:**
- `schemas/workflow_config.yaml` - Workflow execution settings
- `schemas/tool_config.yaml` - Tool management settings  
- `schemas/user_config.yaml` - Complete user configuration schema

## Configuration Loading

### System Configuration
```python
from composer.config import config

# Access service-level settings
host = config.service.host
port = config.service.port
debug = config.service.debug

# Access system database settings
db_url = config.database_url
redis_url = config.redis_url
```

### User Configuration
```python
# Get user-specific configurations with fallbacks
workflow_config = config.get_workflow_config(user_config.workflow)
tool_config = config.get_tool_config(user_config.tool)

# Access user preferences
enable_streaming = workflow_config.enable_streaming
tool_timeout = tool_config.tool_timeout
```

## Environment Variables

### System Settings
```bash
# Composer service configuration
COMPOSER_HOST=0.0.0.0
COMPOSER_PORT=8001
COMPOSER_DEBUG=false
COMPOSER_LOG_LEVEL=INFO

# Infrastructure  
COMPOSER_DATABASE_URL=postgresql://...
COMPOSER_REDIS_URL=redis://...
```

### Default User Settings (Override Defaults)
```bash
# Workflow defaults
COMPOSER_ENABLE_CACHE=true
COMPOSER_CACHE_TTL=3600
COMPOSER_MAX_PARALLEL_TOOLS=5
COMPOSER_ENABLE_STREAMING=true

# Tool defaults
COMPOSER_TOOL_SIMILARITY_THRESHOLD=0.9
COMPOSER_ENABLE_TOOL_GENERATION=true
COMPOSER_TOOL_TIMEOUT=30.0
```

## Schema-Driven Type Safety

All configuration is defined via YAML schemas and automatically generates:

1. **Python Models** (`inference/models/`) - Pydantic models with validation
2. **TypeScript Types** (`ui/src/types/`) - Frontend type definitions  

```bash
# Regenerate after schema changes
./regenerate_models.sh
```

## API Integration

### Composer Service Endpoints

- **`GET /config`** - Returns current system configuration
- **`POST /compose`** - Accepts ConversationCtx with UserConfig
- **`POST /execute`** - Executes workflows with user preferences

### User Configuration in Requests
```typescript
interface ComposerRequest {
  conversation_ctx: ConversationCtx;
  workflow_type: string;
  config_overrides?: Record<string, any>;
}

interface ConversationCtx {
  messages: Message[];
  user_config?: UserConfig;  // User preferences applied here
  // ... other fields
}
```

## UI Integration

User-configurable settings are automatically surfaced in the UI based on schema definitions:

### Workflow Settings Panel
- Caching preferences
- Execution timeouts  
- Streaming configuration
- Multi-agent toggles

### Tool Management Panel
- Tool selection thresholds
- Generation preferences
- Execution settings
- Search configuration

### Model & Memory Panel
- Model profile selection
- Memory retrieval settings
- Circuit breaker thresholds

## Migration Guide

### From Legacy Config
```python
# OLD: Direct config access
if config.enable_streaming:
    # ...

# NEW: User-aware config access  
workflow_config = config.get_workflow_config(user_config.workflow)
if workflow_config.enable_streaming:
    # ...
```

### Schema Updates
1. Update YAML schema file
2. Run `./regenerate_models.sh`
3. Update code references
4. Update documentation
5. Test UI integration

## Best Practices

1. **System Settings**: Use for infrastructure, security, performance limits
2. **User Settings**: Use for behavior, preferences, feature toggles
3. **Environment Variables**: Use for deployment-specific overrides
4. **Schema First**: Always define schemas before implementation
5. **Type Safety**: Leverage generated models for validation
6. **Fallbacks**: Always provide sensible defaults
7. **Documentation**: Update docs when adding new configuration options

## Validation & Constraints

- Pydantic models provide runtime validation
- YAML schemas define min/max values and allowed types
- Environment variable parsing includes type coercion
- UI forms automatically respect schema constraints

## Example: Adding New User Setting

1. **Update Schema** (`schemas/workflow_config.yaml`):
```yaml
  enable_advanced_reasoning:
    type: boolean
    description: Enable advanced reasoning capabilities
    default: false
```

2. **Regenerate Models**:
```bash
./regenerate_models.sh
```

3. **Use in Code**:
```python
workflow_config = config.get_workflow_config(user_config.workflow)
if workflow_config.enable_advanced_reasoning:
    # Advanced reasoning logic
```

4. **Add UI Control** (automatic based on schema)

This architecture ensures scalable, type-safe configuration management while maintaining clear separation between system administration and user preferences.