# Kubernetes Environment Variables Reference

## Composer Service Configuration

### System Configuration (Infrastructure)

These environment variables control the composer service infrastructure and should be managed by operators:

```yaml
# Service binding and behavior
- name: COMPOSER_HOST
  value: "0.0.0.0"                    # Service host address
- name: COMPOSER_PORT  
  value: "8001"                       # Service port (matches container port)
- name: COMPOSER_DEBUG
  value: "false"                      # Enable debug mode (true/false)
- name: COMPOSER_RELOAD
  value: "false"                      # Enable auto-reload in development (true/false)
- name: COMPOSER_LOG_LEVEL
  value: "INFO"                       # Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)

# CORS and security
- name: COMPOSER_ENABLE_CORS
  value: "true"                       # Enable CORS middleware (true/false)

# Rate limiting
- name: COMPOSER_RATE_LIMIT_ENABLED
  value: "true"                       # Enable rate limiting (true/false)
- name: COMPOSER_RATE_LIMIT_RPM
  value: "60"                         # Requests per minute limit (1-1000)

# Health monitoring
- name: COMPOSER_HEALTH_CHECK_ENABLED
  value: "true"                       # Enable health checks (true/false)
- name: COMPOSER_HEALTH_CHECK_INTERVAL
  value: "30"                         # Health check interval in seconds (5-300)

# Virtual environment
- name: COMPOSER_VENV
  value: "/opt/venv/composer"         # Path to composer virtual environment
```

### User Configuration Defaults (UI Customizable)

These environment variables provide system-wide defaults for user-configurable settings. Users can override these through the UI:

#### Workflow Settings

```yaml
# Caching configuration
- name: COMPOSER_ENABLE_CACHE
  value: "true"                       # Enable workflow caching (true/false)
- name: COMPOSER_CACHE_TTL
  value: "3600"                       # Cache TTL in seconds (60-86400)

# Execution configuration  
- name: COMPOSER_MAX_PARALLEL_TOOLS
  value: "5"                          # Max parallel tool executions (1-20)
- name: COMPOSER_ENABLE_MULTI_AGENT
  value: "false"                      # Enable multi-agent workflows (true/false)
- name: COMPOSER_DEFAULT_TIMEOUT
  value: "60.0"                       # Default execution timeout in seconds (5.0-300.0)

# Context management
- name: COMPOSER_MAX_CONTEXT_LENGTH
  value: "128000"                     # Maximum context length (1000-1000000)
- name: COMPOSER_CONTEXT_TRIM_THRESHOLD
  value: "0.8"                        # Context trimming threshold (0.1-1.0)

# Streaming configuration
- name: COMPOSER_ENABLE_STREAMING
  value: "true"                       # Enable streaming responses (true/false)
- name: COMPOSER_STREAM_BUFFER_SIZE
  value: "1024"                       # Streaming buffer size in bytes (256-8192)
```

#### Tool Settings

```yaml
# Tool selection
- name: COMPOSER_TOOL_SIMILARITY_THRESHOLD
  value: "0.9"                        # Similarity threshold for tool matching (0.1-1.0)
- name: COMPOSER_TOOL_MODIFICATION_THRESHOLD
  value: "0.6"                        # Threshold for tool modification (0.1-1.0)

# Tool generation
- name: COMPOSER_ENABLE_TOOL_GENERATION
  value: "true"                       # Enable dynamic tool generation (true/false)

# Tool execution
- name: COMPOSER_MAX_TOOL_RETRIES
  value: "3"                          # Maximum retries for failed executions (0-10)
- name: COMPOSER_TOOL_TIMEOUT
  value: "30.0"                       # Tool execution timeout in seconds (1.0-120.0)

# Tool caching
- name: COMPOSER_ENABLE_TOOL_CACHING
  value: "true"                       # Enable tool result caching (true/false)
- name: COMPOSER_TOOL_CACHE_TTL
  value: "1800"                       # Tool cache TTL in seconds (60-7200)

# Semantic search
- name: COMPOSER_ENABLE_SEMANTIC_SEARCH
  value: "true"                       # Enable semantic tool selection (true/false)
- name: COMPOSER_SEARCH_TOP_K
  value: "10"                         # Number of top tools to consider (1-50)
```

## Configuration Hierarchy

### 1. Environment Variables (Kubernetes)
- Deployed via `k8s/deployment.yaml`
- Provides system-wide defaults
- Managed by operators

### 2. User Configuration (Database)
- Stored in UserConfig model
- Customizable per user via UI  
- Overrides environment defaults

### 3. Request Overrides (API)
- Per-request configuration overrides
- Specified in ComposerRequest.config_overrides
- Highest priority for individual requests

## Validation

Use the validation script to ensure all environment variables are properly configured:

```bash
# In Kubernetes pod
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python debug/test_k8s_env_vars.py

# Local testing with environment file
source .env && python inference/debug/test_k8s_env_vars.py
```

## Deployment Process

1. **Update deployment.yaml** with environment variables
2. **Apply Kubernetes manifests**:
   ```bash
   cd inference/k8s
   ./apply.sh
   ```
3. **Validate deployment**:
   ```bash
   kubectl get pods -n ollama
   kubectl logs -n ollama -l app=ollama --tail=100
   ```
4. **Test composer service**:
   ```bash
   POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
   kubectl exec -it -n ollama $POD_NAME -- /app/v.sh composer python debug/test_k8s_env_vars.py
   ```

## Environment Variable Best Practices

### System Variables
- Use string values for all environment variables in Kubernetes
- Boolean values should be "true" or "false" (lowercase)
- Numeric values should be strings that parse correctly
- Always provide sensible production defaults

### Security Variables
- Store sensitive values in Kubernetes secrets
- Reference secrets using `valueFrom.secretKeyRef`
- Never include credentials in deployment.yaml directly

### Resource Limits
- Set appropriate min/max values for numeric settings
- Consider container resource limits when setting timeouts
- Balance performance vs. resource consumption

### User Defaults
- Provide conservative defaults for user-configurable settings
- Allow users to optimize based on their specific needs
- Document the impact of each setting on performance/behavior

## Troubleshooting

### Common Issues

1. **Service won't start**: Check COMPOSER_HOST and COMPOSER_PORT
2. **Config validation errors**: Run test_k8s_env_vars.py
3. **Performance issues**: Adjust timeout and parallel execution limits
4. **Memory issues**: Reduce context length and buffer sizes

### Monitoring

- Check service logs: `kubectl logs -n ollama -l app=ollama -c ollama --tail=100`
- Health check endpoint: `http://composer:8001/health`
- Config endpoint: `http://composer:8001/config`

### Environment Variable Debugging

```bash
# Check all composer environment variables in pod
kubectl exec -it -n ollama $POD_NAME -- env | grep COMPOSER

# Test configuration loading
kubectl exec -it -n ollama $POD_NAME -- /app/v.sh composer python -c "from composer.config import config; print(config.service.host, config.service.port)"
```