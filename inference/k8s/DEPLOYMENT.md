# Inference Service Environment Configuration and Kubernetes Deployment

This document explains the environment configuration and Kubernetes deployment strategy for the Inference Service.

## Overview

The Inference Service is designed to run in Kubernetes with a fully configurable environment using environment variables. This approach allows for easy configuration changes without rebuilding the container image and supports different deployment environments.

## Key Components

1. **Environment Configuration** (`env.yaml`):
   - ConfigMap for non-sensitive configuration
   - Secret for sensitive configuration

2. **Kubernetes Deployment** (`inference-service.yaml`):
   - Deployment specification
   - Service definition
   - Ingress for external access

3. **Docker Image** (`Dockerfile`):
   - Base image with CUDA support
   - Environment variables with sensible defaults
   - Multiple Python virtual environments for different components

## Configuration Strategy

### Environment Variables

All configuration is exposed through environment variables, organized into logical sections:

- **Server Configuration**: Basic server settings
- **Authentication**: JWT/JWKS auth settings
- **Database**: PostgreSQL connection settings
- **Redis**: Redis connection and configuration
- **Summarization**: Conversation summarization settings
- **Image Generation**: Image generation and storage settings
- **RabbitMQ**: Message queue settings
- **Inference Services**: gRPC and API settings
- **Internal Security**: Security settings for internal APIs

### Configuration Layering

Configuration is applied in layers, with each layer overriding the previous:

1. **Dockerfile Defaults**: Basic defaults set in the Dockerfile
2. **Environment Variables**: Applied from Kubernetes ConfigMap/Secret
3. **Command Line Arguments**: For specific overrides (e.g., server ports)

## Deployment Process

1. **Prepare Configuration**:
   - Review and modify `env.yaml` to set configuration values
   - Ensure secrets are properly secured

2. **Deploy to Kubernetes**:
   - Run the deployment script: `./deploy-inference.sh`
   - The script handles ConfigMap, Secret, and Deployment creation

3. **Verify Deployment**:
   - Check logs and service status
   - Access the service through the ingress

## Additional Information

### Persistent Storage

The deployment uses persistent volumes for:

- Generated images (`generated-images` PVC)
- Application code (`code-base` PVC)

### Security Considerations

- Sensitive configuration is stored in Kubernetes Secrets
- JWT authentication for API endpoints
- Internal API key for service-to-service communication
- IP allowlisting for internal endpoints

### Monitoring and Logging

- Container logs are accessible via `kubectl logs`
- Service status is tracked within the container
- Health checks ensure services are running properly

## Deployment Architecture

```ascii
                           ┌───────────────┐
                           │   Ingress     │
                           │   Controller  │
                           └───────┬───────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────┐
│                  Inference Service                    │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   FastAPI   │  │    gRPC     │  │   Ollama    │   │
│  │   Server    │  │   Server    │  │   Server    │   │
│  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘   │
│        │                │                │           │
└────────┼────────────────┼────────────────┼───────────┘
         │                │                │
         ▼                ▼                ▼
┌────────────────┐  ┌────────────┐  ┌────────────────┐
│   PostgreSQL   │  │   Redis    │  │    RabbitMQ    │
└────────────────┘  └────────────┘  └────────────────┘
```

## Testing and Validation

After deployment, validate that:

1. The REST API is accessible via the ingress
2. The gRPC server is available for internal services
3. Configuration is properly applied from environment variables
4. Authentication and authorization are working as expected

## Troubleshooting

Common issues and solutions:

- **Service won't start**: Check logs with `kubectl logs`
- **Authentication failures**: Verify JWT settings in env.yaml
- **Database connection issues**: Check DB credentials and connectivity
- **Resource limitations**: Adjust resource requests/limits in the deployment YAML

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [gRPC Documentation](https://grpc.io/docs/)
- [Kubernetes ConfigMaps and Secrets](https://kubernetes.io/docs/concepts/configuration/)
