# Inference Service Kubernetes Deployment

This document explains how to deploy the Inference Service to Kubernetes using environment variables for configuration.

## Environment Configuration

The service is configured using environment variables that are defined in the `env.yaml` file. This file contains a ConfigMap for non-sensitive data and a Secret for sensitive data.

### Environment Variables

#### Server Configuration

- `LOG_LEVEL`: The logging level for the application (debug, info, warning, error)

#### Authentication Configuration

- `AUTH_ISSUER`: The issuer URL for JWT authentication
- `AUTH_AUDIENCE`: The audience value for JWT validation
- `AUTH_CLIENT_ID`: The client ID for authentication
- `AUTH_CLIENT_SECRET`: (Secret) The client secret for authentication
- `AUTH_JWKS_URI`: The URL for the JWKS endpoint

#### Database Configuration

- `DB_HOST`: The host address for the PostgreSQL database
- `DB_PORT`: The port number for the PostgreSQL database
- `DB_USER`: The username for the PostgreSQL database
- `DB_PASSWORD`: (Secret) The password for the PostgreSQL database
- `DB_NAME`: The name of the PostgreSQL database
- `DB_SSLMODE`: The SSL mode for PostgreSQL connections

#### Redis Configuration

- `REDIS_ENABLED`: Whether Redis is enabled
- `REDIS_HOST`: The host address for Redis
- `REDIS_PORT`: The port number for Redis
- `REDIS_PASSWORD`: (Secret) The password for Redis
- `REDIS_DB`: The Redis database number
- `REDIS_CONVERSATION_TTL`: Time to live for conversation data in Redis (minutes)
- `REDIS_MESSAGE_TTL`: Time to live for message data in Redis (minutes)
- `REDIS_SUMMARY_TTL`: Time to live for summary data in Redis (minutes)
- `REDIS_POOL_SIZE`: The size of the Redis connection pool
- `REDIS_MIN_IDLE_CONNECTIONS`: Minimum idle connections in the Redis pool
- `REDIS_CONNECT_TIMEOUT`: Timeout for Redis connection attempts (seconds)

#### Summarization Configuration

- `MESSAGES_BEFORE_SUMMARY`: Number of messages before generating a summary
- `SUMMARIES_BEFORE_CONSOLIDATION`: Number of summaries before consolidation
- `SUMMARY_MODEL`: The model to use for summarization
- `SUMMARY_SYSTEM_PROMPT`: The system prompt for summarization
- `MAX_SUMMARY_LEVELS`: Maximum depth of summary hierarchy
- `SUMMARY_WEIGHT_COEFFICIENT`: Weight coefficient for summary relevance
- `MASTER_SUMMARY_PROMPT`: The system prompt for master summaries

#### Image Generation Configuration

- `IMAGE_GENERATION_ENABLED`: Whether image generation is enabled
- `IMAGE_DIR`: Directory to store generated images
- `MAX_IMAGE_SIZE`: Maximum size of generated images (pixels)
- `IMAGE_RETENTION_HOURS`: How long to retain generated images (hours)

#### RabbitMQ Configuration

- `RABBITMQ_ENABLED`: Whether RabbitMQ is enabled
- `RABBITMQ_HOST`: The host address for RabbitMQ
- `RABBITMQ_PORT`: The port number for RabbitMQ
- `RABBITMQ_USER`: The username for RabbitMQ
- `RABBITMQ_PASSWORD`: (Secret) The password for RabbitMQ
- `RABBITMQ_VHOST`: The virtual host for RabbitMQ

#### Inference Services Configuration

- `INFERENCE_SERVICES_OLLAMA_BASE_URL`: Base URL for Ollama API
- `INFERENCE_SERVICES_SD_BASE_URL`: Base URL for Stable Diffusion API
- `INFERENCE_SERVICES_HOST`: Host address for the gRPC server
- `INFERENCE_SERVICES_PORT`: Port number for the gRPC server

#### Internal Security Configuration

- `INTERNAL_ALLOWED_IPS`: Comma-separated list of allowed IP ranges
- `INTERNAL_API_KEY`: (Secret) API key for internal service communication

## Deployment Instructions

1. Review and update the environment variables in `env.yaml` as needed
2. Make sure your kubectl context is set to the correct cluster
3. Run the deployment script:

```bash
./deploy-inference.sh
```

The script will:

- Apply the environment configuration (ConfigMap and Secret)
- Check if necessary persistent volume claims exist and create them if needed
- Deploy the inference service
- Wait for the deployment to be ready
- Display information about accessing the service and logs

## Accessing the Service

Once deployed, the service will be available at:

- Web API: `https://inference.longstorymedia.com`
- gRPC: `inference-service.ollama.svc.cluster.local:50051` (internal to the cluster)

## Monitoring and Troubleshooting

To check the logs:

```bash
kubectl logs -f -n ollama -l app=inference-service
```

To shell into the pod:

```bash
kubectl exec -it -n ollama $(kubectl get pod -n ollama -l app=inference-service -o jsonpath='{.items[0].metadata.name}') -- /bin/bash
```

To check the status of the deployment:

```bash
kubectl get deployment -n ollama inference-service
```

To describe the pod for troubleshooting:

```bash
kubectl describe pod -n ollama -l app=inference-service
```
