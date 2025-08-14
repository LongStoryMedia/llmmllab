# YAML Schemas

This directory contains YAML schema definitions for various components of the LLM ML Lab. These schemas define the structure, data types, and validation rules for objects used throughout the system.

## Overview

The schemas provide a common type system that ensures consistent data structures across all services. They're used for:

1. **Request/Response Validation** - Ensuring API payloads conform to expected formats
2. **Code Generation** - Generating type-safe code in multiple languages
3. **Documentation** - Self-documenting data structures with descriptions
4. **Configuration** - Defining configuration file formats

## Key Schema Categories

### Message Schemas

- `message.yaml` - Base message structure
- `message_content.yaml` - Content structure for messages
- `message_role.yaml` - Role definitions (user, assistant, system)
- `chat_request.yaml` - Chat API request format
- `chat_response.yaml` - Chat API response format

### Model Schemas

- `model.yaml` - Model metadata and configuration
- `model_details.yaml` - Detailed model information
- `model_parameters.yaml` - Inference parameters
- `model_profile.yaml` - User-specific model settings
- `model_task.yaml` - Task types for models

### Infrastructure Schemas

- `rabbitmq_config.yaml` - RabbitMQ configuration
- `database_config.yaml` - Database connection settings
- `inference_service_config.yaml` - Service configuration
- `inference_queue_message.yaml` - Message queue format

### WebSocket Schemas

- `web_socket_connection.yaml` - WebSocket connection definition
- `socket_message.yaml` - WebSocket message format
- `socket_status_update.yaml` - Status update format
- `socket_connection_type.yaml` - Connection type enum

### Context Extension Schemas

- `memory.yaml` - Memory object definition
- `memory_fragment.yaml` - Memory fragment structure
- `summary.yaml` - Summary data format
- `summarization_config.yaml` - Configuration for summarization

## Usage

These schemas are referenced by code generators and validation libraries to ensure type safety and data integrity. Each schema follows the JSON Schema standard (draft-07) with extensions for documentation and cross-references.

```yaml
# Example schema reference
properties:
  metadata:
    $ref: chat_message_metadata.yaml  # Cross-reference to another schema
```

## Integration Points

- **Server Code** - Uses schemas for request/response validation
- **Client Code** - Uses schemas for type-safe API interactions
- **Documentation** - Uses schemas to generate API documentation
- **Testing** - Uses schemas to validate test data

## Schema Format

Each schema follows this general structure:

```yaml
$schema: http://json-schema.org/draft-07/schema#
title: SchemaName
description: Description of what this schema represents
type: object  # Most schemas are objects
properties:
  property1:
    type: string
    description: What this property is for
  property2:
    type: integer
    description: What this property is for
required:
  - property1  # List of required properties
```
