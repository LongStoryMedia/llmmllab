# RabbitMQ Integration for Inference Request Management

## Problem Statement

The current inference request scheduling system has several limitations:
1. Memory availability is checked only after a request is enqueued
2. Prioritization is basic and doesn't dynamically adapt to system load
3. Memory requirements are estimated rather than precisely measured
4. When multiple models are loaded, GPU memory can become fragmented
5. Communication between maistro and inference services lacks robustness

## Proposed Solution: RabbitMQ with Streams

Implementing RabbitMQ with its Streams feature will provide:

1. **Decoupled Communication**: Separate service interactions for better fault tolerance
2. **Persistent Messaging**: Ensure no requests are lost if a service crashes
3. **Advanced Prioritization**: Better request prioritization with dynamic TTL
4. **Improved Resource Awareness**: Better memory management based on model requirements
5. **Conversation Context History**: Store message chains for flexible context retrieval

## Implementation Plan

### 1. RabbitMQ Infrastructure

- **Clusters**: Set up redundant RabbitMQ clusters for high availability
- **Streams**: Use RabbitMQ Streams for message chains and conversation history
- **Exchanges**: Create topic exchanges for different message types:
  - `inference.request` - For new inference requests
  - `inference.result` - For completed inference results
  - `inference.image.request` - For image generation and editing requests

### Task Types

- **Text Generation**: Standard text inference requests
- **Image Generation**: Generate images from text prompts
- **Image Editing**: Edit existing images based on text prompts and image source
- **Embedding Generation**: Generate embeddings for text
  - `inference.status` - For status updates during processing

### 2. Queue Structure

```
- inference.request.high     # Priority 0-3, processed immediately
- inference.request.medium   # Priority 4-7, moderate delay
- inference.request.low      # Priority 8+, longer delay
- inference.image.request    # Image generation requests (separate resource pool)
- inference.embedding.request # Embedding requests (typically smaller models)
```

### 3. Message Structure

#### Text Generation Request

```json
{
  "requestId": "uuid-123",
  "userId": "user-456",
  "conversationId": 789,
  "modelProfile": {
    "modelName": "qwen3-30b",
    "requirements": {
      "vram": 10737418240,
      "cpuMemory": 2000000000
    }
  },
  "priority": 3,
  "messages": [{"role": "user", "content": "..."}],
  "streamResponse": true,
  "timestamp": "2025-06-26T15:29:37.028Z"
}
```

#### Image Editing Request

```json
{
  "correlation_id": "uuid-123",
  "priority": 10,
  "type": "request",
  "task": "image_editing",
  "timestamp": "2025-06-26T15:29:37.028Z",
  "payload": {
    "prompt": "Make the sky blue",
    "negative_prompt": "anime, cartoon, sketch, drawing",
    "model": "stabilityai/stable-diffusion-3.5-large",
    "width": 1024,
    "height": 1024,
    "inference_steps": 30,
    "guidance_scale": 7.5,
    "image": "http://maistro:8080/internal/images/user-456/image-789.png",
    "conversation_id": 789
  }
}
```

### 4. Consumer Design

#### Maistro Service

```go
type RabbitMQHandler struct {
    client *amqp.Connection
    publishChannel *amqp.Channel
    resultConsumer *amqp.Channel
}

func (r *RabbitMQHandler) PublishRequest(req *InferenceRequest) (string, error) {
    // Determine queue based on priority
    queue := "inference.request.high"
    if req.Priority > 3 && req.Priority < 8 {
        queue = "inference.request.medium"
    } else if req.Priority >= 8 {
        queue = "inference.request.low"
    }
    
    // Generate request ID
    requestID := uuid.New().String()
    
    // Convert to message
    msg := createMessage(requestID, req)
    
    // Publish with TTL based on priority
    ttl := req.Priority * 1000 // milliseconds
    return requestID, r.publishChannel.Publish(
        "inference.exchange", 
        queue,
        false,
        false,
        amqp.Publishing{
            ContentType: "application/json",
            Body: msg,
            Expiration: strconv.Itoa(ttl),
        })
}

func (r *RabbitMQHandler) ConsumeResults() {
    // Set up consumer for inference.result queue
    // Process results back to client
}
```

#### Inference Service

```python
class RabbitMQConsumer:
    def __init__(self, config):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=config.rabbitmq_host))
        self.channel = self.connection.channel()
        self.hardware_mgr = HardwareManager()
        
    async def consume_requests(self):
        # Setup queues with priority consumers
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue="inference.request.high",
            on_message_callback=self.process_request
        )
        # Similar for medium and low queues
        
    def process_request(self, ch, method, properties, body):
        # Parse request
        request = json.loads(body)
        
        # Check hardware resources
        if self.hardware_mgr.check_availability(request["modelProfile"]["requirements"]):
            # Process request
            result = self.process_inference(request)
            
            # Publish result
            self.publish_result(request["requestId"], result)
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            # Reject and requeue if resources not available
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            time.sleep(1)  # Wait before retrying
```

### 5. Stream for Conversation Context

```
- conversation.{conversationId} # Stream for each conversation
```

This allows:
1. Multiple consumers to read conversation history
2. Different models to use varying amounts of context
3. Processing of conversation summaries asynchronously

## Benefits of this Approach

1. **Fault Tolerance**: If the inference service crashes, requests remain in the queue
2. **Scalability**: Can add more inference workers dynamically
3. **Resource Optimization**: Better utilization of GPU resources
4. **Priority Handling**: True priority-based processing with delays
5. **Monitoring**: Better observability of system throughput and bottlenecks
6. **Context Management**: Flexible context storage and retrieval

## Implementation Phases

1. **Phase 1**: Set up RabbitMQ basic integration with simple queues
2. **Phase 2**: Implement priority-based queue system
3. **Phase 3**: Add Streams for conversation history
4. **Phase 4**: Optimize resource allocation with dynamic adjustment
5. **Phase 5**: Add monitoring and observability

## Metrics to Track

- Queue depth per priority level
- Request processing time
- Wait time in queue
- Memory utilization
- Success/failure rates
- Model load/unload frequency

## Conclusion

This RabbitMQ integration will significantly improve the stability, performance, and flexibility of the inference system, solving the current issues with memory management and prioritization while enabling better conversational context management.
