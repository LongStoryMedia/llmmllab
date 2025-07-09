package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"maistro/config"
	"maistro/models"
	"maistro/util"
	"strings"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"github.com/sirupsen/logrus"
)

// RabbitMQ connection details
const (
	ExchangeNameInference     = "inference.exchange"
	QueueNameRequestHigh      = "inference.request.high"      // Priority 0-3
	QueueNameRequestMedium    = "inference.request.medium"    // Priority 4-7
	QueueNameRequestLow       = "inference.request.low"       // Priority 8+
	QueueNameImageRequest     = "inference.image.request"     // Image generation
	QueueNameEmbeddingRequest = "inference.embedding.request" // Embeddings
	QueueNameResults          = "inference.results"
	QueueNameStatus           = "inference.status"
)

// RabbitMQClient handles communication with RabbitMQ
type RabbitMQClient struct {
	conn           *amqp.Connection
	pubChan        *amqp.Channel
	resultChan     *amqp.Channel
	statusChan     *amqp.Channel
	resultConsumer <-chan amqp.Delivery
	statusConsumer <-chan amqp.Delivery
	connString     string
	initialized    bool
	ctx            context.Context
	cancelCtx      context.CancelFunc
}

// NewRabbitMQClient creates a new RabbitMQ client
func NewRabbitMQClient() *RabbitMQClient {
	// Create a context that can be cancelled when shutting down
	ctx, cancel := context.WithCancel(context.Background())

	// Get config and set connection string
	conf := config.GetConfig(nil)
	connString := fmt.Sprintf("amqp://%s:%s@%s:%d/",
		conf.Rabbitmq.User,
		conf.Rabbitmq.Password,
		conf.Rabbitmq.Host,
		conf.Rabbitmq.Port)

	return &RabbitMQClient{
		connString: connString,
		ctx:        ctx,
		cancelCtx:  cancel,
	}
}

// Initialize sets up the RabbitMQ connection and channels
func (c *RabbitMQClient) Initialize() error {
	if c.initialized {
		return nil
	}

	// Initialize handler registry
	initHandlerRegistry()

	var err error

	// Connect to RabbitMQ with retry logic
	const connectRetries = 3
	var lastErr error

	for i := 0; i < connectRetries; i++ {
		c.conn, err = amqp.Dial(c.connString)
		if err == nil {
			break
		}

		lastErr = err
		util.LogWarning("Failed to connect to RabbitMQ, retrying", logrus.Fields{
			"error":      err.Error(),
			"retry":      i + 1,
			"maxRetries": connectRetries,
		})

		if i < connectRetries-1 {
			time.Sleep(time.Duration(i+1) * 2 * time.Second)
		}
	}

	if err != nil {
		return fmt.Errorf("failed to connect to RabbitMQ after %d retries: %w", connectRetries, lastErr)
	}

	// Set up connection closed notification to trigger reconnection
	connClosedChan := make(chan *amqp.Error)
	c.conn.NotifyClose(connClosedChan)

	// Handle connection closure in a separate goroutine
	go func() {
		for {
			select {
			case err := <-connClosedChan:
				if err != nil {
					util.LogWarning("RabbitMQ connection closed unexpectedly", logrus.Fields{
						"error": err.Error(),
					})
					c.reconnect()
				}
				return
			case <-c.ctx.Done():
				return
			}
		}
	}()

	// Create publisher channel
	c.pubChan, err = c.conn.Channel()
	if err != nil {
		c.conn.Close()
		return fmt.Errorf("failed to open channel: %w", err)
	}

	// Create result consumer channel
	c.resultChan, err = c.conn.Channel()
	if err != nil {
		c.pubChan.Close()
		c.conn.Close()
		return fmt.Errorf("failed to open result channel: %w", err)
	}

	// Create status consumer channel
	c.statusChan, err = c.conn.Channel()
	if err != nil {
		c.resultChan.Close()
		c.pubChan.Close()
		c.conn.Close()
		return fmt.Errorf("failed to open status channel: %w", err)
	}

	// Declare exchange
	err = c.pubChan.ExchangeDeclare(
		ExchangeNameInference, // name
		"topic",               // type
		true,                  // durable
		false,                 // auto-deleted
		false,                 // internal
		false,                 // no-wait
		nil,                   // arguments
	)
	if err != nil {
		c.cleanup()
		return fmt.Errorf("failed to declare exchange: %w", err)
	}

	// Declare queues
	queues := []string{
		QueueNameRequestHigh,
		QueueNameRequestMedium,
		QueueNameRequestLow,
		QueueNameImageRequest,
		QueueNameEmbeddingRequest,
		QueueNameResults,
		QueueNameStatus,
	}

	for _, q := range queues {
		_, err = c.pubChan.QueueDeclare(
			q,     // name
			true,  // durable
			false, // delete when unused
			false, // exclusive
			false, // no-wait
			nil,   // arguments
		)
		if err != nil {
			c.cleanup()
			return fmt.Errorf("failed to declare queue %s: %w", q, err)
		}
	}

	// Bind queues to exchange with routing keys
	bindingKeys := map[string]string{
		QueueNameRequestHigh:      "request.high",
		QueueNameRequestMedium:    "request.medium",
		QueueNameRequestLow:       "request.low",
		QueueNameImageRequest:     "request.image",
		QueueNameEmbeddingRequest: "request.embedding",
		QueueNameResults:          "result.#", // Changed to # wildcard for multi-level routing key matching
		QueueNameStatus:           "status.*",
	}

	for queue, key := range bindingKeys {
		err = c.pubChan.QueueBind(
			queue,                 // queue name
			key,                   // routing key
			ExchangeNameInference, // exchange
			false,
			nil,
		)
		if err != nil {
			c.cleanup()
			return fmt.Errorf("failed to bind queue %s: %w", queue, err)
		}
	}

	// Set up consumers with retry
	maxRetries := 3
	var retryErr error

	for i := 0; i < maxRetries; i++ {
		c.resultConsumer, err = c.resultChan.Consume(
			QueueNameResults, // queue
			"",               // consumer
			false,            // auto-ack
			false,            // exclusive
			false,            // no-local
			false,            // no-wait
			nil,              // args
		)

		if err == nil {
			break
		}

		retryErr = err
		util.LogWarning("Failed to register result consumer, retrying", logrus.Fields{
			"error": err.Error(),
			"retry": i + 1,
		})

		if i < maxRetries-1 {
			time.Sleep(time.Duration(i+1) * time.Second)
		}
	}

	if err != nil {
		c.cleanup()
		return fmt.Errorf("failed to register result consumer after %d retries: %w", maxRetries, retryErr)
	}

	// Set up status consumer with retry
	for i := 0; i < maxRetries; i++ {
		c.statusConsumer, err = c.statusChan.Consume(
			QueueNameStatus, // queue
			"",              // consumer
			false,           // auto-ack
			false,           // exclusive
			false,           // no-local
			false,           // no-wait
			nil,             // args
		)

		if err == nil {
			break
		}

		retryErr = err
		util.LogWarning("Failed to register status consumer, retrying", logrus.Fields{
			"error": err.Error(),
			"retry": i + 1,
		})

		if i < maxRetries-1 {
			time.Sleep(time.Duration(i+1) * time.Second)
		}
	}

	if err != nil {
		c.cleanup()
		return fmt.Errorf("failed to register status consumer after %d retries: %w", maxRetries, retryErr)
	}

	c.initialized = true
	logrus.Info("RabbitMQ client initialized successfully")

	// Verify connection status
	c.VerifyConnection()

	// Start consumers
	go c.consumeResults()
	go c.consumeStatus()
	go c.startHealthCheck()

	return nil
}

func (c *RabbitMQClient) cleanup() {
	if c.statusChan != nil {
		c.statusChan.Close()
	}
	if c.resultChan != nil {
		c.resultChan.Close()
	}
	if c.pubChan != nil {
		c.pubChan.Close()
	}
	if c.conn != nil {
		c.conn.Close()
	}
}

// Close closes the connection and channels
func (c *RabbitMQClient) Close() {
	c.cancelCtx()
	c.cleanup()
	c.initialized = false
	// Close handler registry
	if handlerReg != nil {
		closeHandlerRegistry()
	}
	logrus.Info("RabbitMQ client closed")
}

// Initialized returns whether the client is initialized
func (c *RabbitMQClient) Initialized() bool {
	return c.initialized
}

// PublishRequest publishes an inference request to the appropriate queue based on priority
func (c *RabbitMQClient) PublishRequest(req models.InferenceQueueMessage) error {
	if !c.initialized {
		return fmt.Errorf("RabbitMQ client not initialized")
	}

	// Determine queue based on priority
	routingKey := "request.high" // Default to high priority
	if req.Priority >= 4 && req.Priority <= 7 {
		routingKey = "request.medium"
	} else if req.Priority > 7 {
		routingKey = "request.low"
	}

	// Handle image requests separately
	if req.Task == models.InferenceQueueMessageTaskImageGeneration ||
		req.Task == models.InferenceQueueMessageTaskImageEditing {
		routingKey = "request.image"
	}

	// Set TTL based on priority (milliseconds)
	ttl := max(req.Priority*1000, 0)

	// Convert request to JSON
	body, err := json.Marshal(req)
	if err != nil {
		return util.HandleError(err)
	}

	// Publish to exchange
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = c.pubChan.PublishWithContext(
		ctx,
		ExchangeNameInference, // exchange
		routingKey,            // routing key
		false,                 // mandatory
		false,                 // immediate
		amqp.Publishing{
			DeliveryMode:  amqp.Persistent,
			ContentType:   "application/json",
			Body:          body,
			Timestamp:     time.Now(),
			Expiration:    fmt.Sprintf("%d", ttl),
			CorrelationId: req.CorrelationID,
		},
	)

	if err != nil {
		return util.HandleError(err)
	}

	util.LogInfo("Published inference request to RabbitMQ", logrus.Fields{
		"routingKey": routingKey,
		"priority":   req.Priority,
		"ttl":        ttl,
	})
	return nil
}

// consumeResults processes messages from the result queue
func (c *RabbitMQClient) consumeResults() {
	for {
		select {
		case <-c.ctx.Done():
			return
		case msg, ok := <-c.resultConsumer:
			if !ok {
				util.LogWarning("Result consumer channel closed")
				// Try to reconnect
				go c.reconnect() // Do reconnection in a new goroutine to avoid blocking
				return
			}

			// Log extensive details about the received message
			util.LogInfo("Received raw result message", logrus.Fields{
				"routingKey":    msg.RoutingKey,
				"correlationId": msg.CorrelationId,
				"exchange":      msg.Exchange,
				"contentType":   msg.ContentType,
				"bodyLength":    len(msg.Body),
				"deliveryTag":   msg.DeliveryTag,
			})

			// Log the raw message body for debugging
			msgBodyStr := string(msg.Body)
			if len(msgBodyStr) > 500 {
				msgBodyStr = msgBodyStr[:500] + "... (truncated)"
			}
			util.LogDebug("Raw message body", logrus.Fields{
				"body": msgBodyStr,
			})

			var result models.InferenceQueueMessage
			if err := json.Unmarshal(msg.Body, &result); err != nil {
				util.HandleError(err, logrus.Fields{
					"message":    "Failed to unmarshal result message",
					"body":       msgBodyStr,
					"routingKey": msg.RoutingKey,
				})
				msg.Reject(false)
				continue
			}

			// Log detailed result message info after unmarshaling
			util.LogInfo("Unmarshaled result message", logrus.Fields{
				"correlation_id": result.CorrelationID,
				"routing_key":    msg.RoutingKey,
				"task":           result.Task,
				"type":           result.Type,
				"payload_type":   fmt.Sprintf("%T", result.Payload),
				"timestamp":      result.Timestamp,
			})

			// Try to extract request ID from payload if available
			requestID := ""
			if resultMap, ok := result.Payload.(map[string]interface{}); ok {
				// Look for requestId or request_id in the payload
				if id, ok := resultMap["requestId"].(string); ok {
					requestID = id
					util.LogInfo("Found requestId in payload", logrus.Fields{
						"requestId": id,
					})
				} else if id, ok := resultMap["request_id"].(string); ok {
					requestID = id
					util.LogInfo("Found request_id in payload", logrus.Fields{
						"request_id": id,
					})
				}
			}

			// If we found a request ID in the payload that doesn't match the correlation ID,
			// try using both to find a handler
			correlationID := CorrelationID(result.CorrelationID)
			handled := false

			// Try various correlation ID formats:
			// 1. First try with the original correlation ID
			handled = handlerReg.handle(correlationID, result, nil)

			// 2. If that didn't work and we have a different request ID from the payload, try that
			if !handled && requestID != "" && requestID != result.CorrelationID {
				util.LogInfo("Trying alternative request ID from payload", logrus.Fields{
					"original":    result.CorrelationID,
					"alternative": requestID,
				})
				handled = handlerReg.handle(CorrelationID(requestID), result, nil)
			}

			// 3. For image editing tasks, try to extract the UUID part if it's in a format like -99-CgNsc20SBGxkYXA::4369feb9-a493-403f-a4d6-4f007c3bb574
			if !handled && result.Task == "image" && strings.Contains(result.CorrelationID, "::") {
				parts := strings.Split(result.CorrelationID, "::")
				if len(parts) == 2 {
					// Try to use just the UUID part (after ::)
					uuidPart := CorrelationID(parts[1])
					util.LogInfo("Trying UUID part as correlation ID", logrus.Fields{
						"original": result.CorrelationID,
						"uuid":     uuidPart,
					})
					handled = handlerReg.handle(uuidPart, result, nil)

					// If that didn't work, try using a registered handler with a different prefix
					// but the same UUID part
					if !handled {
						util.LogInfo("Searching for handler with matching UUID part")
						handled = handlerReg.handleWithUUIDPart(parts[1], result)
					}
				}
			}

			if !handled {
				util.LogWarning("No handler found for correlation ID", logrus.Fields{
					"correlationId": correlationID,
					"requestId":     requestID,
					"routingKey":    msg.RoutingKey,
				})

				// Dump the current handlers for debugging
				handlerReg.logRegisteredHandlers()
			} else {
				util.LogInfo("Successfully handled result message", logrus.Fields{
					"correlationId": correlationID,
				})
			}

			// Acknowledge the message
			msg.Ack(false)
		}
	}
}

// consumeStatus processes messages from the status queue
func (c *RabbitMQClient) consumeStatus() {
	for {
		select {
		case <-c.ctx.Done():
			return
		case msg, ok := <-c.statusConsumer:
			if !ok {
				util.LogWarning("Status consumer channel closed")
				// Try to reconnect
				go c.reconnect() // Do reconnection in a new goroutine to avoid blocking
				return
			}

			// Process status update
			logrus.WithField("routingKey", msg.RoutingKey).Info("Received status update")

			// Acknowledge the message
			msg.Ack(false)
		}
	}
}

// reconnect attempts to reconnect to RabbitMQ after a connection failure
func (c *RabbitMQClient) reconnect() {
	// Clean up existing connections
	c.cleanup()
	c.initialized = false

	// Try to reconnect
	util.LogInfo("Attempting to reconnect to RabbitMQ")

	// Try multiple times with increasing delay
	maxRetries := 5
	for i := 0; i < maxRetries; i++ {
		err := c.Initialize()
		if err == nil {
			util.LogInfo("Successfully reconnected to RabbitMQ")
			return
		}

		delay := time.Duration(i+1) * 5 * time.Second
		util.LogWarning("Failed to reconnect to RabbitMQ", logrus.Fields{
			"error":       err.Error(),
			"retry":       i + 1,
			"maxRetries":  maxRetries,
			"nextRetryIn": delay.String(),
		})

		if i < maxRetries-1 {
			time.Sleep(delay)
		}
	}

	// After maxRetries, schedule a background reconnection attempt
	go func() {
		time.Sleep(30 * time.Second)
		c.reconnect()
	}()
}

// startHealthCheck starts a goroutine to periodically check the health of the connection
func (c *RabbitMQClient) startHealthCheck() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		// Counter for periodic test message publishing
		counter := 0

		for {
			select {
			case <-ticker.C:
				c.VerifyConnection()

				// Every 5 minutes, send a test message to verify the results queue
				counter++
				if counter >= 10 { // 10 * 30 seconds = 5 minutes
					counter = 0
					err := c.TestResultsQueue()
					if err != nil {
						util.LogWarning("Failed to publish test message", logrus.Fields{
							"error": err.Error(),
						})
					}
				}
			case <-c.ctx.Done():
				return
			}
		}
	}()
}

// ForceReconnect forces the client to reconnect to RabbitMQ
// This is useful for debugging connection issues or when messages stop flowing
func (c *RabbitMQClient) ForceReconnect() error {
	util.LogInfo("Forcing RabbitMQ reconnection")

	// Close existing connections
	c.cleanup()
	c.initialized = false

	// Reinitialize
	err := c.Initialize()
	if err != nil {
		return util.HandleError(err, logrus.Fields{
			"message": "Failed to force reconnect to RabbitMQ",
		})
	}

	// Send a test message to verify the connection
	err = c.TestResultsQueue()
	if err != nil {
		return util.HandleError(err, logrus.Fields{
			"message": "Failed to send test message after force reconnect",
		})
	}

	util.LogInfo("Force reconnect completed successfully")
	return nil
}

// VerifyConnection checks the connection and channel status
func (c *RabbitMQClient) VerifyConnection() {
	if c.conn == nil {
		util.LogWarning("RabbitMQ connection is nil")
		return
	}

	if c.conn.IsClosed() {
		util.LogWarning("RabbitMQ connection is closed")
		c.reconnect()
		return
	}

	// Check channels
	if c.resultChan == nil || c.resultChan.IsClosed() {
		util.LogWarning("Result channel is nil or closed")
		c.reconnect()
		return
	}

	if c.statusChan == nil || c.statusChan.IsClosed() {
		util.LogWarning("Status channel is nil or closed")
		c.reconnect()
		return
	}

	if c.pubChan == nil || c.pubChan.IsClosed() {
		util.LogWarning("Publisher channel is nil or closed")
		c.reconnect()
		return
	}

	// Log detailed status of result consumer
	c.LogResultConsumerStatus()

	util.LogInfo("RabbitMQ connection verified and healthy")
}

// LogResultConsumerStatus logs the status of the result consumer
func (c *RabbitMQClient) LogResultConsumerStatus() {
	if c.resultConsumer == nil {
		util.LogWarning("Result consumer channel is nil")
		return
	}

	// Check if the channel is open
	if c.resultChan == nil || c.resultChan.IsClosed() {
		util.LogWarning("Result channel is closed or nil")
		return
	}

	// Log channel details
	util.LogInfo("Result consumer channel status", logrus.Fields{
		"channelOpen":       !c.resultChan.IsClosed(),
		"consumerAvailable": c.resultConsumer != nil,
	})

	// Try to inspect the queue using QueueDeclare with passive=true (but no extra arguments)
	queue, err := c.resultChan.QueueDeclare(
		QueueNameResults,            // name
		true,                        // durable
		false,                       // delete when unused
		false,                       // exclusive
		false,                       // no-wait
		amqp.Table{"passive": true}, // only set passive flag without other arguments
	)
	if err != nil {
		util.LogWarning("Failed to inspect result queue", logrus.Fields{
			"error": err.Error(),
		})
		return
	}

	util.LogInfo("Result queue status", logrus.Fields{
		"queueName":      QueueNameResults,
		"messageCount":   queue.Messages,
		"consumerCount":  queue.Consumers,
		"bindingPattern": "result.#",
	})
}

// TestResultsQueue sends a test message to the results queue to verify it's working
func (c *RabbitMQClient) TestResultsQueue() error {
	if !c.initialized {
		return fmt.Errorf("RabbitMQ client not initialized")
	}

	// Create a test message
	testMsg := models.InferenceQueueMessage{
		CorrelationID:  "test-correlation-id",
		Priority:       0,
		Task:           models.InferenceQueueMessageTaskImageEditing,
		Timestamp:      time.Now(),
		MemoryRequired: 0,
		Type:           util.StrPtr("result"),
		Payload: map[string]interface{}{
			"test":       true,
			"request_id": "test-correlation-id", // Add request_id for compatibility
			"timestamp":  time.Now().Format(time.RFC3339),
		},
	}

	// Convert to JSON
	body, err := json.Marshal(testMsg)
	if err != nil {
		return util.HandleError(err)
	}

	// Try to inspect the queue before publishing
	queue, err := c.resultChan.QueueDeclare(
		QueueNameResults,            // name
		true,                        // durable
		false,                       // delete when unused
		false,                       // exclusive
		false,                       // no-wait
		amqp.Table{"passive": true}, // only set passive flag without other arguments
	)
	if err != nil {
		util.LogWarning("Failed to inspect results queue", logrus.Fields{
			"error": err.Error(),
		})
	} else {
		util.LogInfo("Results queue status before test message", logrus.Fields{
			"messageCount":  queue.Messages,
			"consumerCount": queue.Consumers,
			"name":          queue.Name,
		})
	}

	// Publish using multiple routing keys for better testing
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	util.LogInfo("Publishing test message to results queue")

	// Try multiple routing keys
	routingKeys := []string{"result.test", "result.inference", "result"}

	for _, key := range routingKeys {
		err = c.pubChan.PublishWithContext(
			ctx,
			ExchangeNameInference, // exchange
			key,                   // routing key
			false,                 // mandatory
			false,                 // immediate
			amqp.Publishing{
				DeliveryMode:  amqp.Persistent,
				ContentType:   "application/json",
				Body:          body,
				Timestamp:     time.Now(),
				CorrelationId: "test-correlation-id",
			},
		)

		if err != nil {
			util.LogWarning("Failed to publish test message with routing key", logrus.Fields{
				"routingKey": key,
				"error":      err.Error(),
			})
		} else {
			util.LogInfo("Test message published successfully", logrus.Fields{
				"routingKey": key,
			})
		}
	}

	// Also try direct publish to queue
	err = c.pubChan.PublishWithContext(
		ctx,
		"",               // use default exchange
		QueueNameResults, // use queue name as routing key
		false,            // mandatory
		false,            // immediate
		amqp.Publishing{
			DeliveryMode:  amqp.Persistent,
			ContentType:   "application/json",
			Body:          body,
			Timestamp:     time.Now(),
			CorrelationId: "test-correlation-id",
		},
	)

	if err != nil {
		util.LogWarning("Failed to publish test message directly to queue", logrus.Fields{
			"error": err.Error(),
		})
	} else {
		util.LogInfo("Test message published directly to queue successfully")
	}

	// Check queue again after publishing
	queue, err = c.resultChan.QueueDeclare(
		QueueNameResults,            // name
		true,                        // durable
		false,                       // delete when unused
		false,                       // exclusive
		false,                       // no-wait
		amqp.Table{"passive": true}, // only set passive flag without other arguments
	)
	if err != nil {
		util.LogWarning("Failed to inspect results queue after publishing", logrus.Fields{
			"error": err.Error(),
		})
	} else {
		util.LogInfo("Results queue status after test message", logrus.Fields{
			"messageCount":  queue.Messages,
			"consumerCount": queue.Consumers,
			"name":          queue.Name,
		})
	}

	return nil
}
