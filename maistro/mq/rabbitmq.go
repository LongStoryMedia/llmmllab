package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"maistro/config"
	"maistro/models"
	"maistro/util"
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

	var err error

	// Connect to RabbitMQ
	c.conn, err = amqp.Dial(c.connString)
	if err != nil {
		return fmt.Errorf("failed to connect to RabbitMQ: %w", err)
	}

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
		QueueNameResults:          "result.*",
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

	// Set up consumers
	c.resultConsumer, err = c.resultChan.Consume(
		QueueNameResults, // queue
		"",               // consumer
		false,            // auto-ack
		false,            // exclusive
		false,            // no-local
		false,            // no-wait
		nil,              // args
	)
	if err != nil {
		c.cleanup()
		return fmt.Errorf("failed to register result consumer: %w", err)
	}

	c.statusConsumer, err = c.statusChan.Consume(
		QueueNameStatus, // queue
		"",              // consumer
		false,           // auto-ack
		false,           // exclusive
		false,           // no-local
		false,           // no-wait
		nil,             // args
	)
	if err != nil {
		c.cleanup()
		return fmt.Errorf("failed to register status consumer: %w", err)
	}

	c.initialized = true
	logrus.Info("RabbitMQ client initialized successfully")

	// Start consumers
	go c.consumeResults()
	go c.consumeStatus()

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
	if req.Type == models.InferenceQueueMessageTypeImage {
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
				logrus.Warn("Result consumer channel closed")
				return
			}

			util.LogDebug("Received result message", logrus.Fields{
				"routingKey":    msg.RoutingKey,
				"correlationId": msg.CorrelationId,
				"type":          msg.Type,
			})

			var result models.InferenceQueueMessage
			if err := json.Unmarshal(msg.Body, &result); err != nil {
				util.HandleError(err)
				msg.Reject(false)
				continue
			}

			// Process the result with the registered handler
			util.LogInfo("Received inference result", logrus.Fields{
				"correlation_id": result.CorrelationID,
			})

			// Call the handler registered for this request ID
			HandleResult(CorrelationID(result.CorrelationID), result, nil)
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
				logrus.Warn("Status consumer channel closed")
				return
			}

			// Process status update
			logrus.WithField("routingKey", msg.RoutingKey).Info("Received status update")

			// Acknowledge the message
			msg.Ack(false)
		}
	}
}
