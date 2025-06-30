package mq

import (
	"fmt"
	"maistro/models"
	"maistro/util"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

type CorrelationID string

// ResultHandler is a function that processes a result from RabbitMQ
type ResultHandler func(correlationID CorrelationID, result models.InferenceQueueMessage, err error)

// InferenceResultHandlers maps request IDs to their handlers
var InferenceResultHandlers = make(map[CorrelationID]ResultHandler)

// RegisterResultHandler registers a handler for a specific request ID
func RegisterResultHandler(correlationID CorrelationID, handler ResultHandler) {
	util.LogDebug("Registering result handler for request ID", logrus.Fields{
		"requestId": correlationID,
	})
	InferenceResultHandlers[correlationID] = handler
}

// HandleResult processes a result for a specific request ID
func HandleResult(correlationID CorrelationID, result models.InferenceQueueMessage, err error) {
	handler, ok := InferenceResultHandlers[correlationID]
	if !ok {
		util.LogWarning("No handler registered for request ID", logrus.Fields{
			"requestId": correlationID,
		})
		return
	}

	// Call the handler
	handler(correlationID, result, err)

	// Remove the handler once it's been called
	delete(InferenceResultHandlers, correlationID)
}

// ToRequestID generates a unique request ID based on conversation ID and user ID
// If guid is provided, it uses that; otherwise, it generates a new UUID
// The request ID format is "UUID::conversationID-userID"
func ToRequestID(conversationID int, userID string, guid *string) CorrelationID {
	var id string
	if guid == nil {
		id = uuid.New().String()
	} else {
		id = *guid
	}
	// Generate a unique request ID based on conversation ID and user ID
	return CorrelationID(fmt.Sprintf("%s::%s", util.CorrelationID(conversationID, userID), id))
}

// FromRequestID extracts the user defined or generated uuid, user ID and conversation ID from a request ID
// Returns the user defined or generated uuid, user ID, conversation ID, and an error if the format is invalid
// The request ID format is "UUID::conversationID-userID"
func FromRequestID(requestID string) (string, int, string, error) {
	// Split the request ID into conversation ID and user ID
	parts := strings.Split(requestID, "::")
	if len(parts) != 2 {
		return "", -1, "", fmt.Errorf("invalid request ID format")
	}

	cid, uid, err := util.FromCorrelationID(parts[0])
	if err != nil {
		return "", -1, "", fmt.Errorf("invalid conversation ID: %w", err)
	}

	return parts[1], cid, uid, nil
}

// SubmitImageGenerationRequest sends an image generation request to RabbitMQ
func SubmitImageGenerationRequest(
	client *RabbitMQClient,
	image models.ImageGenerateRequest,
	userID string,
	conversationID int,
	priority int,
	handler ResultHandler,
) (CorrelationID, error) {
	// Generate a unique request ID
	requestID := ToRequestID(conversationID, userID, nil)

	// Register the handler for this request
	RegisterResultHandler(requestID, handler)

	req := models.InferenceQueueMessage{
		CorrelationID:  string(requestID),
		Payload:        &image,
		Priority:       priority,
		Type:           models.InferenceQueueMessageTypeImage,
		Timestamp:      time.Now(),
		MemoryRequired: util.Gb2b(10), // Convert GB to bytes
	}

	// Publish the request to RabbitMQ
	if err := client.PublishRequest(req); err != nil {
		delete(InferenceResultHandlers, requestID)
		return "", err
	}

	return requestID, nil
}
