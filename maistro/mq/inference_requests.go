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
// Supports multiple formats:
// - "UUID::conversationID-userID" (standard format)
// - "-99-CgNsc20SBGxkYXA::UUID" (alternate format from external systems)
// - Any other format will attempt to extract just a UUID if possible
func FromRequestID(requestID string) (string, int, string, error) {
	// Handle the inference service format which might look like -99-CgNsc20SBGxkYXA::4369feb9-a493-403f-a4d6-4f007c3bb574
	if strings.HasPrefix(requestID, "-") && strings.Contains(requestID, "::") {
		util.LogInfo("Found alternate format request ID", logrus.Fields{
			"requestId": requestID,
		})

		// Split on :: and use the UUID part
		parts := strings.Split(requestID, "::")
		if len(parts) == 2 {
			uuid := parts[1]
			// Return default values for conversation ID and user ID
			return uuid, -99, "system", nil
		}
	}

	// Try standard format
	parts := strings.Split(requestID, "::")
	if len(parts) != 2 {
		// If we can't parse the ID format, just return the whole thing as a UUID
		return requestID, -1, "", fmt.Errorf("invalid request ID format: %s", requestID)
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
	handlerReg.register(requestID, handler, HandlerTimeoutDuration)

	req := models.InferenceQueueMessage{
		CorrelationID:  string(requestID),
		Payload:        &image,
		Priority:       priority,
		Task:           models.InferenceQueueMessageTaskImageGeneration,
		Timestamp:      time.Now(),
		MemoryRequired: util.Gb2b(10), // Convert GB to bytes
	}

	// Publish the request to RabbitMQ
	if err := client.PublishRequest(req); err != nil {
		handlerReg.deregister(requestID)
		return "", err
	}

	return requestID, nil
}
