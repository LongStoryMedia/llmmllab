package mq

import (
	"maistro/models"
	"maistro/util"
	"time"

	"github.com/sirupsen/logrus"
)

// SubmitImageEditRequest sends an image editing request to RabbitMQ
func SubmitImageEditRequest(
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

	// Create the request message
	reqType := "request"
	request := models.InferenceQueueMessage{
		CorrelationID: string(requestID),
		Timestamp:     time.Now().UTC(),
		Priority:      priority,
		Type:          &reqType,
		Task:          models.InferenceQueueMessageTaskImageEditing,
		Payload:       image,
	}

	// Log the request
	util.LogInfo("Submitting image editing request to RabbitMQ", logrus.Fields{
		"requestId": requestID,
		"userId":    userID,
	})

	// Publish the request to RabbitMQ
	err := client.PublishRequest(request)
	if err != nil {
		util.LogWarning("Failed to publish image editing request to RabbitMQ", logrus.Fields{
			"requestId": requestID,
			"error":     err.Error(),
		})

		// Clean up the handler on error
		handlerReg.deregister(requestID)

		return "", err
	}

	return requestID, nil
}
