package socket

import (
	"maistro/models"
	"maistro/util"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/sirupsen/logrus"
)

// ImageGenerationNotification represents a notification about image generation status
type ImageGenerationNotification struct {
	Type      string                `json:"type"`
	Status    string                `json:"status"`
	ImageID   string                `json:"image_id"`
	ImageURL  string                `json:"image_url,omitempty"`
	Prompt    string                `json:"prompt,omitempty"`
	Timestamp int64                 `json:"timestamp"`
	Metadata  *models.ImageMetadata `json:"metadata,omitempty"`
}

// ImageGenStatus represents the possible statuses for image generation
const (
	ImageGenStatusQueued     = "queued"
	ImageGenStatusGenerating = "generating"
	ImageGenStatusComplete   = "complete"
	ImageGenStatusError      = "error"
)

// handleImageWebSocket manages WebSocket connections for image generation notifications
func handleImageWebSocket(c *websocket.Conn) {
	userID := getUserIDFromContext(c)
	if userID == "" {
		util.LogWarning("User ID not found in WebSocket context", nil)
		return
	}

	// Register connection
	connID := registerConnection(userID, "image", c)

	// Send initial connected status
	notification := ImageGenerationNotification{
		Type:      "connected",
		Status:    "Connected to image generation WebSocket",
		Timestamp: time.Now().Unix(),
	}

	if err := c.WriteJSON(notification); err != nil {
		util.HandleError(err)
	}

	// Keep the connection open and handle incoming messages (mostly just for close detection)
	for {
		_, _, err := c.ReadMessage()
		if err != nil {
			break
		}
	}

	// Unregister when connection closes
	unregisterConnection(userID, connID)
}

// NotifyImageQueued sends a notification that an image generation has been queued
func NotifyImageQueued(userID string, imageID string, prompt string, metadata *models.ImageMetadata) {
	notification := ImageGenerationNotification{
		Type:      "image_update",
		Status:    ImageGenStatusQueued,
		ImageID:   imageID,
		Prompt:    prompt,
		Timestamp: time.Now().Unix(),
		Metadata:  metadata,
	}

	broadcastToUser(userID, notification)

	util.LogInfo("Image generation queued notification sent", logrus.Fields{
		"userID":  userID,
		"imageID": imageID,
	})
}

// NotifyImageGenerating sends a notification that an image is being generated
func NotifyImageGenerating(userID string, imageID string, prompt string) {
	notification := ImageGenerationNotification{
		Type:      "image_update",
		Status:    ImageGenStatusGenerating,
		ImageID:   imageID,
		Prompt:    prompt,
		Timestamp: time.Now().Unix(),
	}

	broadcastToUser(userID, notification)

	util.LogInfo("Image generation in progress notification sent", logrus.Fields{
		"userID":  userID,
		"imageID": imageID,
	})
}

// NotifyImageGenerated sends a notification that an image has been generated
func NotifyImageGenerated(userID string, imageID string, imageURL string, metadata *models.ImageMetadata) {
	notification := ImageGenerationNotification{
		Type:      "image_update",
		Status:    ImageGenStatusComplete,
		ImageID:   imageID,
		ImageURL:  imageURL,
		Timestamp: time.Now().Unix(),
		Metadata:  metadata,
	}

	broadcastToUser(userID, notification)

	util.LogInfo("Image generation completion notification sent", logrus.Fields{
		"userID":   userID,
		"imageID":  imageID,
		"imageURL": imageURL,
	})
}

// NotifyImageError sends a notification that there was an error generating an image
func NotifyImageError(userID string, imageID string, errorMsg string) {
	notification := ImageGenerationNotification{
		Type:      "image_update",
		Status:    ImageGenStatusError,
		ImageID:   imageID,
		Prompt:    errorMsg, // Use prompt field to send error message
		Timestamp: time.Now().Unix(),
	}

	broadcastToUser(userID, notification)

	util.LogWarning("Image generation error notification sent", logrus.Fields{
		"userID":  userID,
		"imageID": imageID,
		"error":   errorMsg,
	})
}
