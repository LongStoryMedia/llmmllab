package socket

import (
	"encoding/json"
	"fmt"
	"maistro/models"
	"maistro/util"

	"github.com/gofiber/contrib/websocket"
	"github.com/sirupsen/logrus"
)

// // ChatMessage represents a message received from a WebSocket client
// type ChatMessage struct {
// 	Type           string          `json:"type"`
// 	ConversationID string          `json:"conversation_id,omitempty"`
// 	Content        json.RawMessage `json:"content"`
// }

// handleChatWebSocket handles WebSocket connections for chat
func handleChatWebSocket(c *websocket.Conn) {
	// Get user ID from context
	userID := getUserIDFromContext(c)
	if userID == "" {
		util.HandleError(fmt.Errorf("WebSocket connection rejected: missing user ID"))
		c.Close()
		return
	}

	// Register connection
	connID := registerConnection(userID, models.WebSocketConnectionTypeChat, c)
	defer unregisterConnection(userID, connID)

	util.LogInfo("Chat WebSocket connection established", logrus.Fields{
		"userID": userID,
		"connID": connID,
	})

	// Set ping handler to keep connection alive
	c.SetPingHandler(func(appData string) error {
		return c.WriteMessage(websocket.PongMessage, []byte{})
	})

	// Main message handling loop
	for {
		// Read message from client
		messageType, msg, err := c.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err) {
				util.LogInfo("Chat WebSocket connection closed", logrus.Fields{
					"userID": userID,
					"connID": connID,
				})
			} else {
				util.HandleError(err)
			}
			break
		}

		// Handle ping messages
		if messageType == websocket.PingMessage {
			if err := c.WriteMessage(websocket.PongMessage, nil); err != nil {
				util.HandleError(err)
				break
			}
			continue
		}

		// Process message
		var chatMessage models.ChatMessage
		if err := json.Unmarshal(msg, &chatMessage); err != nil {
			util.HandleError(err)
			sendErrorMessage(c, "Invalid message format")
			continue
		}

		// Handle message based on type
		switch chatMessage.Metadata.Type {
		case models.ChatMessageMetadataTypePause:
			handlePauseRequest(userID, chatMessage.ConversationID)
		case models.ChatMessageMetadataTypeResume:
			handleResumeRequest(userID, chatMessage.ConversationID)
		case models.ChatMessageMetadataTypeCancel:
			handleCancelRequest(userID, chatMessage.ConversationID)
		case models.ChatMessageMetadataTypeChat:
			continue
		case models.ChatMessageMetadataTypeImage:
			continue
		default:
			sendErrorMessage(c, "Unknown message type")
		}
	}
}

// handlePauseRequest processes a request to pause generation
func handlePauseRequest(userID string, conversationID int) {
	if conversationID <= 0 {
		util.LogWarning("Pause request missing conversation ID", nil)
		return
	}

	util.LogInfo("Pause request received", logrus.Fields{
		"userID":         userID,
		"conversationID": conversationID,
	})

	// Signal to pause the conversation
	// This will be implemented by the proxy/context package
	// PauseConversationGeneration(conversationID)
}

// handleResumeRequest processes a request to resume generation
func handleResumeRequest(userID string, conversationID int) {
	if conversationID <= 0 {
		util.LogWarning("Resume request missing conversation ID", nil)
		return
	}

	util.LogInfo("Resume request received", logrus.Fields{
		"userID":         userID,
		"conversationID": conversationID,
	})

	// Signal to resume the conversation
	// This will be implemented by the proxy/context package
	// ResumeConversationGeneration(conversationID)
}

// handleCancelRequest processes a request to cancel generation
func handleCancelRequest(userID string, conversationID int) {
	if conversationID <= 0 {
		util.LogWarning("Cancel request missing conversation ID", nil)
		return
	}

	util.LogInfo("Cancel request received", logrus.Fields{
		"userID":         userID,
		"conversationID": conversationID,
	})

	// Signal to cancel the conversation
	// This will be implemented by the proxy/context package
	// CancelConversationGeneration(conversationID)
}

// sendErrorMessage sends an error message to the WebSocket client
func sendErrorMessage(conn *websocket.Conn, message string) {
	errorMsg := map[string]string{
		"type":    "error",
		"message": message,
	}

	if err := conn.WriteJSON(errorMsg); err != nil {
		util.HandleError(err)
	}
}
