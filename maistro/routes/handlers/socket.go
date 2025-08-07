package handlers

import (
	"encoding/json"
	"fmt"
	"maistro/middleware"
	"maistro/models"
	svc "maistro/services"
	"maistro/util"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

func HandleChatSocket(c *websocket.Conn) {
	uidp := c.Locals(string(middleware.UIDPKey))
	if uidp == nil {
		util.LogWarning("Status WebSocket connection missing user ID")
		c.Close()
		return
	}
	uid, ok := uidp.(string)
	if !ok || uid == "" {
		util.LogWarning("Status WebSocket connection has invalid user ID", logrus.Fields{
			"uid": uidp,
		})
		c.Close()
		return
	}
	s := svc.GetSocketService()

	util.LogDebug("New chat WebSocket connection established", logrus.Fields{
		"userID": uid,
	})
	// Register connection
	s.RegisterConnection(uid, models.WebSocketConnectionTypeChat, c)
	defer s.UnregisterConnection(uid, models.WebSocketConnectionTypeChat)

	util.LogInfo("Chat WebSocket connection established", logrus.Fields{
		"userID": uid,
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
					"userID": uid,
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
		var chatMessage models.SocketMessage
		if err := json.Unmarshal(msg, &chatMessage); err != nil {
			util.HandleError(err)
			s.SendError(models.SocketStageTypeInitializing, -1, uid, "Invalid message format")
			continue
		}

		if chatMessage.Data == nil {
			util.LogWarning("Chat message missing data", logrus.Fields{
				"userID":  uid,
				"message": chatMessage,
			})
			s.SendError(models.SocketStageTypeInitializing, -1, uid, "Message data is required")
			continue
		}

		for _, data := range chatMessage.Data {
			if chMsg, ok := data.(models.Message); !ok {
				util.LogWarning("Chat message data is not of type Message", logrus.Fields{
					"userID": uid,
					"data":   chatMessage.Data,
				})
				s.SendError(models.SocketStageTypeInitializing, -1, uid, "Invalid message data type")
				continue
			} else {

				// Handle message based on type
				switch chatMessage.Type {
				case models.MessageTypePause:
					handlePauseRequest(uid, chMsg.ConversationID)
				case models.MessageTypeResume:
					handleResumeRequest(uid, chMsg.ConversationID)
				case models.MessageTypeCancel:
					handleCancelRequest(uid, chMsg.ConversationID)
				default:
					s.SendError(models.SocketStageTypeInitializing, -1, uid, fmt.Sprintf("Unknown message type: %s", chatMessage.Type))
				}
			}
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

	svc.GetSocketService().PauseAndBroadcast(conversationID, userID)
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

	svc.GetSocketService().ResumeAndBroadcast(conversationID, userID)
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

	svc.GetSocketService().CancelAndBroadcast(conversationID, userID)
}

// HandleImageSocket manages WebSocket connections for image generation notifications
func HandleImageSocket(c *websocket.Conn) {
	uidp := c.Locals(string(middleware.UIDPKey))
	if uidp == nil {
		util.LogWarning("Status WebSocket connection missing user ID")
		c.Close()
		return
	}
	uid, ok := uidp.(string)
	if !ok || uid == "" {
		util.LogWarning("Status WebSocket connection has invalid user ID", logrus.Fields{
			"uid": uidp,
		})
		c.Close()
		return
	}
	s := svc.GetSocketService()
	// Register connection
	s.RegisterConnection(uid, models.WebSocketConnectionTypeImage, c)
	// Unregister when connection closes
	defer s.UnregisterConnection(uid, models.WebSocketConnectionTypeImage)

	if err := c.WriteJSON(models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeConnected,
		Content:   util.StrPtr("Connected to image generation WebSocket"),
		State:     models.SocketStageTypeOpen,
		Timestamp: time.Now(),
	}); err != nil {
		util.HandleError(err)
	}

	// Keep the connection open and handle incoming messages (mostly just for close detection)
	for {
		_, _, err := c.ReadMessage()
		if err != nil {
			break
		}
	}
}

// HandleStatusSocket manages WebSocket connections for status updates
func HandleStatusSocket(c *websocket.Conn) {
	uidp := c.Locals(string(middleware.UIDPKey))
	if uidp == nil {
		util.LogWarning("Status WebSocket connection missing user ID")
		c.Close()
		return
	}
	uid, ok := uidp.(string)
	if !ok || uid == "" {
		util.LogWarning("Status WebSocket connection has invalid user ID", logrus.Fields{
			"uid": uidp,
		})
		c.Close()
		return
	}

	s := svc.GetSocketService()
	// Register connection
	s.RegisterConnection(uid, models.WebSocketConnectionTypeStatus, c)
	defer s.UnregisterConnection(uid, models.WebSocketConnectionTypeStatus)

	util.LogInfo("Status WebSocket connection established", logrus.Fields{
		"userID": uid,
	})

	// Set ping handler to keep connection alive
	c.SetPingHandler(func(appData string) error {
		return c.WriteMessage(websocket.PongMessage, []byte{})
	})

	// Keep the connection open until the client disconnects
	for {
		// Read message from client
		_, _, err := c.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err) {
				util.LogInfo("Status WebSocket connection closed", logrus.Fields{
					"userID": uid,
				})
			} else {
				util.HandleError(err)
			}
			break
		}
	}
}
