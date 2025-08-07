// Deprecated: Use svc.SocketService package instead
package socket

// import (
// 	"encoding/json"
// 	"fmt"
// 	"maistro/models"
// 	"maistro/util"

// 	"github.com/gofiber/contrib/websocket"
// 	"github.com/sirupsen/logrus"
// )

// // // Message represents a message received from a WebSocket client
// // type Message struct {
// // 	Type           string          `json:"type"`
// // 	ConversationID string          `json:"conversation_id,omitempty"`
// // 	Content        json.RawMessage `json:"content"`
// // }

// // Deprecated: Use svc.SocketService package instead
// func handleChatWebSocket(c *websocket.Conn) {
// 	// Get user ID from context
// 	userID := GetUserIDFromSocketConnection(c)
// 	if userID == "" {
// 		util.HandleError(fmt.Errorf("WebSocket connection rejected: missing user ID"))
// 		c.Close()
// 		return
// 	}

// 	// Register connection
// 	connID := registerConnection(userID, models.WebSocketConnectionTypeChat, c)
// 	defer unregisterConnection(userID, connID)

// 	util.LogInfo("Chat WebSocket connection established", logrus.Fields{
// 		"userID": userID,
// 		"connID": connID,
// 	})

// 	// Set ping handler to keep connection alive
// 	c.SetPingHandler(func(appData string) error {
// 		return c.WriteMessage(websocket.PongMessage, []byte{})
// 	})

// 	// Main message handling loop
// 	for {
// 		// Read message from client
// 		messageType, msg, err := c.ReadMessage()
// 		if err != nil {
// 			if websocket.IsUnexpectedCloseError(err) {
// 				util.LogInfo("Chat WebSocket connection closed", logrus.Fields{
// 					"userID": userID,
// 					"connID": connID,
// 				})
// 			} else {
// 				util.HandleError(err)
// 			}
// 			break
// 		}

// 		// Handle ping messages
// 		if messageType == websocket.PingMessage {
// 			if err := c.WriteMessage(websocket.PongMessage, nil); err != nil {
// 				util.HandleError(err)
// 				break
// 			}
// 			continue
// 		}

// 		// Process message
// 		var chatMessage models.SocketMessage
// 		if err := json.Unmarshal(msg, &chatMessage); err != nil {
// 			util.HandleError(err)
// 			SendError(models.SocketStageTypeInitializing, -1, userID, "Invalid message format")
// 			continue
// 		}

// 		if chatMessage.Data == nil {
// 			util.LogWarning("Chat message missing data", logrus.Fields{
// 				"userID":  userID,
// 				"message": chatMessage,
// 			})
// 			SendError(models.SocketStageTypeInitializing, -1, userID, "Message data is required")
// 			continue
// 		}

// 		for _, data := range chatMessage.Data {
// 			if chMsg, ok := data.(models.Message); !ok {
// 				util.LogWarning("Chat message data is not of type Message", logrus.Fields{
// 					"userID": userID,
// 					"data":   chatMessage.Data,
// 				})
// 				SendError(models.SocketStageTypeInitializing, -1, userID, "Invalid message data type")
// 				continue
// 			} else {

// 				// Handle message based on type
// 				switch chatMessage.Type {
// 				case models.MessageTypePause:
// 					handlePauseRequest(userID, chMsg.ConversationID)
// 				case models.MessageTypeResume:
// 					handleResumeRequest(userID, chMsg.ConversationID)
// 				case models.MessageTypeCancel:
// 					handleCancelRequest(userID, chMsg.ConversationID)
// 				default:
// 					SendError(models.SocketStageTypeInitializing, -1, userID, fmt.Sprintf("Unknown message type: %s", chatMessage.Type))
// 				}
// 			}
// 		}
// 	}
// }

// // handlePauseRequest processes a request to pause generation
// func handlePauseRequest(userID string, conversationID int) {
// 	if conversationID <= 0 {
// 		util.LogWarning("Pause request missing conversation ID", nil)
// 		return
// 	}

// 	util.LogInfo("Pause request received", logrus.Fields{
// 		"userID":         userID,
// 		"conversationID": conversationID,
// 	})

// 	// Signal to pause the conversation
// 	// This will be implemented by the proxy/context package
// 	// PauseConversationGeneration(conversationID)
// }

// // handleResumeRequest processes a request to resume generation
// func handleResumeRequest(userID string, conversationID int) {
// 	if conversationID <= 0 {
// 		util.LogWarning("Resume request missing conversation ID", nil)
// 		return
// 	}

// 	util.LogInfo("Resume request received", logrus.Fields{
// 		"userID":         userID,
// 		"conversationID": conversationID,
// 	})

// 	// Signal to resume the conversation
// 	// This will be implemented by the proxy/context package
// 	// ResumeConversationGeneration(conversationID)
// }

// // handleCancelRequest processes a request to cancel generation
// func handleCancelRequest(userID string, conversationID int) {
// 	if conversationID <= 0 {
// 		util.LogWarning("Cancel request missing conversation ID", nil)
// 		return
// 	}

// 	util.LogInfo("Cancel request received", logrus.Fields{
// 		"userID":         userID,
// 		"conversationID": conversationID,
// 	})

// 	// Signal to cancel the conversation
// 	// This will be implemented by the proxy/context package
// 	// CancelConversationGeneration(conversationID)
// }
