// Deprecated: Use svc.SocketService package instead
package socket

// import (
// 	"errors"
// 	"fmt"
// 	"maistro/models"
// 	"maistro/util"
// 	"time"

// 	"github.com/gofiber/contrib/websocket"
// 	"github.com/google/uuid"
// 	"github.com/sirupsen/logrus"
// )

// // handleStatusWebSocket handles WebSocket connections for status updates
// func handleStatusWebSocket(c *websocket.Conn) {
// 	// Get user ID from context
// 	userID := GetUserIDFromSocketConnection(c)
// 	if userID == "" {
// 		util.HandleError(fmt.Errorf("WebSocket connection rejected: missing user ID"))
// 		c.Close()
// 		return
// 	}

// 	// Register connection
// 	connID := registerConnection(userID, models.WebSocketConnectionTypeStatus, c)
// 	defer unregisterConnection(userID, connID)

// 	util.LogInfo("Status WebSocket connection established", logrus.Fields{
// 		"userID": userID,
// 		"connID": connID,
// 	})

// 	// Set ping handler to keep connection alive
// 	c.SetPingHandler(func(appData string) error {
// 		return c.WriteMessage(websocket.PongMessage, []byte{})
// 	})

// 	// Keep the connection open until the client disconnects
// 	for {
// 		// Read message from client
// 		_, _, err := c.ReadMessage()
// 		if err != nil {
// 			if websocket.IsUnexpectedCloseError(err) {
// 				util.LogInfo("Status WebSocket connection closed", logrus.Fields{
// 					"userID": userID,
// 					"connID": connID,
// 				})
// 			} else {
// 				util.HandleError(err)
// 			}
// 			break
// 		}
// 	}
// }

// func sanitizePayload[T any](payload ...T) []interface{} {
// 	pld := make([]models.ImageGenerationNotification, 0)
// 	if len(payload) > 0 {
// 		for _, p := range payload {
// 			switch v := any(p).(type) {
// 			case models.ImageGenerationNotification:
// 				pld = append(pld, v)
// 			default:
// 				util.LogWarningAtCallLevel("Invalid payload type for image generation notification", 2, logrus.Fields{
// 					"payloadType":  fmt.Sprintf("%T", p),
// 					"expectedType": "models.ImageGenerationNotification",
// 				})
// 				continue
// 			}
// 		}
// 	}
// 	// Convert []T to []interface{}
// 	data := make([]interface{}, len(pld))
// 	for i, v := range pld {
// 		data[i] = v
// 	}

// 	return data
// }

// // Deprecated: Use SocketService package instead
// func SendInfo(stage models.SocketStageType, conversationID int, userID string, message string, progress int) {
// 	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
// 		"stage":          stage,
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 		"progress":       progress,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeInfo,
// 		Content:   &message,
// 		State:     stage,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 		Progress:  &progress,
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendWarning(stage models.SocketStageType, conversationID int, userID string, message string) {
// 	util.LogWarningAtCallLevel(message, 2, logrus.Fields{
// 		"stage":          stage,
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeWarning,
// 		Content:   &message,
// 		State:     stage,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendError(stage models.SocketStageType, conversationID int, userID string, message string) {
// 	util.HandleErrorAtCallLevel(errors.New(message), 2, logrus.Fields{
// 		"stage":          stage,
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeError,
// 		Content:   &message,
// 		State:     stage,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendCompletion[T any](stage models.SocketStageType, conversationID int, userID string, message string, payload ...T) {
// 	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
// 		"stage":          stage,
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeComplete,
// 		Content:   &message,
// 		State:     stage,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 		Data:      sanitizePayload(payload...),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendCancelled(conversationID int, userID string) {
// 	message := "Session cancelled"
// 	util.LogWarningAtCallLevel(message, 2, logrus.Fields{
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeCancelled,
// 		Content:   &message,
// 		State:     models.SocketStageTypeProcessing,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendPaused(conversationID int, userID string) {
// 	message := "Session paused"
// 	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypePaused,
// 		Content:   &message,
// 		State:     models.SocketStageTypeProcessing,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendResumed(conversationID int, userID string) {
// 	message := "Session resumed"
// 	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeResumed,
// 		Content:   &message,
// 		State:     models.SocketStageTypeProcessing,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }

// // Deprecated: Use SocketService package instead
// func SendConnected(stage models.SocketStageType, conversationID int, userID string, message string) {
// 	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
// 		"stage":          stage,
// 		"conversationID": conversationID,
// 		"userID":         userID,
// 	})
// 	csr := models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeConnected,
// 		Content:   &message,
// 		State:     stage,
// 		SessionID: GetChatSocketSessionID(userID, conversationID),
// 		Timestamp: time.Now(),
// 	}

// 	// Broadcast to user
// 	BroadcastToUser(userID, csr)
// }
