// Deprecated: Use svc.SocketService package instead
package socket

// import (
// 	"maistro/models"
// 	"maistro/util"
// 	"time"

// 	"github.com/gofiber/contrib/websocket"
// 	"github.com/google/uuid"
// )

// // handleImageWebSocket manages WebSocket connections for image generation notifications
// func handleImageWebSocket(c *websocket.Conn) {
// 	userID := GetUserIDFromSocketConnection(c)
// 	if userID == "" {
// 		util.LogWarning("User ID not found in WebSocket context", nil)
// 		return
// 	}

// 	// Register connection
// 	connID := registerConnection(userID, models.WebSocketConnectionTypeImage, c)

// 	if err := c.WriteJSON(models.SocketMessage{
// 		ID:        uuid.New().String(),
// 		Type:      models.MessageTypeConnected,
// 		Content:   util.StrPtr("Connected to image generation WebSocket"),
// 		State:     models.SocketStageTypeOpen,
// 		Timestamp: time.Now(),
// 	}); err != nil {
// 		util.HandleError(err)
// 	}

// 	// Keep the connection open and handle incoming messages (mostly just for close detection)
// 	for {
// 		_, _, err := c.ReadMessage()
// 		if err != nil {
// 			break
// 		}
// 	}

// 	// Unregister when connection closes
// 	unregisterConnection(userID, connID)
// }
