package socket

import (
	"fmt"
	"maistro/models"
	"maistro/util"

	"github.com/gofiber/contrib/websocket"
	"github.com/sirupsen/logrus"
)

// StatusUpdate represents a status update message for WebSocket clients
// type StatusUpdate struct {
// 	Type      string `json:"type"`
// 	Stage     string `json:"stage"`
// 	Message   string `json:"message,omitempty"`
// 	Progress  int    `json:"progress"`
// 	Timestamp int64  `json:"timestamp"`
// 	IsComplete bool  `json:"is_complete"`
// }

// handleStatusWebSocket handles WebSocket connections for status updates
func handleStatusWebSocket(c *websocket.Conn) {
	// Get user ID from context
	userID := getUserIDFromContext(c)
	if userID == "" {
		util.HandleError(fmt.Errorf("WebSocket connection rejected: missing user ID"))
		c.Close()
		return
	}

	// Register connection
	connID := registerConnection(userID, models.WebSocketConnectionTypeStatus, c)
	defer unregisterConnection(userID, connID)

	util.LogInfo("Status WebSocket connection established", logrus.Fields{
		"userID": userID,
		"connID": connID,
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
					"userID": userID,
					"connID": connID,
				})
			} else {
				util.HandleError(err)
			}
			break
		}
	}
}

// SendStatusUpdate sends a status update to a specific user
func SendStatusUpdate(userID string, statusType models.StatusUpdateType, stage models.StatusUpdateStage, message string, progress int, isComplete bool) {
	update := models.StatusUpdate{
		Type:     statusType,
		Stage:    stage,
		Message:  &message,
		Progress: progress,
		Done:     isComplete,
		UserID:   userID,
	}

	// Broadcast to user
	broadcastToUser(userID, update)
}

// BroadcastProcessingStage sends a processing stage status update to a specific user
func BroadcastProcessingStage(userID string, stage models.StatusUpdateStage, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, stage, "", progress, false)
}

// BroadcastError sends an error status update to a specific user
func BroadcastError(userID string, message string) {
	SendStatusUpdate(userID, models.StatusUpdateTypeError, models.StatusUpdateStageError, message, 0, false)
}

// BroadcastCompletion sends a completion status update to a specific user
func BroadcastCompletion(userID string, message string) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageCompleted, message, 100, true)
}

// BroadcastMemoryStage sends a memory retrieval status update
func BroadcastMemoryStage(userID string, message string, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageRetrievingMemories, message, progress, false)
}

// BroadcastSearchStage sends a web search status update
func BroadcastSearchStage(userID string, message string, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageSearchingWeb, message, progress, false)
}

// BroadcastSummarizingStage sends a summarization status update
func BroadcastSummarizingStage(userID string, message string, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageSummarizing, message, progress, false)
}
