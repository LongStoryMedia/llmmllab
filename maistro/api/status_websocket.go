package api

import (
	"encoding/json"
	"log"
	"maistro/models"
	"sync"

	"github.com/gofiber/contrib/websocket"
)

var (
	// statusConnections stores WebSocket connections by userID
	statusConnections = make(map[string][]*websocket.Conn)
	statusMutex       sync.RWMutex
)

// HandleStatusWebSocket manages WebSocket connections for status updates
func HandleStatusWebSocket(conn *websocket.Conn, userID string) {
	// Register the connection
	registerStatusConnection(userID, conn)
	defer func() {
		unregisterStatusConnection(userID, conn)
		conn.Close()
	}()

	// Keep the connection alive until client disconnects
	for {
		// Read message (just to detect disconnection)
		_, _, err := conn.ReadMessage()
		if err != nil {
			break
		}
	}
}

// registerStatusConnection adds a new WebSocket connection to the user's connection list
func registerStatusConnection(userID string, conn *websocket.Conn) {
	statusMutex.Lock()
	defer statusMutex.Unlock()

	statusConnections[userID] = append(statusConnections[userID], conn)
	log.Printf("Status WebSocket registered for user %s", userID)
}

// unregisterStatusConnection removes a WebSocket connection from the user's connection list
func unregisterStatusConnection(userID string, conn *websocket.Conn) {
	statusMutex.Lock()
	defer statusMutex.Unlock()

	connections := statusConnections[userID]
	for i, c := range connections {
		if c == conn {
			statusConnections[userID] = append(connections[:i], connections[i+1:]...)
			break
		}
	}

	// If no more connections for this user, remove the user entry
	if len(statusConnections[userID]) == 0 {
		delete(statusConnections, userID)
	}

	log.Printf("Status WebSocket unregistered for user %s", userID)
}

// SendStatusUpdate sends a status update to all WebSocket connections for a user
func SendStatusUpdate(userID string, updateType models.StatusUpdateType, stage models.StatusUpdateStage, message string, progress int, done bool) {
	statusMutex.RLock()
	connections := statusConnections[userID]
	statusMutex.RUnlock()

	update := models.StatusUpdate{
		UserID:   userID,
		Type:     updateType,
		Stage:    stage,
		Message:  &message,
		Progress: progress,
		Done:     done,
	}

	payload, err := json.Marshal(update)
	if err != nil {
		log.Printf("Error marshaling status update: %v", err)
		return
	}

	for _, conn := range connections {
		if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
			log.Printf("Error sending status update to WebSocket: %v", err)
			// Continue trying to send to other connections
		}
	}
}
