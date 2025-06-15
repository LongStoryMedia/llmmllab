package socket

import (
	"context"
	"maistro/models"
	"maistro/util"
	"sync"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// connectionRegistry stores active WebSocket connections
var connectionRegistry = struct {
	sync.RWMutex
	byUser map[string]map[string]*models.WebSocketConnection                         // userID -> connID -> conn
	byType map[models.WebSocketConnectionType]map[string]*models.WebSocketConnection // type -> connID -> conn
	all    map[string]*models.WebSocketConnection                                    // connID -> conn
}{
	byUser: make(map[string]map[string]*models.WebSocketConnection),
	byType: map[models.WebSocketConnectionType]map[string]*models.WebSocketConnection{
		models.WebSocketConnectionTypeChat:   make(map[string]*models.WebSocketConnection),
		models.WebSocketConnectionTypeImage:  make(map[string]*models.WebSocketConnection),
		models.WebSocketConnectionTypeStatus: make(map[string]*models.WebSocketConnection),
	},
	all: make(map[string]*models.WebSocketConnection),
}

// SetupWebSocketRoutes configures WebSocket routes for the application
func SetupWebSocketRoutes(app *fiber.App) {
	// WebSocket configuration
	wsConfig := websocket.Config{
		HandshakeTimeout: 10 * time.Second,
	}
	// Create a route group for WebSocket endpoints
	wsGroup := app.Group("/ws")

	// Add authentication middleware specific for WebSockets
	wsGroup.Use(websocketAuthMiddleware())

	// Chat WebSocket route
	wsGroup.Use("/chat", websocket.New(func(c *websocket.Conn) {
		handleChatWebSocket(c)
	}, wsConfig))

	// Image WebSocket route
	wsGroup.Use("/image", websocket.New(func(c *websocket.Conn) {
		handleImageWebSocket(c)
	}, wsConfig))

	// Status WebSocket route
	wsGroup.Use("/status", websocket.New(func(c *websocket.Conn) {
		handleStatusWebSocket(c)
	}, wsConfig))

	util.LogInfo("WebSocket routes initialized", nil)
}

// websocketAuthMiddleware validates the token for WebSocket connections
func websocketAuthMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Check for token in query parameters
		token := c.Query("token")
		if token == "" {
			return fiber.ErrUnauthorized
		}

		// Validate the token and get user ID
		// TODO: Use auth package to validate the token
		// For now, we'll just use the token as the user ID
		userID := token
		c.Locals("userID", userID)

		// Only allow WebSocket upgrade requests
		if websocket.IsWebSocketUpgrade(c) {
			return c.Next()
		}

		return fiber.ErrUpgradeRequired
	}
}

// getUserIDFromContext extracts the user ID from the WebSocket context
func getUserIDFromContext(c *websocket.Conn) string {
	userID, ok := c.Locals("userID").(string)
	if !ok {
		// Extract user ID from headers, query params, or JWT
		userID := c.Query("user_id")
		if userID == "" {
			// Try to get from headers
			userID = c.Headers("X-User-ID")
		}
	}

	// For development, assign a default user ID if none is provided
	// if userID == "" && util.IsDevelopment() {
	// 	userID = "dev-user"
	// 	util.LogWarning("Using default dev-user ID for WebSocket connection", nil)
	// }

	return userID
}

// registerConnection adds a new connection to the registry
func registerConnection(userID string, connType models.WebSocketConnectionType, conn *websocket.Conn) string {
	connectionRegistry.Lock()
	defer connectionRegistry.Unlock()

	// Generate a connection ID
	connID := uuid.New().String()

	// Create WebSocket connection object
	wsConn := &models.WebSocketConnection{
		ID:        connID,
		UserID:    userID,
		Type:      connType,
		Conn:      conn,
		CreatedAt: time.Now(),
	}

	// Add to user map
	if _, exists := connectionRegistry.byUser[userID]; !exists {
		connectionRegistry.byUser[userID] = make(map[string]*models.WebSocketConnection)
	}
	connectionRegistry.byUser[userID][connID] = wsConn

	// Add to type map
	connectionRegistry.byType[connType][connID] = wsConn

	// Add to global map
	connectionRegistry.all[connID] = wsConn

	util.LogInfo("WebSocket connection registered", logrus.Fields{
		"userID":     userID,
		"connType":   connType,
		"connID":     connID,
		"totalConns": len(connectionRegistry.all),
	})

	return connID
}

// unregisterConnection removes a connection from the registry
func unregisterConnection(userID string, connID string) {
	connectionRegistry.Lock()
	defer connectionRegistry.Unlock()

	// Get connection details before removal
	wsConn, exists := connectionRegistry.all[connID]
	if !exists {
		return
	}

	// Remove from all maps
	delete(connectionRegistry.byUser[userID], connID)
	delete(connectionRegistry.byType[wsConn.Type], connID)
	delete(connectionRegistry.all, connID)

	// Clean up user map if empty
	if len(connectionRegistry.byUser[userID]) == 0 {
		delete(connectionRegistry.byUser, userID)
	}

	util.LogInfo("WebSocket connection unregistered", logrus.Fields{
		"userID":     userID,
		"connType":   wsConn.Type,
		"connID":     connID,
		"totalConns": len(connectionRegistry.all),
	})
}

// broadcastToUser sends a message to all connections for a specific user
func broadcastToUser(userID string, message interface{}) {
	connectionRegistry.RLock()
	userConns, exists := connectionRegistry.byUser[userID]
	if !exists || len(userConns) == 0 {
		connectionRegistry.RUnlock()
		return
	}

	// Make a copy of connections to avoid holding the lock during writes
	connsCopy := make([]*models.WebSocketConnection, 0, len(userConns))
	for _, conn := range userConns {
		connsCopy = append(connsCopy, conn)
	}
	connectionRegistry.RUnlock()

	// Send to all connections
	for _, wsConn := range connsCopy {
		c := wsConn.Conn.(*websocket.Conn)

		if c != nil {
			err := c.WriteJSON(message)
			if err != nil {
				util.HandleError(err)
				// Don't remove connection here, let ping/pong or read failure handle it
			}
		}
	}
}

// broadcastToType sends a message to all connections of a specific type
func broadcastToType(connType models.WebSocketConnectionType, message interface{}) {
	connectionRegistry.RLock()
	typeConns := connectionRegistry.byType[connType]
	if len(typeConns) == 0 {
		connectionRegistry.RUnlock()
		return
	}

	// Make a copy of connections to avoid holding the lock during writes
	connsCopy := make([]*models.WebSocketConnection, 0, len(typeConns))
	for _, conn := range typeConns {
		connsCopy = append(connsCopy, conn)
	}
	connectionRegistry.RUnlock()

	// Send to all connections
	for _, wsConn := range connsCopy {
		c := wsConn.Conn.(*websocket.Conn)

		if c != nil {
			err := c.WriteJSON(message)
			if err != nil {
				util.HandleError(err)
				// Don't remove connection here, let ping/pong or read failure handle it
			}
		}
	}
}

// broadcastToAll sends a message to all active WebSocket connections
func broadcastToAll(message interface{}) {
	connectionRegistry.RLock()
	conns := connectionRegistry.all
	if len(conns) == 0 {
		connectionRegistry.RUnlock()
		return
	}

	// Make a copy of connections
	connsCopy := make([]*models.WebSocketConnection, 0, len(conns))
	for _, conn := range conns {
		connsCopy = append(connsCopy, conn)
	}
	connectionRegistry.RUnlock()

	// Send to all connections
	for _, wsConn := range connsCopy {
		c := wsConn.Conn.(*websocket.Conn)

		if c != nil {
			err := c.WriteJSON(message)
			if err != nil {
				util.HandleError(err)
				// Don't remove connection here, let ping/pong or read failure handle it
			}
		}
	}
}

// RunBackgroundCleanup starts a periodic job to clean up closed connections
func RunBackgroundCleanup(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cleanupInactiveConnections()
		}
	}
}

// cleanupInactiveConnections removes closed or inactive connections
func cleanupInactiveConnections() {
	connectionRegistry.Lock()
	defer connectionRegistry.Unlock()

	// Find connections to remove
	toRemove := make([]string, 0)
	for id, conn := range connectionRegistry.all {
		// Check if connection is closed
		if conn.Conn.(*websocket.Conn) == nil {
			toRemove = append(toRemove, id)
		}
	}

	// Remove connections
	for _, id := range toRemove {
		conn := connectionRegistry.all[id]
		userID := conn.UserID
		connType := conn.Type

		// Remove from all maps
		delete(connectionRegistry.byUser[userID], id)
		delete(connectionRegistry.byType[connType], id)
		delete(connectionRegistry.all, id)

		// Clean up user map if empty
		if len(connectionRegistry.byUser[userID]) == 0 {
			delete(connectionRegistry.byUser, userID)
		}
	}

	if len(toRemove) > 0 {
		util.LogInfo("Cleaned up inactive WebSocket connections", logrus.Fields{
			"removed":   len(toRemove),
			"remaining": len(connectionRegistry.all),
		})
	}
}
