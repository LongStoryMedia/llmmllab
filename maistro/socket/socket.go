// Deprecated: Use svc.SocketService package instead
package socket

// import (
// 	"context"
// 	"fmt"
// 	"maistro/auth"
// 	"maistro/models"
// 	"maistro/util"
// 	"sync"
// 	"time"

// 	"github.com/gofiber/contrib/websocket"
// 	"github.com/gofiber/fiber/v2"
// 	"github.com/google/uuid"
// 	"github.com/sirupsen/logrus"
// )

// // Deprecated: Use svc.SocketService package instead

// type WebSocketConnectionRegistry struct {
// 	ByUser       map[string]map[string]*models.WebSocketConnection                         // userID -> connID -> conn
// 	ByType       map[models.WebSocketConnectionType]map[string]*models.WebSocketConnection // type -> connID -> conn
// 	All          map[string]*models.WebSocketConnection                                    // connID -> conn
// 	sync.RWMutex                                                                           // Use RWMutex for concurrent access
// }

// // Deprecated: Use svc.SocketService package instead
// var connectionRegistry = WebSocketConnectionRegistry{
// 	ByUser: make(map[string]map[string]*models.WebSocketConnection),
// 	ByType: map[models.WebSocketConnectionType]map[string]*models.WebSocketConnection{
// 		models.WebSocketConnectionTypeChat:   make(map[string]*models.WebSocketConnection),
// 		models.WebSocketConnectionTypeImage:  make(map[string]*models.WebSocketConnection),
// 		models.WebSocketConnectionTypeStatus: make(map[string]*models.WebSocketConnection),
// 	},
// 	All: make(map[string]*models.WebSocketConnection),
// }

// // Deprecated: Use svc.SocketService package instead
// func GetConnectionRegistry() *WebSocketConnectionRegistry {
// 	return &connectionRegistry
// }

// // Deprecated: Use svc.SocketService package instead
// func GetUserIDFromSocketConnection(c *websocket.Conn) string {
// 	userID, ok := c.Locals("userID").(string)
// 	if !ok {
// 		// Extract user ID from headers, query params, or JWT
// 		userID = c.Query("user_id")
// 		if userID == "" {
// 			// Try to get from headers
// 			userID = c.Headers("X-User-ID")
// 		}
// 	}

// 	if userID == "" {
// 		util.LogWarning("Failed to get user ID from WebSocket connection", nil)
// 	}

// 	// For development, assign a default user ID if none is provided
// 	// if userID == "" && util.IsDevelopment() {
// 	// 	userID = "dev-user"
// 	// 	util.LogWarning("Using default dev-user ID for WebSocket connection", nil)
// 	// }

// 	return userID
// }

// // Deprecated: Use svc.SocketService package instead
// func GetChatSocketSessionIDFromConnection(c *websocket.Conn, conversationID int) string {
// 	uid := GetUserIDFromSocketConnection(c)
// 	// Generate a unique session ID based on user ID and conversation ID
// 	return fmt.Sprintf("%s-%d", uid, conversationID)
// }

// // Deprecated: Use svc.SocketService package instead
// func GetChatSocketSessionID(uid string, conversationID int) string {
// 	// Generate a unique session ID based on user ID and conversation ID
// 	return fmt.Sprintf("%s-%d", uid, conversationID)
// }

// // Deprecated: Use svc.SocketService package instead
// func SetupWebSocketRoutes(app *fiber.App) {
// 	// WebSocket configuration
// 	wsConfig := websocket.Config{
// 		HandshakeTimeout: 10 * time.Second,
// 	}
// 	// Create a route group for WebSocket endpoints
// 	wsGroup := app.Group("/ws")

// 	// Add authentication middleware specific for WebSockets
// 	wsGroup.Use(websocketAuthMiddleware())

// 	// Chat WebSocket route
// 	wsGroup.Use("/chat", websocket.New(func(c *websocket.Conn) {
// 		handleChatWebSocket(c)
// 	}, wsConfig))

// 	// Image WebSocket route
// 	wsGroup.Use("/image", websocket.New(func(c *websocket.Conn) {
// 		handleImageWebSocket(c)
// 	}, wsConfig))

// 	// Status WebSocket route
// 	wsGroup.Use("/status", websocket.New(func(c *websocket.Conn) {
// 		handleStatusWebSocket(c)
// 	}, wsConfig))

// 	util.LogInfo("WebSocket routes initialized", nil)
// }

// // Deprecated: Use svc.SocketService package instead
// func websocketAuthMiddleware() fiber.Handler {
// 	return func(c *fiber.Ctx) error {
// 		// Check for token in query parameters
// 		token := c.Query("token")
// 		if token == "" {
// 			return fiber.ErrForbidden
// 		}

// 		// Validate the token and get user ID
// 		userID, err := auth.ValidateAndGetUserID(token)
// 		if err != nil {
// 			return fiber.ErrUnauthorized
// 		}
// 		c.Locals("userID", userID)

// 		// Only allow WebSocket upgrade requests
// 		if websocket.IsWebSocketUpgrade(c) {
// 			return c.Next()
// 		}

// 		return fiber.ErrUpgradeRequired
// 	}
// }

// // Deprecated: Use svc.SocketService package instead
// func registerConnection(userID string, connType models.WebSocketConnectionType, conn *websocket.Conn) string {
// 	connectionRegistry.Lock()
// 	defer connectionRegistry.Unlock()

// 	// Generate a connection ID
// 	connID := uuid.New().String()

// 	// Create WebSocket connection object
// 	wsConn := &models.WebSocketConnection{
// 		ID:        connID,
// 		UserID:    userID,
// 		Type:      connType,
// 		Conn:      conn,
// 		CreatedAt: time.Now(),
// 	}

// 	// Add to user map
// 	if _, exists := connectionRegistry.ByUser[userID]; !exists {
// 		connectionRegistry.ByUser[userID] = make(map[string]*models.WebSocketConnection)
// 	}
// 	connectionRegistry.ByUser[userID][connID] = wsConn

// 	// Add to type map
// 	connectionRegistry.ByType[connType][connID] = wsConn

// 	// Add to global map
// 	connectionRegistry.All[connID] = wsConn

// 	util.LogInfo("WebSocket connection registered", logrus.Fields{
// 		"userID":     userID,
// 		"connType":   connType,
// 		"connID":     connID,
// 		"totalConns": len(connectionRegistry.All),
// 	})

// 	return connID
// }

// // Deprecated: Use svc.SocketService package instead
// func unregisterConnection(userID string, connID string) {
// 	connectionRegistry.Lock()
// 	defer connectionRegistry.Unlock()

// 	// Get connection details before removal
// 	wsConn, exists := connectionRegistry.All[connID]
// 	if !exists {
// 		return
// 	}

// 	// Remove from all maps
// 	delete(connectionRegistry.ByUser[userID], connID)
// 	delete(connectionRegistry.ByType[wsConn.Type], connID)
// 	delete(connectionRegistry.All, connID)

// 	// Clean up user map if empty
// 	if len(connectionRegistry.ByUser[userID]) == 0 {
// 		delete(connectionRegistry.ByUser, userID)
// 	}

// 	util.LogInfo("WebSocket connection unregistered", logrus.Fields{
// 		"userID":     userID,
// 		"connType":   wsConn.Type,
// 		"connID":     connID,
// 		"totalConns": len(connectionRegistry.All),
// 	})
// }

// // Deprecated: Use svc.SocketService package instead
// func BroadcastToUser(userID string, message models.SocketMessage) {
// 	connectionRegistry.RLock()
// 	userConns, exists := connectionRegistry.ByUser[userID]
// 	if !exists || len(userConns) == 0 {
// 		connectionRegistry.RUnlock()
// 		return
// 	}

// 	// Make a copy of connections to avoid holding the lock during writes
// 	connsCopy := make([]*models.WebSocketConnection, 0, len(userConns))
// 	for _, conn := range userConns {
// 		connsCopy = append(connsCopy, conn)
// 	}
// 	connectionRegistry.RUnlock()

// 	// Send to all connections
// 	for _, wsConn := range connsCopy {
// 		c := wsConn.Conn.(*websocket.Conn)

// 		if c != nil {
// 			err := c.WriteJSON(message)
// 			if err != nil {
// 				util.HandleError(err)
// 				// Don't remove connection here, let ping/pong or read failure handle it
// 			}
// 		}
// 	}
// }

// // Deprecated: Use svc.SocketService package instead
// func BroadcastToType(connType models.WebSocketConnectionType, message models.SocketMessage) {
// 	connectionRegistry.RLock()
// 	typeConns := connectionRegistry.ByType[connType]
// 	if len(typeConns) == 0 {
// 		connectionRegistry.RUnlock()
// 		return
// 	}

// 	// Make a copy of connections to avoid holding the lock during writes
// 	connsCopy := make([]*models.WebSocketConnection, 0, len(typeConns))
// 	for _, conn := range typeConns {
// 		connsCopy = append(connsCopy, conn)
// 	}
// 	connectionRegistry.RUnlock()

// 	// Send to all connections
// 	for _, wsConn := range connsCopy {
// 		c := wsConn.Conn.(*websocket.Conn)

// 		if c != nil {
// 			err := c.WriteJSON(message)
// 			if err != nil {
// 				util.HandleError(err)
// 				// Don't remove connection here, let ping/pong or read failure handle it
// 			}
// 		}
// 	}
// }

// // Deprecated: Use svc.SocketService package instead
// func BroadcastToAll(message models.SocketMessage) {
// 	connectionRegistry.RLock()
// 	conns := connectionRegistry.All
// 	if len(conns) == 0 {
// 		connectionRegistry.RUnlock()
// 		return
// 	}

// 	// Make a copy of connections
// 	connsCopy := make([]*models.WebSocketConnection, 0, len(conns))
// 	for _, conn := range conns {
// 		connsCopy = append(connsCopy, conn)
// 	}
// 	connectionRegistry.RUnlock()

// 	// Send to all connections
// 	for _, wsConn := range connsCopy {
// 		c := wsConn.Conn.(*websocket.Conn)

// 		if c != nil {
// 			err := c.WriteJSON(message)
// 			if err != nil {
// 				util.HandleError(err)
// 				// Don't remove connection here, let ping/pong or read failure handle it
// 			}
// 		}
// 	}
// }

// // Deprecated: Use svc.SocketService package instead
// func RunBackgroundCleanup(ctx context.Context) {
// 	ticker := time.NewTicker(5 * time.Minute)
// 	defer ticker.Stop()

// 	for {
// 		select {
// 		case <-ctx.Done():
// 			return
// 		case <-ticker.C:
// 			cleanupInactiveConnections()
// 		}
// 	}
// }

// // Deprecated: Use svc.SocketService package instead
// func cleanupInactiveConnections() {
// 	connectionRegistry.Lock()
// 	defer connectionRegistry.Unlock()

// 	// Find connections to remove
// 	toRemove := make([]string, 0)
// 	for id, conn := range connectionRegistry.All {
// 		// Check if connection is closed
// 		if conn.Conn.(*websocket.Conn) == nil {
// 			toRemove = append(toRemove, id)
// 		}
// 	}

// 	// Remove connections
// 	for _, id := range toRemove {
// 		conn := connectionRegistry.All[id]
// 		userID := conn.UserID
// 		connType := conn.Type

// 		// Remove from all maps
// 		delete(connectionRegistry.ByUser[userID], id)
// 		delete(connectionRegistry.ByType[connType], id)
// 		delete(connectionRegistry.All, id)

// 		// Clean up user map if empty
// 		if len(connectionRegistry.ByUser[userID]) == 0 {
// 			delete(connectionRegistry.ByUser, userID)
// 		}
// 	}

// 	if len(toRemove) > 0 {
// 		util.LogInfo("Cleaned up inactive WebSocket connections", logrus.Fields{
// 			"removed":   len(toRemove),
// 			"remaining": len(connectionRegistry.All),
// 		})
// 	}
// }
