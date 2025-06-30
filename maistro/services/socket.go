package svc

import (
	"context"
	"errors"
	"fmt"
	"maistro/models"
	"maistro/session"
	"maistro/util"
	"sync"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// WebSocketConnectionRegistry holds all active WebSocket connections
type WebSocketConnectionRegistry struct {
	Map map[string]map[models.WebSocketConnectionType]*models.WebSocketConnection // userID -> connID -> conn
	sync.RWMutex
}

type SocketService interface {
	// RunBackgroundCleanup starts a background cleanup process for sockets.
	RunBackgroundCleanup(ctx context.Context)
	// SendInfo sends an informational message to a user.
	SendInfo(stage models.SocketStageType, conversationID int, userID string, message string, progress int)
	// SendWarning sends a warning message to a user.
	SendWarning(stage models.SocketStageType, conversationID int, userID string, message string)
	// SendError sends an error message to a user.
	SendError(stage models.SocketStageType, conversationID int, userID string, message string) error
	// SendCompletion sends a completion message to a user.
	SendCompletion(stage models.SocketStageType, conversationID int, userID string, message string, payload ...any)
	// CancelAndBroadcast sends a cancellation message to a user.
	CancelAndBroadcast(conversationID int, userID string)
	// PauseAndBroadcast sends a paused message to a user.
	PauseAndBroadcast(conversationID int, userID string)
	// ResumeAndBroadcast sends a resumed message to a user.
	ResumeAndBroadcast(conversationID int, userID string)
	// SendConnected sends a connected message to a user.
	SendConnected(stage models.SocketStageType, conversationID int, userID string, message string)
	// RegisterConnection registers a new WebSocket connection for a user.
	RegisterConnection(userID string, connType models.WebSocketConnectionType, conn *websocket.Conn)
	// UnregisterConnection unregisters a WebSocket connection for a user.
	UnregisterConnection(userID string, connType models.WebSocketConnectionType)
}

type socketService struct {
	connectionRegistry WebSocketConnectionRegistry
}

var (
	SocketSvc = &socketService{
		connectionRegistry: WebSocketConnectionRegistry{
			Map: make(map[string]map[models.WebSocketConnectionType]*models.WebSocketConnection), // userID -> connID -> conn
		},
	}
)

// GetSocketService returns the singleton instance of the socket service
func GetSocketService() SocketService {
	return SocketSvc
}

func getChatSocketSessionID(uid string, conversationID int) string {
	// Generate a unique session ID based on user ID and conversation ID
	return fmt.Sprintf("%s-%d", uid, conversationID)
}

// SendInfo sends a status update to a specific user
func (s *socketService) SendInfo(stage models.SocketStageType, conversationID int, userID string, message string, progress int) {
	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
		"stage":          stage,
		"conversationID": conversationID,
		"userID":         userID,
		"progress":       progress,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeInfo,
		Content:   &message,
		State:     stage,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
		Progress:  &progress,
	}

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
}

func (s *socketService) SendWarning(stage models.SocketStageType, conversationID int, userID string, message string) {
	util.LogWarningAtCallLevel(message, 2, logrus.Fields{
		"stage":          stage,
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeWarning,
		Content:   &message,
		State:     stage,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
	}

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
}

func (s *socketService) SendError(stage models.SocketStageType, conversationID int, userID string, message string) error {
	util.HandleErrorAtCallLevel(errors.New(message), 2, logrus.Fields{
		"stage":          stage,
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeError,
		Content:   &message,
		State:     stage,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
	}

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)

	return errors.New(message)
}

func (s *socketService) SendCompletion(stage models.SocketStageType, conversationID int, userID string, message string, payload ...any) {
	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
		"stage":          stage,
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeComplete,
		Content:   &message,
		State:     stage,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
		Data:      payload,
	}

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
}

func (s *socketService) CancelAndBroadcast(conversationID int, userID string) {
	message := "Session cancelled"
	util.LogWarningAtCallLevel(message, 2, logrus.Fields{
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeCancelled,
		Content:   &message,
		State:     models.SocketStageTypeProcessing,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
	}

	session.GlobalStageManager.GetSessionState(userID, conversationID).Cancel()

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
}

func (s *socketService) PauseAndBroadcast(conversationID int, userID string) {
	message := "Session paused"
	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypePaused,
		Content:   &message,
		State:     models.SocketStageTypeProcessing,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
	}

	ss := session.GlobalStageManager.GetSessionState(userID, conversationID)

	ss.Pause()
	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
	ss.Checkpoint()
}

func (s *socketService) ResumeAndBroadcast(conversationID int, userID string) {
	message := "Session resumed"
	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeResumed,
		Content:   &message,
		State:     models.SocketStageTypeProcessing,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
	}

	session.GlobalStageManager.GetSessionState(userID, conversationID).Resume()

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
}

func (s *socketService) SendConnected(stage models.SocketStageType, conversationID int, userID string, message string) {
	util.LogInfoAtCallLevel(message, 2, logrus.Fields{
		"stage":          stage,
		"conversationID": conversationID,
		"userID":         userID,
	})
	csr := models.SocketMessage{
		ID:        uuid.New().String(),
		Type:      models.MessageTypeConnected,
		Content:   &message,
		State:     stage,
		SessionID: getChatSocketSessionID(userID, conversationID),
		Timestamp: time.Now(),
	}

	// Broadcast to user
	s.broadcastToUser(userID, models.WebSocketConnectionTypeStatus, csr)
}

// broadcastToUser sends a message to all connections for a specific user
func (s *socketService) broadcastToUser(userID string, connectionType models.WebSocketConnectionType, message models.SocketMessage) {
	s.connectionRegistry.RLock()
	userConns, exists := s.connectionRegistry.Map[userID]
	if !exists || len(userConns) == 0 {
		s.connectionRegistry.RUnlock()
		return
	}

	// Make a copy of connections to avoid holding the lock during writes
	connsCopy := make([]*models.WebSocketConnection, 0, len(userConns))
	for _, conn := range userConns {
		connsCopy = append(connsCopy, conn)
	}
	s.connectionRegistry.RUnlock()

	util.LogDebug("Broadcasting message to user connections", logrus.Fields{
		"userID":         userID,
		"connectionType": connectionType,
		"message":        message,
	})
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
func (s *socketService) RunBackgroundCleanup(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.cleanupInactiveConnections()
		}
	}
}

// cleanupInactiveConnections removes closed or inactive connections
func (s *socketService) cleanupInactiveConnections() {
	s.connectionRegistry.Lock()
	defer s.connectionRegistry.Unlock()

	// Find connections to remove
	for uid, uc := range s.connectionRegistry.Map {
		for t, conn := range uc {
			if conn == nil || conn.Conn == nil {
				util.LogInfo("Removing inactive WebSocket connection", logrus.Fields{
					"userID":   conn.UserID,
					"connType": conn.Type,
					"connID":   conn.ID,
				})
				delete(uc, t)
			}
		}

		// Clean up user map if empty
		if len(uc) == 0 {
			delete(s.connectionRegistry.Map, uid)
		}
	}
}

// registerConnection adds a new connection to the registry
func (s *socketService) RegisterConnection(userID string, connType models.WebSocketConnectionType, conn *websocket.Conn) {
	s.connectionRegistry.Lock()
	defer s.connectionRegistry.Unlock()

	// Create WebSocket connection object
	wsConn := &models.WebSocketConnection{
		UserID:    userID,
		Type:      connType,
		Conn:      conn,
		CreatedAt: time.Now(),
	}

	// Add to user map
	uc, exists := s.connectionRegistry.Map[userID]
	if !exists {
		s.connectionRegistry.Map[userID] = make(map[models.WebSocketConnectionType]*models.WebSocketConnection)
		uc = s.connectionRegistry.Map[userID] // Update uc to reference the newly created map
	}

	uc[connType] = wsConn

	util.LogInfo("WebSocket connection registered", logrus.Fields{
		"userID":   userID,
		"connType": connType,
	})
}

// unregisterConnection removes a connection from the registry
func (s *socketService) UnregisterConnection(userID string, connType models.WebSocketConnectionType) {
	s.connectionRegistry.Lock()
	defer s.connectionRegistry.Unlock()

	// Get connection details before removal
	wsConn, exists := s.connectionRegistry.Map[userID]
	if !exists {
		return
	}

	delete(wsConn, connType)

	// Clean up user map if empty
	if len(s.connectionRegistry.Map[userID]) == 0 {
		delete(s.connectionRegistry.Map, userID)
	}

	util.LogInfo("WebSocket connection unregistered", logrus.Fields{
		"userID":   userID,
		"connType": connType,
	})
}
