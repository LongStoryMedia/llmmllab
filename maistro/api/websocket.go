package api

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"maistro/auth"
	pxcx "maistro/context"
	"maistro/models"
	"maistro/proxy"
	"maistro/util"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// WebSocketManager manages active WebSocket connections by user
type WebSocketManager struct {
	// Maps userID to a map of connectionID to connection
	connections map[string]map[string]*websocket.Conn
	mu          sync.RWMutex
}

// Global WebSocket connection manager
var wsManager = &WebSocketManager{
	connections: make(map[string]map[string]*websocket.Conn),
}

// RegisterWebSocketRoutes adds WebSocket endpoints to the app
func RegisterWebSocketRoutes(app *fiber.App) {
	// Create a separate group for WebSocket routes that bypasses the main auth middleware
	// This is critical since WebSockets handle auth differently (via query param, not headers)
	wsGroup := app.Group("/ws")

	// Custom WebSocket authentication middleware
	wsGroup.Use(func(c *fiber.Ctx) error {
		// Check authentication via query parameter
		token := c.Query("token")
		if token == "" {
			return fiber.ErrUnauthorized
		}

		// Validate the token
		userID, err := auth.ValidateToken(token)
		if err != nil {
			util.LogWarning("WebSocket auth failed: Invalid token", logrus.Fields{
				"error": err.Error(),
			})
			return fiber.ErrUnauthorized
		}

		// Store user ID in context for use in the WebSocket handler
		ctx := context.WithValue(c.Context(), auth.UserIDKey, userID)
		c.SetUserContext(ctx)
		c.Locals("userID", userID)

		// Require WebSocket upgrade
		if websocket.IsWebSocketUpgrade(c) {
			return c.Next()
		}
		return fiber.ErrUpgradeRequired
	})

	// WebSocket endpoint for image generation notifications
	wsGroup.Get("/images", websocket.New(handleImageWebSocket))

	// WebSocket endpoint for chat functionality
	wsGroup.Get("/chat", websocket.New(WebSocketChatEndpoint))

	// WebSocket endpoint for status updates
	wsGroup.Get("/status", websocket.New(handleStatusWebSocket))
}

// handleImageWebSocket handles WebSocket connections for image generation updates
func handleImageWebSocket(c *websocket.Conn) {
	// Get user ID from context
	userID, ok := c.Locals("userID").(string)
	if !ok {
		util.LogWarning("WebSocket connection without valid user ID", nil)
		c.Close()
		return
	}

	// Generate a connection ID
	connID := uuid.New().String()

	// Register the new connection
	wsManager.mu.Lock()
	if _, exists := wsManager.connections[userID]; !exists {
		wsManager.connections[userID] = make(map[string]*websocket.Conn)
	}
	wsManager.connections[userID][connID] = c
	wsManager.mu.Unlock()

	util.LogInfo("New WebSocket connection established", logrus.Fields{
		"userID": userID,
		"connID": connID,
	})

	// Send initial message
	if err := c.WriteJSON(map[string]string{
		"type":    "connected",
		"message": "WebSocket connection established",
	}); err != nil {
		util.HandleError(err)
	}

	// Handle incoming messages (usually just ping/pong or connection close)
	for {
		_, _, err := c.ReadMessage()
		if err != nil {
			break // Client disconnected
		}
	}

	// Remove connection when done
	wsManager.mu.Lock()
	delete(wsManager.connections[userID], connID)
	if len(wsManager.connections[userID]) == 0 {
		delete(wsManager.connections, userID)
	}
	wsManager.mu.Unlock()

	util.LogInfo("WebSocket connection closed", logrus.Fields{
		"userID": userID,
		"connID": connID,
	})
}

// NotifyImageGenerated sends a notification to all connected clients for a user
// that an image has been generated
func NotifyImageGenerated(userID string, notification models.ImageGenerationNotification) {
	wsManager.mu.RLock()
	conns, exists := wsManager.connections[userID]
	wsManager.mu.RUnlock()

	if !exists {
		// No active connections for this user
		return
	}

	util.LogInfo("Sending image generation notification", logrus.Fields{
		"userID":  userID,
		"imageID": notification.ImageID,
	})

	// Send notification to all connections for this user
	for connID, conn := range conns {
		go func(connID string, conn *websocket.Conn) {
			// Set write deadline to prevent blocking indefinitely
			conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
			if err := conn.WriteJSON(notification); err != nil {
				util.HandleError(err, logrus.Fields{
					"userID":  userID,
					"connID":  connID,
					"imageID": notification.ImageID,
				})
			}
		}(connID, conn)
	}
}

// ChatSessionState represents the state of a chat session
type ChatSessionState string

const (
	ChatSessionActive   ChatSessionState = "active"
	ChatSessionPaused   ChatSessionState = "paused"
	ChatSessionComplete ChatSessionState = "complete"
	ChatSessionError    ChatSessionState = "error"
)

// ChatCommandType represents the type of WebSocket command
type ChatCommandType string

const (
	ChatCommandSend   ChatCommandType = "send"
	ChatCommandPause  ChatCommandType = "pause"
	ChatCommandResume ChatCommandType = "resume"
	ChatCommandCancel ChatCommandType = "cancel"
)

// ChatSocketCommand represents a command sent over WebSocket
type ChatSocketCommand struct {
	Type           ChatCommandType        `json:"type"`
	Message        string                 `json:"message"`
	ConversationID int                    `json:"conversation_id"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// ChatSocketResponse represents a response sent back over WebSocket
type ChatSocketResponse struct {
	Type      string `json:"type"`
	Content   string `json:"content,omitempty"`
	Error     string `json:"error,omitempty"`
	State     string `json:"state"`
	SessionID string `json:"session_id"`
	Timestamp int64  `json:"timestamp"`
}

// activeChatSessions keeps track of active chat WebSocket sessions
var activeChatSessions = struct {
	sync.Mutex
	sessions map[string]*chatSession
}{
	sessions: make(map[string]*chatSession),
}

type chatSession struct {
	sessionID      string
	state          ChatSessionState
	conversationID int
	userID         string
	cancelFunc     context.CancelFunc
	pausedContent  string
	lastMessage    string
	responseBuffer string
}

// WebSocketChatHandler handles WebSocket connections for chat
func WebSocketChatHandler(c *fiber.Ctx) error {
	if websocket.IsWebSocketUpgrade(c) {
		c.Locals("allowed", true)
		return c.Next()
	}
	return fiber.ErrUpgradeRequired
}

// WebSocketChatEndpoint is the endpoint for WebSocket chat connections
func WebSocketChatEndpoint(c *websocket.Conn) {
	// Get user ID from context
	uid := c.Locals("uid").(string)
	if uid == "" {
		util.HandleError(errors.New("User ID not found in WebSocket context"))
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Authentication required",
			State:     string(ChatSessionError),
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Create a new session
	sessionID := uuid.New().String()
	ctx, cancel := context.WithCancel(context.Background())

	session := &chatSession{
		sessionID:  sessionID,
		state:      ChatSessionActive,
		userID:     uid,
		cancelFunc: cancel,
	}

	// Register the session
	activeChatSessions.Lock()
	activeChatSessions.sessions[sessionID] = session
	activeChatSessions.Unlock()

	// Send initial response with session ID
	err := c.WriteJSON(ChatSocketResponse{
		Type:      "connected",
		Content:   "WebSocket connection established",
		State:     string(ChatSessionActive),
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
	})

	if err != nil {
		util.HandleError(err, logrus.Fields{"message": "Error sending initial WebSocket response"})
		cleanupChatSession(sessionID)
		return
	}

	// Main loop to handle incoming commands
	for {
		var cmd ChatSocketCommand
		if err := c.ReadJSON(&cmd); err != nil {
			util.HandleError(err, logrus.Fields{"message": "Error reading WebSocket message"})
			break
		}

		switch cmd.Type {
		case ChatCommandSend:
			// Handle new message
			handleChatSend(ctx, c, sessionID, uid, cmd)

		case ChatCommandPause:
			// Handle pause request
			handleChatPause(c, sessionID)

		case ChatCommandResume:
			// Handle resume request with corrections
			handleChatResume(c, sessionID, cmd)

		case ChatCommandCancel:
			// Handle cancel request
			handleChatCancel(c, sessionID)
		}
	}

	// Clean up when connection closes
	cleanupChatSession(sessionID)
}

// handleChatSend processes a new chat message
func handleChatSend(ctx context.Context, c *websocket.Conn, sessionID, userID string, cmd ChatSocketCommand) {
	activeChatSessions.Lock()
	session := activeChatSessions.sessions[sessionID]
	activeChatSessions.Unlock()

	if session == nil {
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Session not found",
			State:     string(ChatSessionError),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Update session state
	session.state = ChatSessionActive
	session.conversationID = cmd.ConversationID
	session.lastMessage = cmd.Message

	// Get conversation context
	cc, err := pxcx.GetCachedConversation(userID, cmd.ConversationID)
	if err != nil {
		util.HandleError(err, logrus.Fields{"message": "Error getting conversation"})
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Failed to retrieve conversation",
			State:     string(ChatSessionError),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// // Process metadata
	// generateImage := false
	// if cmd.Metadata != nil {
	// 	if genImage, ok := cmd.Metadata["generate_image"].(bool); ok {
	// 		generateImage = genImage
	// 	}
	// }

	// // Prepare message for processing
	// chatReq := models.ChatRequest{
	// 	Content:        cmd.Message,
	// 	ConversationID: cmd.ConversationID,
	// 	Metadata: &models.ChatRequestMetadata{
	// 		GenerateImage: generateImage,
	// 	},
	// }

	// Start a goroutine to process the message and stream responses
	go func() {
		// Prepare ollama request
		ollamaReqBody, err := cc.PrepareOllamaRequest(ctx, cmd.Message)
		if err != nil {
			util.HandleError(err, logrus.Fields{"message": "Failed to prepare Ollama request"})
			_ = c.WriteJSON(ChatSocketResponse{
				Type:      "error",
				Error:     "Failed to prepare request",
				State:     string(ChatSessionError),
				SessionID: sessionID,
				Timestamp: time.Now().Unix(),
			})
			return
		}

		// Process with ollama and stream results
		streamChatResponseToWebSocket(ctx, c, cc, ollamaReqBody, sessionID)
	}()
}

// handleChatPause pauses an ongoing chat session
func handleChatPause(c *websocket.Conn, sessionID string) {
	activeChatSessions.Lock()
	session := activeChatSessions.sessions[sessionID]
	activeChatSessions.Unlock()

	if session == nil {
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Session not found",
			State:     string(ChatSessionError),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Only pause if active
	if session.state != ChatSessionActive {
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "warning",
			Content:   "Session is not active",
			State:     string(session.state),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Cancel the current processing
	session.cancelFunc()

	// Create a new context for future operations
	_, cancel := context.WithCancel(context.Background())
	session.cancelFunc = cancel

	// Update state
	session.state = ChatSessionPaused

	// Notify client
	_ = c.WriteJSON(ChatSocketResponse{
		Type:      "paused",
		Content:   "Chat processing paused",
		State:     string(ChatSessionPaused),
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
	})

	util.LogInfo("Chat session paused", logrus.Fields{
		"sessionID": sessionID,
		"userID":    session.userID,
	})
}

// handleChatResume resumes a paused chat session with corrections
func handleChatResume(c *websocket.Conn, sessionID string, cmd ChatSocketCommand) {
	activeChatSessions.Lock()
	session := activeChatSessions.sessions[sessionID]
	activeChatSessions.Unlock()

	if session == nil {
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Session not found",
			State:     string(ChatSessionError),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Only resume if paused
	if session.state != ChatSessionPaused {
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "warning",
			Content:   "Session is not paused",
			State:     string(session.state),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Get conversation context
	cc, err := pxcx.GetCachedConversation(session.userID, session.conversationID)
	if err != nil {
		util.HandleError(err, logrus.Fields{"message": "Error getting conversation for resume"})
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Failed to retrieve conversation",
			State:     string(ChatSessionError),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Create combined message with original + corrections
	combinedMessage := session.lastMessage + "\n\nAdditional context/corrections: " + cmd.Message

	// Create a new context for the resumed operation
	ctx, cancel := context.WithCancel(context.Background())
	session.cancelFunc = cancel

	// Update session state
	session.state = ChatSessionActive

	// Notify client of resumption
	_ = c.WriteJSON(ChatSocketResponse{
		Type:      "resuming",
		Content:   "Chat processing resumed with corrections",
		State:     string(ChatSessionActive),
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
	})

	// Start a goroutine to process the message and stream responses
	go func() {
		// Add system note about continuation
		if cc.Notes == nil {
			cc.Notes = make([]string, 0)
		}
		cc.Notes = append(cc.Notes, "This is a continuation of a previous request with additional context.")

		// Prepare ollama request with continuation flag
		ollamaReqBody, err := cc.PrepareOllamaRequest(ctx, combinedMessage)
		if err != nil {
			util.HandleError(err, logrus.Fields{"message": "Failed to prepare Ollama request for resumed chat"})
			_ = c.WriteJSON(ChatSocketResponse{
				Type:      "error",
				Error:     "Failed to prepare request for continuation",
				State:     string(ChatSessionError),
				SessionID: sessionID,
				Timestamp: time.Now().Unix(),
			})
			return
		}

		// // Add metadata to indicate this is a continuation
		// if ollamaReqBody.Metadata == nil {
		// 	ollamaReqBody.Metadata = &models.ChatRequestMetadata{}
		// }
		// ollamaReqBody.Metadata.IsContinuation = true

		// Process with ollama and stream results
		streamChatResponseToWebSocket(ctx, c, cc, ollamaReqBody, sessionID)
	}()
}

// handleChatCancel cancels an ongoing chat session
func handleChatCancel(c *websocket.Conn, sessionID string) {
	activeChatSessions.Lock()
	session := activeChatSessions.sessions[sessionID]
	activeChatSessions.Unlock()

	if session == nil {
		_ = c.WriteJSON(ChatSocketResponse{
			Type:      "error",
			Error:     "Session not found",
			State:     string(ChatSessionError),
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		})
		return
	}

	// Cancel the current processing
	session.cancelFunc()

	// Create a new context for future operations
	_, cancel := context.WithCancel(context.Background())
	session.cancelFunc = cancel

	// Update state
	session.state = ChatSessionComplete

	// Notify client
	_ = c.WriteJSON(ChatSocketResponse{
		Type:      "cancelled",
		Content:   "Chat processing cancelled",
		State:     string(ChatSessionComplete),
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
	})

	util.LogInfo("Chat session cancelled", logrus.Fields{
		"sessionID": sessionID,
		"userID":    session.userID,
	})
}

// handleStatusWebSocket handles WebSocket connections for status updates
func handleStatusWebSocket(c *websocket.Conn) {
	// Get user ID from context
	userID, ok := c.Locals("userID").(string)
	if !ok {
		util.LogWarning("WebSocket connection without valid user ID", nil)
		c.Close()
		return
	}

	// Generate a connection ID
	connID := uuid.New().String()

	// Register the new connection
	wsManager.mu.Lock()
	if _, exists := wsManager.connections[userID]; !exists {
		wsManager.connections[userID] = make(map[string]*websocket.Conn)
	}
	wsManager.connections[userID][connID] = c
	wsManager.mu.Unlock()

	util.LogInfo("New WebSocket connection established for status updates", logrus.Fields{
		"userID": userID,
		"connID": connID,
	})

	// Send initial message
	if err := c.WriteJSON(map[string]string{
		"type":    "connected",
		"message": "WebSocket connection established for status updates",
	}); err != nil {
		util.HandleError(err)
	}

	// Handle incoming messages (usually just ping/pong or connection close)
	for {
		_, _, err := c.ReadMessage()
		if err != nil {
			break // Client disconnected
		}
	}

	// Remove connection when done
	wsManager.mu.Lock()
	delete(wsManager.connections[userID], connID)
	if len(wsManager.connections[userID]) == 0 {
		delete(wsManager.connections, userID)
	}
	wsManager.mu.Unlock()

	util.LogInfo("WebSocket connection closed for status updates", logrus.Fields{
		"userID": userID,
		"connID": connID,
	})
}

// NotifyStatusUpdate sends a status update to all connected clients for a user
func NotifyStatusUpdate(userID string, statusUpdate models.StatusUpdate) {
	wsManager.mu.RLock()
	conns, exists := wsManager.connections[userID]
	wsManager.mu.RUnlock()

	if !exists {
		// No active connections for this user
		return
	}

	util.LogInfo("Sending status update", logrus.Fields{
		"userID": userID,
		"status": statusUpdate.Stage,
	})

	// Send status update to all connections for this user
	for connID, conn := range conns {
		go func(connID string, conn *websocket.Conn) {
			// Set write deadline to prevent blocking indefinitely
			conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
			if err := conn.WriteJSON(statusUpdate); err != nil {
				util.HandleError(err, logrus.Fields{
					"userID": userID,
					"connID": connID,
					"status": statusUpdate.Stage,
				})
			}
		}(connID, conn)
	}
}

// streamChatResponseToWebSocket streams the chat response to the WebSocket connection
func streamChatResponseToWebSocket(ctx context.Context, c *websocket.Conn, cc *pxcx.ConversationContext, ollamaReqBody []byte, sessionID string) {
	activeChatSessions.Lock()
	session := activeChatSessions.sessions[sessionID]
	activeChatSessions.Unlock()

	if session == nil {
		return
	}

	// Start processing with ollama
	responseChan := make(chan *models.OllamaChatResp, 1000)
	errorChan := make(chan error, 1)

	handleChunk := func(chunk *models.OllamaChatResp) {
		select {
		case responseChan <- chunk:
		default:
			util.LogWarning("Response channel is full, dropping chunk", logrus.Fields{
				"sessionID": sessionID,
				"chunk":     chunk,
			})
		}
	}

	go func() {
		handler, statusCode, err := proxy.GetProxyHandler[*models.OllamaChatResp](ctx, ollamaReqBody, "/api/chat", http.MethodPost, true, time.Minute*10, &handleChunk)
		if err != nil {
			errorChan <- err
		}

		if statusCode != http.StatusOK {
			errorChan <- fmt.Errorf("unexpected status code: %d", statusCode)
		}
		defer close(responseChan)
		defer c.Close()
		// Use a buffered channel to wait for streaming to complete
		// Create a bufio.Writer that writes to the WebSocket connection
		w := &bytes.Buffer{}
		wr := bufio.NewWriter(w)

		if _, err := handler(wr); err != nil {
			errorChan <- fmt.Errorf("error during streaming: %w", err)
		}
		if err := wr.Flush(); err != nil {
			errorChan <- fmt.Errorf("error flushing writer: %w", err)
		}
	}()

	// Start sending responses to client
	var fullResponse strings.Builder

	for {
		select {
		case <-ctx.Done():
			// Context cancelled (paused or cancelled)
			activeChatSessions.Lock()
			if session, ok := activeChatSessions.sessions[sessionID]; ok {
				session.pausedContent = fullResponse.String()
			}
			activeChatSessions.Unlock()
			return

		case err := <-errorChan:
			// Error occurred
			util.HandleError(err, logrus.Fields{"message": "Error processing chat"})
			_ = c.WriteJSON(ChatSocketResponse{
				Type:      "error",
				Error:     "Processing error: " + err.Error(),
				State:     string(ChatSessionError),
				SessionID: sessionID,
				Timestamp: time.Now().Unix(),
			})
			return

		case chunk, ok := <-responseChan:
			if !ok {
				// Channel closed, processing complete
				completeResponse := fullResponse.String()

				// Store the response in the conversation
				go func() {
					ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
					defer cancel()
					_, err := cc.AddAssistantMessage(ctx, completeResponse)
					if err != nil {
						util.HandleError(err, logrus.Fields{"message": "Error storing assistant message"})
					}
				}()

				// Send completion message
				_ = c.WriteJSON(ChatSocketResponse{
					Type:      "complete",
					Content:   "",
					State:     string(ChatSessionComplete),
					SessionID: sessionID,
					Timestamp: time.Now().Unix(),
				})

				// Update session state
				activeChatSessions.Lock()
				if session, ok := activeChatSessions.sessions[sessionID]; ok {
					session.state = ChatSessionComplete
					session.responseBuffer = ""
				}
				activeChatSessions.Unlock()

				return
			}

			// Append to full response
			fullResponse.WriteString(chunk.GetChunkContent())

			// Send chunk to client
			err := c.WriteJSON(ChatSocketResponse{
				Type:      "chunk",
				Content:   chunk.GetChunkContent(),
				State:     string(ChatSessionActive),
				SessionID: sessionID,
				Timestamp: time.Now().Unix(),
			})

			if err != nil {
				util.HandleError(err, logrus.Fields{"message": "Error sending WebSocket response"})
				return
			}
		}
	}
}

// cleanupChatSession removes a chat session and releases resources
func cleanupChatSession(sessionID string) {
	activeChatSessions.Lock()
	defer activeChatSessions.Unlock()

	if session, ok := activeChatSessions.sessions[sessionID]; ok {
		session.cancelFunc()
		delete(activeChatSessions.sessions, sessionID)
	}
}
