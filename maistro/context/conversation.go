package context

import (
	"context"
	"fmt"
	"maistro/models"
	"maistro/proxy"
	"maistro/storage"
	"maistro/util"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/sirupsen/logrus"
)

type CtxKey string

const (
	ConversationContextKey CtxKey = "conversation_id"
)

// ConversationContext defines the interface for managing conversation context
type ConversationContext interface {
	GetUserID() string
	GetConversationID() int
	GetTitle() string
	GetMasterSummary() *models.Summary
	GetSummaries() []models.Summary
	GetMessages() []models.Message
	GetRetrievedMemories() []models.Memory
	GetSearchResults() []models.SearchResult
	GetImages() []models.ImageMetadata
	GetIntent() *Intent
	AddUserMessage(ctx context.Context, content string) ([][]float32, int, error)
	AddAssistantMessage(ctx context.Context, content string) ([][]float32, error)
	SummarizeMessages(ctx context.Context) (*models.Summary, error)
	PrepareOllamaRequest(ctx context.Context, request models.ChatRequest) ([]byte, *models.ChatReq, error)
	ClearNotes()
}

// conversationContext manages context for a conversation
type conversationContext struct {
	userID            string
	conversationID    int
	title             string
	masterSummary     *models.Summary
	summaries         []models.Summary
	messages          []models.Message
	retrievedMemories []models.Memory       // Memories retrieved using semantic or keyword search
	searchResults     []models.SearchResult // Search results from web search
	notes             []string
	images            []models.ImageMetadata // Images associated with the conversation
	afterThoughts     []models.Message
	intent            *Intent // Detected intent for the conversation
}

func AsCC(cc any) ConversationContext {
	if cc == nil {
		return nil
	}

	if cc, ok := cc.(*conversationContext); ok {
		return cc
	}
	return nil
}

func (cc *conversationContext) GetUserID() string {
	return cc.userID
}
func (cc *conversationContext) GetConversationID() int {
	return cc.conversationID
}
func (cc *conversationContext) GetTitle() string {
	return cc.title
}
func (cc *conversationContext) GetMasterSummary() *models.Summary {
	return cc.masterSummary
}
func (cc *conversationContext) GetSummaries() []models.Summary {
	return cc.summaries
}
func (cc *conversationContext) GetMessages() []models.Message {
	return cc.messages
}
func (cc *conversationContext) GetRetrievedMemories() []models.Memory {
	return cc.retrievedMemories
}
func (cc *conversationContext) GetSearchResults() []models.SearchResult {
	return cc.searchResults
}
func (cc *conversationContext) GetImages() []models.ImageMetadata {
	return cc.images
}
func (cc *conversationContext) GetIntent() *Intent {
	return cc.intent
}
func (cc *conversationContext) ClearNotes() {
	cc.notes = make([]string, 0)
}

var cc *conversationContext

// GetOrCreateConversation retrieves or creates a conversation context
func GetOrCreateConversation(ctx context.Context, userID string, conversationID *int) (ConversationContext, error) {
	if cc != nil && cc.userID == userID && (conversationID == nil || (cc.conversationID == *conversationID)) {
		util.LogInfo("Using existing conversation context", logrus.Fields{
			"userId":         userID,
			"conversationId": cc.conversationID,
		})
		return cc, nil
	}
	cc = &conversationContext{
		userID:         userID,
		conversationID: -1,
	}

	// Ensure user exists and load user-specific configuration
	if err := storage.EnsureUser(ctx, userID); err != nil {
		return nil, fmt.Errorf("failed to ensure user exists: %w", err)
	}
	// Check if we have a conversation ID
	if conversationID != nil {
		// Try to get from cache first
		if cache := GetCache(); cache != nil {
			if cachedContext, found := cache.Get(userID, *conversationID); found {
				util.LogInfo("Retrieved conversation from cache", logrus.Fields{
					"conversationId": *conversationID,
					"userId":         userID,
				})
				return cachedContext, nil
			}
		}

		// Verify the conversation exists and belongs to the user
		conv, err := storage.ConversationStoreInstance.GetConversation(ctx, *conversationID)
		if err != nil {
			if err == pgx.ErrNoRows {
				cid, err := storage.ConversationStoreInstance.CreateConversation(ctx, userID, "")
				if err != nil {
					return nil, fmt.Errorf("failed to create conversation: %w", err)
				}
				return GetOrCreateConversation(ctx, userID, &cid)
			}
			return nil, fmt.Errorf("failed to get conversation: %w", err)
		}

		if conv.UserID != userID {
			return nil, fmt.Errorf("conversation does not belong to user")
		}

		cc.userID = userID
		cc.conversationID = conv.ID
		cc.title = conv.Title

		// Load messages
		if err := loadConversationMessages(ctx, cc); err != nil {
			return nil, err
		}

		// Load summaries
		if err := loadConversationSummaries(ctx, cc); err != nil {
			return nil, err
		}

		// Store in cache
		if cache := GetCache(); cache != nil {
			cache.Set(cc)
			util.LogInfo("Cached conversation", logrus.Fields{
				"conversationId": cc.conversationID,
				"userId":         userID,
			})
		}
	} else {
		// Create a new conversation
		id, err := storage.ConversationStoreInstance.CreateConversation(ctx, userID, "New conversation")
		if err != nil {
			return nil, fmt.Errorf("failed to create conversation: %w", err)
		}
		cc.conversationID = id

		// Store new conversation in cache
		if cache := GetCache(); cache != nil {
			cache.Set(cc)
		}
	}

	if cc.messages == nil {
		cc.messages = make([]models.Message, 0)
	}
	if cc.summaries == nil {
		cc.summaries = make([]models.Summary, 0)
	}
	if cc.retrievedMemories == nil {
		cc.retrievedMemories = make([]models.Memory, 0)
	}
	if cc.searchResults == nil {
		cc.searchResults = make([]models.SearchResult, 0)
	}
	if cc.notes == nil {
		cc.notes = make([]string, 0)
	}
	if cc.images == nil {
		cc.images = make([]models.ImageMetadata, 0)
	}
	if cc.afterThoughts == nil {
		cc.afterThoughts = make([]models.Message, 0)
	}
	if cc.intent == nil {
		cc.intent = &Intent{
			ImageGeneration: false,
			Memory:          false,
			DeepResearch:    false,
			WebSearch:       false,
		}
	}

	return cc, nil
}

// loadConversationMessages loads messages for a conversation
func loadConversationMessages(ctx context.Context, cc *conversationContext) error {
	messages, err := storage.MessageStoreInstance.GetConversationHistory(ctx, cc.conversationID)
	if err != nil && err != pgx.ErrNoRows {
		return fmt.Errorf("failed to load conversation history: %w", err)
	}

	// Convert storage.Message to models.Message
	for _, msg := range messages {
		cc.messages = append(cc.messages, models.Message{
			Role:    msg.Role,
			Content: msg.Content,
			ID:      msg.ID,
		})
	}

	return nil
}

// loadConversationSummaries loads summaries for a conversation
func loadConversationSummaries(ctx context.Context, cc *conversationContext) error {
	summaries, err := storage.SummaryStoreInstance.GetSummariesForConversation(ctx, cc.conversationID)
	if err != nil {
		return fmt.Errorf("failed to load summaries: %w", err)
	}

	// Convert storage.Summary to models.Summary
	for _, summary := range summaries {
		// Check if this is a master summary (level 0)
		if summary.Level == 0 {
			// Find the most recent master summary
			if cc.masterSummary == nil || summary.ID > cc.masterSummary.ID {
				cc.masterSummary = &models.Summary{
					Content: summary.Content,
					Level:   summary.Level,
					ID:      summary.ID,
				}
			}
		} else {
			cc.summaries = append(cc.summaries, models.Summary{
				Content: summary.Content,
				Level:   summary.Level,
				ID:      summary.ID,
			})
		}
	}

	return nil
}

// createMessageMemory creates a memory for a message and stores it in the database
func (cc *conversationContext) createMessageMemory(ctx context.Context, msg models.Message, usrCfg *models.UserConfig) ([][]float32, error) {
	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.EmbeddingProfileID)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to get model profile for embedding: %w", err))
	}

	embeddings, err := proxy.GetOllamaEmbedding(ctx, msg.Content, profile)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to get Ollama embedding: %w", err))
	}

	// Add to database
	if err := storage.MemoryStoreInstance.StoreMemory(ctx, cc.userID, "message", msg.Role, msg.ID, embeddings); err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to add memory: %w", err))
	}

	return embeddings, nil
}

// createSummaryMemory creates a memory for a summary and stores it in the database
func (cc *conversationContext) createSummaryMemory(ctx context.Context, summary models.Summary, usrCfg *models.UserConfig) ([][]float32, error) {
	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.EmbeddingProfileID)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to get model profile for embedding: %w", err))
	}

	embeddings, err := proxy.GetOllamaEmbedding(ctx, summary.Content, profile)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to get Ollama embedding: %w", err))
	}

	// Add to database
	if err := storage.MemoryStoreInstance.StoreMemory(ctx, cc.userID, "summary", "system", summary.ID, embeddings); err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to add memory: %w", err))
	}

	return embeddings, nil
}

// AddUserMessage adds a user message to the conversation
func (cc *conversationContext) AddUserMessage(ctx context.Context, content string) ([][]float32, int, error) {
	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		return nil, -1, util.HandleError(fmt.Errorf("failed to get user config: %w", err))
	}

	// Add to database
	msgID, err := storage.MessageStoreInstance.AddMessage(ctx, cc.conversationID, "user", content, usrCfg)
	if err != nil {
		return nil, -1, util.HandleError(fmt.Errorf("failed to add user message: %w", err))
	}

	msg := models.Message{
		Role:    "user",
		Content: content,
		ID:      msgID,
	}

	// Add to context
	cc.messages = append(cc.messages, msg)

	// Create memory for the message
	embeddings, err := cc.createMessageMemory(ctx, msg, usrCfg)
	if err != nil {
		return nil, msgID, util.HandleError(fmt.Errorf("failed to create message memory: %w", err))
	}

	// Update title if this is the first message
	if len(cc.messages) == 1 {
		title := generateTitle(content)
		if err := storage.ConversationStoreInstance.UpdateConversationTitle(ctx, cc.conversationID, title); err != nil {
			util.HandleError(fmt.Errorf("failed to update conversation title: %v", err))
		}
	}

	// Update cache with modified conversation context
	if cache := GetCache(); cache != nil {
		cache.Set(cc)
	}

	return embeddings, msgID, nil
}

// AddAssistantMessage adds an assistant message to the conversation
func (cc *conversationContext) AddAssistantMessage(ctx context.Context, content string) ([][]float32, error) {
	util.LogInfo("Storing assistant response", logrus.Fields{"length": len(content)})

	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user config: %w", err)
	}
	// Add to database
	msgID, err := storage.MessageStoreInstance.AddMessage(ctx, cc.conversationID, "assistant", content, usrCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to add assistant message: %w", err)
	}

	msg := models.Message{
		Role:    "assistant",
		Content: content,
		ID:      msgID,
	}

	// Add to context
	cc.messages = append(cc.messages, msg)

	// Create memory for the message
	embeddings, err := cc.createMessageMemory(ctx, msg, usrCfg)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to create message memory: %w", err))
	}

	// Check if we need to summarize messages
	if cc.shouldSummarize() {
		util.LogInfo("Summarizing messages for conversation")
		go func() {
			bgrndCtx, cancel := context.WithTimeout(context.Background(), 120*time.Minute)
			defer cancel()
			if _, err := cc.SummarizeMessages(bgrndCtx); err != nil {
				util.HandleError(fmt.Errorf("failed to summarize messages: %w", err))
				// Continue even if summarization fails
			}
		}()
	}

	go func(cc *conversationContext) {
		// Update cache with modified conversation context
		if cache := GetCache(); cache != nil {
			cache.Set(cc)
		}
	}(cc)

	return embeddings, nil
}

// generateTitle creates a title from the first user message
func generateTitle(content string) string {
	// Simple implementation: use first 30 chars of content or less
	maxLen := 30
	title := content
	if len(title) > maxLen {
		title = title[:maxLen] + "..."
	}
	return title
}

// truncateForLog truncates a string for logging purposes
func truncateForLog(content string) string {
	maxLen := 100
	if len(content) <= maxLen {
		return content
	}
	return content[:maxLen] + "..."
}

// shouldSummarize determines if the conversation needs to be summarized
func (cc *conversationContext) shouldSummarize() bool {
	// Load user-specific configuration
	userConfig, err := GetUserConfig(cc.userID)
	if err != nil {
		util.LogWarning("Could not load user configuration, using system defaults", logrus.Fields{"error": err})
		return false
	}

	// Get message threshold from configuration
	messageThreshold := userConfig.Summarization.MessagesBeforeSummary

	// Count messages since last summary
	messagesSinceLastSummary := len(cc.messages)

	// If we have summaries, adjust the count to only include messages since the last summary
	if len(cc.summaries) > 0 || cc.masterSummary != nil {
		// Find the most recent summary ID
		var maxSummaryID int
		if cc.masterSummary != nil {
			maxSummaryID = cc.masterSummary.ID
		}

		for _, summary := range cc.summaries {
			if summary.ID > maxSummaryID {
				maxSummaryID = summary.ID
			}
		}

		// Count only messages that came after the most recent summary
		messagesSinceLastSummary = 0
		for _, msg := range cc.messages {
			if msg.ID > maxSummaryID {
				messagesSinceLastSummary++
			}
		}
	}

	// Determine if we need to summarize
	return messagesSinceLastSummary >= messageThreshold
}
