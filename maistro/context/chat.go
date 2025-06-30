package context

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maistro/models"
	"maistro/proxy"
	"maistro/recherche"
	"maistro/session"
	"maistro/storage"
	"maistro/util"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// generateSummarization creates a summary using Ollama
func (cc *ConversationContext) generateSummarization(messages []models.Message, summaryModel *models.ModelProfile) (string, error) {
	// Build Ollama messages
	ollamaMessages := []models.ChatMessage{
		{
			Role:    "system",
			Content: summaryModel.SystemPrompt,
		},
	}

	// Add user messages
	for _, message := range messages {
		ollamaMessages = append(ollamaMessages, models.ChatMessage{
			Role:    message.Role,
			Content: message.Content,
		})
	}

	// Ensure the last message is a user message with the summarization instruction
	if len(ollamaMessages) == 0 || ollamaMessages[len(ollamaMessages)-1].Role != "user" {
		ollamaMessages = append(ollamaMessages, models.ChatMessage{
			Role:    "user",
			Content: summaryModel.SystemPrompt, // Use the system prompt as the summarization instruction
		})
	}

	util.LogInfo("Using model for text generation", logrus.Fields{"model": summaryModel.ModelName})

	// Create a long-lived context for generation
	longCtx, cancel := context.WithTimeout(context.Background(), 120*time.Minute)
	defer cancel()

	resp, err := proxy.StreamOllamaChatRequest(longCtx, summaryModel, ollamaMessages, cc.UserID, cc.ConversationID)
	r := util.RemoveThinkTags(resp)
	if err != nil {
		return r, util.HandleError(err)
	}
	util.LogDebug("Ollama summary response", logrus.Fields{"resp": resp, "err": err})
	return r, err
}

// PrepareOllamaRequest prepares the request for Ollama
func (cc *ConversationContext) PrepareOllamaRequest(ctx context.Context, request models.ChatRequest) ([]byte, *models.ChatReq, error) {
	if request.Content == "" {
		return nil, nil, util.HandleError(errors.New("message cannot be empty"))
	}

	usrCfg, err := GetUserConfig(cc.UserID)
	if err != nil {
		return nil, nil, util.HandleError(err)
	}

	pp, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.PrimaryProfileID)
	if err != nil {
		return nil, nil, util.HandleError(err)
	}
	if pp == nil {
		return nil, nil, util.HandleError(errors.New("model profile not found"))
	}

	ep, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.EmbeddingProfileID)
	if err != nil {
		return nil, nil, util.HandleError(err)
	}
	if ep == nil {
		return nil, nil, util.HandleError(errors.New("embedding profile not found"))
	}

	embeddings, mid, err := cc.AddUserMessage(ctx, request.Content)
	if err != nil {
		return nil, nil, util.HandleError(err)
	}

	ss := session.GlobalStageManager.GetSessionState(cc.UserID, cc.ConversationID)
	ss.AddRollbackFunc(func() error {
		return storage.MessageStoreInstance.DeleteMessage(ctx, mid)
	})
	if ss.IsPaused() {
		cc.AfterThoughts = append(cc.AfterThoughts, models.Message{
			Role:    "user",
			Content: request.Content,
			ID:      request.ConversationID,
		})
		if err := ss.Resume(); err != nil {
			return nil, nil, util.HandleError(err)
		}
		return nil, nil, nil // Will pick up at Checkpoint
	}

	ollamaReq := models.ChatReq{
		Messages:       []models.ChatMessage{}, // Will be populated later
		ConversationID: &cc.ConversationID,
	}
	ss.CurrentRequest = &ollamaReq

	// Always set streaming to true to prevent timeouts
	ollamaReq.Stream = true
	ollamaReq.Options = pp.Parameters.ToMap()
	ollamaReq.Model = pp.ModelName

	var wg sync.WaitGroup

	cc.Intent = &Intent{}

	ss.GetStage(models.SocketStageTypeInterpreting).UpdateProgress(0, "Detecting intent and extracting content")
	if request.Metadata != nil && request.Metadata.Type == models.ChatMessageMetadataTypeImage {
		// If the request is an image generation request, update intent to image generation
		cc.Intent.ImageGeneration = true
	}

	if err := cc.DetectIntent(ctx, request); err != nil {
		ss.GetStage(models.SocketStageTypeInterpreting).Fail("Error detecting intent", err)
		// Non-critical error, we can continue without intent
	} else {
		// If the intent indicates a web search, get the web search results
		if cc.Intent.WebSearch {
			wg.Add(1)
			go func(cc *ConversationContext, q string, ss *session.SessionState) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
				defer cancel()
				ss.Checkpoint()
				// Inject the search results into the conversation context
				if err := cc.SearchAndInjectResults(ctx, q); err != nil {
					ss.GetStage(models.SocketStageTypeSearchingWeb).Fail("Error searching web", err)
					// Non-critical error, we can continue without web search results
				}
			}(cc, request.Content, ss)
		}

		if cc.Intent.Memory {
			wg.Add(1)
			go func(cc *ConversationContext, embedding [][]float32, ss *session.SessionState) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
				defer cancel()
				ss.Checkpoint()
				// Attempt to retrieve and inject relevant memories for the user's query
				if err := cc.RetrieveAndInjectMemories(ctx, embedding, nil, nil); err != nil {
					// Non-critical error, we can continue without memories
					ss.GetStage(models.SocketStageTypeRetrievingMemories).Fail("Error retrieving memories", err)
				}
			}(cc, embeddings, ss)
		}
	}

	// Try to extract content from any URLs in the user message
	wg.Add(1)
	go func(cc *ConversationContext, ss *session.SessionState) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
		defer cancel()
		ss.Checkpoint()
		results, err := recherche.ExtractUrlContentFromQuery(ctx, request.Content, cc.UserID, cc.ConversationID)
		if err != nil {
			util.LogWarning("Error extracting URL content", logrus.Fields{"error": err})
			// Non-critical error, we can continue without URL content
		} else if results != nil {
			// Inject the extracted content into the conversation context
			if err := cc.InjectSearchResults(ctx, results, "Here is the content from the user's included url"); err != nil {
				util.LogWarning("Error injecting URL content", logrus.Fields{"error": err})
				// Non-critical error, we can continue without URL content
			}
		}
	}(cc, ss)

	// Wait for all goroutines to finish
	wg.Wait()

	// If embeddings are empty, log a warning and return an empty JSON object
	if len(embeddings) == 0 {
		util.LogWarning("Empty embedding vector", logrus.Fields{"userMessage": request})
		return []byte("{}"), nil, nil
	}

	util.LogDebug("Conversation context prepared for Ollama request", logrus.Fields{
		"conversationId": cc.ConversationID,
		"userId":         cc.UserID,
		"message_num":    len(cc.Messages),
		"search_results": len(cc.SearchResults),
		"memories":       len(cc.RetrievedMemories),
		"summaries":      len(cc.Summaries),
	})

	ss.Checkpoint()

	// Convert conversation context to Ollama format (includes summaries, memories, and web search results)
	return cc.ChainMessages(&ollamaReq)
}

// ChainMessages uses the conversation context to chain messages together
// This prepares the request for Ollama by enhancing it with RAG, summaries, and recent messages
// It returns the JSON-encoded request body for Ollama
// and handles any errors that occur during the process.
func (cc *ConversationContext) ChainMessages(req *models.ChatReq) ([]byte, *models.ChatReq, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	ss := session.GlobalStageManager.GetSessionState(cc.UserID, cc.ConversationID)

	req.Messages = make([]models.ChatMessage, 0)

	ss.Checkpoint()
	if err := cc.EnhanceRequestWithRAG(ctx, req); err != nil {
		return nil, req, ss.GetStage(models.SocketStageTypeProcessing).Fail("Error enhancing request with RAG", err)
	}

	// Add master summary if available
	if cc.MasterSummary != nil {
		// Add system message to introduce the conversation with master summary
		req.Messages = append([]models.ChatMessage{CreateSystemMessage(
			fmt.Sprintf("This is a continued conversation. Here is a comprehensive summary of the conversation history:\n%s", cc.MasterSummary.Content))}, req.Messages...)

		util.LogInfo("Using master summary for conversation context", logrus.Fields{
			"summaryId": cc.MasterSummary.ID,
		})
	}

	ss.Checkpoint()
	// Add level summaries (one from each level)
	if err := cc.addLevelSummariesToReq(req); err != nil {
		return nil, req, ss.GetStage(models.SocketStageTypeProcessing).Fail("Error adding level summaries to request", err)
	}

	ss.Checkpoint()
	// Add recent messages
	if err := cc.addRecentMessagesToReq(req); err != nil {
		return nil, req, ss.GetStage(models.SocketStageTypeProcessing).Fail("Error adding recent messages to request", err)
	}

	// Add any notes to the request
	if len(cc.Notes) > 0 {
		for _, note := range cc.Notes {
			req.Messages = append(req.Messages, models.ChatMessage{
				Role:    "system",
				Content: fmt.Sprintf("Note: %s", note),
			})
			util.LogDebug("Added note to request", logrus.Fields{
				"note": note,
			})
		}
		util.LogInfo("Added notes to request", logrus.Fields{
			"count": len(cc.Notes),
			"notes": cc.Notes,
		})
	}
	msgsInOrder := make([]string, 0)

	// debug log the content of each message in the request
	for i, msg := range req.Messages {
		util.LogDebug(truncateForLog(msg.Content), logrus.Fields{
			"index": i,
			"role":  msg.Role,
		})

		msgsInOrder = append(msgsInOrder, msg.Role)
	}

	util.LogDebug("Added messages to request", logrus.Fields{
		"count":    len(req.Messages),
		"messages": msgsInOrder,
	})

	bytes, err := json.Marshal(req)
	if err != nil {
		return nil, req, ss.GetStage(models.SocketStageTypeProcessing).Fail("Error preparing request for Ollama", err)
	}
	ss.GetStage(models.SocketStageTypeProcessing).Complete("Request prepared for Ollama", nil)
	return bytes, req, nil
}

// addLevelSummariesToReq adds one summary from each level to the request
func (cc *ConversationContext) addLevelSummariesToReq(req *models.ChatReq) error {
	userConfig, err := GetUserConfig(cc.UserID)
	if err != nil {
		util.LogWarning("Could not load user configuration, using system defaults", logrus.Fields{"error": err})
		return err
	}

	highestLevel := findMaxLevel(cc.Summaries)
	summariesByLevel := groupSummariesByLevel(cc.Summaries)
	summaryCount := 0
	maxLevel := userConfig.Summarization.MaxSummaryLevels
	for level := highestLevel; level >= 0 && level <= maxLevel; level-- {
		levelSummaries := summariesByLevel[level]
		if len(levelSummaries) == 0 {
			util.LogDebug("No summaries found for level", logrus.Fields{"level": level})
			continue
		}
		mostRecentSummary := levelSummaries[len(levelSummaries)-1]
		req.Messages = append([]models.ChatMessage{CreateSystemMessage(fmt.Sprintf("Previous conversation summary (level %d): %s", level, mostRecentSummary.Content))}, req.Messages...)
		summaryCount++
	}
	if summaryCount > 0 {
		util.LogInfo("Using summaries (one per level) for conversation context", logrus.Fields{
			"count": summaryCount,
		})
	}
	return nil
}

// addRecentMessagesToReq adds the most recent messages to the request
func (cc *ConversationContext) addRecentMessagesToReq(req *models.ChatReq) error {
	// Get user-specific configuration
	userConfig, err := GetUserConfig(cc.UserID)
	if err != nil {
		util.LogWarning("Could not load user configuration, using system defaults", logrus.Fields{"error": err})
		return err
	}

	// Calculate starting index for messages
	startIndex := max(len(cc.Messages)-userConfig.Summarization.MessagesBeforeSummary, 0)
	startIndex += len(cc.AfterThoughts)
	msgs := append(cc.Messages, cc.AfterThoughts...)

	util.LogInfo("Including most recent messages in request to Ollama", logrus.Fields{
		"count": userConfig.Summarization.MessagesBeforeSummary})

	// Add regular messages (most recent based on configuration)
	// Ensure messages are in chronological order (oldest first, newest last)
	for i := startIndex; i < len(msgs); i++ {
		if msgs[i].ID < 0 {
			util.LogWarning("Message ID is not set", logrus.Fields{
				"role":    cc.Messages[i].Role,
				"content": cc.Messages[i].Content,
			})
			continue
		}
		req.Messages = append(req.Messages, models.ChatMessage{
			Role:    msgs[i].Role,
			Content: msgs[i].Content,
		})
	}

	return nil
}
