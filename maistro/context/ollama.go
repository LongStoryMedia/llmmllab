package context

import (
	"context"
	"errors"
	"maistro/models"
	"maistro/proxy"
	"maistro/recherche"
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

	resp, err := proxy.StreamOllamaChatRequest(longCtx, summaryModel, ollamaMessages, &cc.UserID)
	r := util.RemoveThinkTags(resp)
	if err != nil {
		return r, util.HandleError(err)
	}
	util.LogDebug("Ollama summary response", logrus.Fields{"resp": resp, "err": err})
	return r, err
}

// PrepareOllamaRequest prepares the request for Ollama
func (cc *ConversationContext) PrepareOllamaRequest(ctx context.Context, message string) ([]byte, error) {
	if message == "" {
		return nil, util.HandleError(errors.New("message cannot be empty"))
	}

	usrCfg, err := GetUserConfig(cc.UserID)
	if err != nil {
		return nil, util.HandleError(err)
	}

	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.PrimaryProfileID)
	if err != nil {
		return nil, util.HandleError(err)
	}
	if profile == nil {
		return nil, util.HandleError(errors.New("model profile not found"))
	}

	embedding, err := cc.AddUserMessage(ctx, message)
	if err != nil {
		return nil, util.HandleError(err)
	}

	ollamaReq := models.ChatReq{
		Messages:       []models.ChatMessage{}, // Will be populated later
		ConversationID: &cc.ConversationID,
	}

	// Always set streaming to true to prevent timeouts
	ollamaReq.Stream = true
	ollamaReq.Options = profile.Parameters.ToMap()
	ollamaReq.Model = profile.ModelName

	var wg sync.WaitGroup

	// Try to extract content from any URLs in the user message
	wg.Add(1)
	go func(cc *ConversationContext) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
		defer cancel()
		results, err := recherche.ExtractUrlContentFromQuery(ctx, message)
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
	}(cc)

	// Get the user's intent to determine if we should perform a web search or retrieve memories
	intent, err := cc.DetectIntent(ctx, message)
	if err != nil {
		util.LogWarning("Error detecting intent", logrus.Fields{"error": err})
		// Non-critical error, we can continue without intent
	} else if intent != nil {
		// If the intent indicates a web search, get the web search results
		if intent.WebSearch {
			wg.Add(1)
			go func(cc *ConversationContext, q string) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
				defer cancel()
				// Inject the search results into the conversation context
				if err := cc.SearchAndInjectResults(ctx, q); err != nil {
					util.LogWarning("Error injecting web search results", logrus.Fields{"error": err})
					// Non-critical error, we can continue without web search results
				}
			}(cc, message)
		}

		if intent.Memory {
			wg.Add(1)
			go func(cc *ConversationContext, embedding [][]float32) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
				defer cancel()
				// Attempt to retrieve and inject relevant memories for the user's query
				if err := cc.RetrieveAndInjectMemories(ctx, embedding, nil, nil); err != nil {
					util.LogWarning("Error retrieving memories", logrus.Fields{"error": err})
					// Non-critical error, we can continue without memories
				}
			}(cc, embedding)
		}
	}

	// Wait for all goroutines to finish
	wg.Wait()

	// If embeddings are empty, log a warning and return an empty JSON object
	if len(embedding) == 0 {
		util.LogWarning("Empty embedding vector", logrus.Fields{"userMessage": message})
		return []byte("{}"), nil
	}

	util.LogDebug("Conversation context prepared for Ollama request", logrus.Fields{
		"conversationId": cc.ConversationID,
		"userId":         cc.UserID,
		"message_num":    len(cc.Messages),
		"search_results": len(cc.SearchResults),
		"memories":       len(cc.RetrievedMemories),
		"summaries":      len(cc.Summaries),
	})

	// Convert conversation context to Ollama format (includes summaries, memories, and web search results)
	return cc.ChainMessages(&ollamaReq)
}
