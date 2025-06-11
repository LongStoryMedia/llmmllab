package context

import (
	"context"
	"errors"
	"fmt"
	"maistro/models"
	"maistro/storage"
	"maistro/util"
	"slices"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// RetrievedMemory represents a message that was retrieved based on vector similarity
type RetrievedMemory struct {
	Message    models.Memory
	Similarity float32
}

// EnhanceRequestWithRAG adds relevant memories to the request based on the latest user query
func (cc *ConversationContext) EnhanceRequestWithRAG(ctx context.Context, req *models.ChatReq) error {
	// Find the latest user message to use as query
	var latestUserMessage string
	for i := len(cc.Messages) - 1; i >= 0; i-- {
		if cc.Messages[i].Role == "user" {
			latestUserMessage = cc.Messages[i].Content
			break
		}
	}

	if latestUserMessage == "" {
		return util.HandleError(fmt.Errorf("no user message found in request"))
	}

	if len(cc.RetrievedMemories) == 0 && len(cc.SearchResults) == 0 {
		util.LogInfo("No relevant memories or search results found, skipping RAG enhancement")
		return nil // No relevant memories found, continue with original request
	}

	// Create a new request with memories inserted at the right position
	var enhancedMessages []models.ChatMessage
	var relevantMemories []models.ChatMessage
	var searchResults []models.ChatMessage
	// If there are search results, format them as system messages
	if len(cc.SearchResults) > 0 {
		util.LogInfo("Adding search results to request", logrus.Fields{"count": len(cc.SearchResults)})
		for _, result := range cc.SearchResults {
			msg := models.ChatMessage{Role: "system"}
			for _, content := range result.Contents {
				var preamble string
				if result.IsFromURLInUserQuery {
					preamble = "Here is the content from the user provided URL"
				} else {
					preamble = "Here is a relevant finding from a web search at"
				}
				msg.Content = fmt.Sprintf(
					"%s, %s:\n%s\nThis may help answer the current query.",
					preamble,
					content.URL,
					content.Content,
				)
				if result.IsFromURLInUserQuery {
					searchResults = append(searchResults, msg) // If the result is from a URL in the user query, add it to the end of searchResults
				} else {
					searchResults = append([]models.ChatMessage{msg}, searchResults...) // Otherwise, add it to the front
				}
			}
		}
	}
	// Add a system message to explain the search results
	if len(cc.RetrievedMemories) > 0 {
		for _, mem := range cc.RetrievedMemories {
			msg := models.ChatMessage{Role: "system"}
			if mem.Source == models.MemorySourceMessage {
				if len(mem.Fragments) != 2 {
					util.LogWarning("Message memory does not represent a full interaction. Skipping", logrus.Fields{"message_number": len(mem.Fragments)})
					continue
				}
				// Format the message content
				msg.Content = fmt.Sprintf(
					"Similar interaction from %s:\nUser:\n%s\nAssistant:\n%s\nThis interaction may help answer the current query.",
					mem.CreatedAt.Format(time.RFC3339Nano),
					util.SanitizeText(mem.Fragments[0].Content),
					mem.Fragments[1].Content,
				)
			} else if mem.Source == models.MemorySourceSummary {
				// Format the summary content
				msg.Content = fmt.Sprintf(
					"Here is a summary of a relevant conversation from %s:\n%s\nThis summary may help answer the current query.",
					mem.CreatedAt.Format(time.RFC3339Nano),
					util.SanitizeText(mem.Fragments[0].Content),
				)
			}

			relevantMemories = append([]models.ChatMessage{msg}, relevantMemories...) // Add to the front of relevantMemories as most recent and similar memories come in first
		}
	}

	// First add any system messages (these should come first)
	systemMessagesCount := 0
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			enhancedMessages = append(enhancedMessages, msg)
			systemMessagesCount++
		}
	}

	// Add a system message to explain the memories
	enhancedMessages = append(enhancedMessages, models.ChatMessage{
		Role:    "system",
		Content: fmt.Sprintf("Here are %d relevant memories from previous conversations that may help answer the current query:", len(relevantMemories)),
	})

	// Add the retrieved memories
	enhancedMessages = append(enhancedMessages, relevantMemories...)

	enhancedMessages = append(enhancedMessages, searchResults...) // Add search results after memories
	// TODO: consider using summarization for search results

	// Add another system message to separate memories and search results from the current conversation
	enhancedMessages = append(enhancedMessages, models.ChatMessage{
		Role:    "system",
		Content: "Now continuing with the current conversation:",
	})

	// Add the non-system messages from the original request
	for i, msg := range req.Messages {
		if i >= systemMessagesCount {
			enhancedMessages = append(enhancedMessages, msg)
		}
	}

	// Replace messages in the request
	req.Messages = enhancedMessages

	util.LogInfo("Enhanced request with relevant memories", logrus.Fields{"count": len(relevantMemories)})
	return nil
}

// RetrieveAndInjectMemories retrieves relevant memories based on the current user query
func (cc *ConversationContext) RetrieveAndInjectMemories(ctx context.Context, queryEmbeddings [][]float32, startDate, endDate *time.Time) error {
	// Clear any previous memories
	cc.RetrievedMemories = nil

	// Get user-specific configuration
	userConfig, err := GetUserConfig(cc.UserID)
	if err != nil {
		return util.HandleError(err)
	}

	// Set a search limit from user config
	limit := userConfig.Memory.Limit
	if limit <= 0 {
		limit = 5
	}

	// First try vector similarity search if RAG is enabled
	if userConfig.Memory.Enabled {
		util.LogInfo("Performing semantic search for memories")
		if len(queryEmbeddings) > 0 {
			var wg sync.WaitGroup

			// Channel for current conversation results
			results := make(chan []models.Memory, 1)
			errs := make(chan error, 1)
			defer close(results)
			defer close(errs)
			threshold := userConfig.Memory.SimilarityThreshold

			var userID string
			var conversationID int

			if !userConfig.Memory.EnableCrossConversation {
				conversationID = cc.ConversationID
			}

			if !userConfig.Memory.EnableCrossUser {
				userID = cc.UserID
			}

			// memory search
			wg.Add(1)
			go func(cid, limit int, embeddings [][]float32, threshold float64, userID *string, conversationID *int, startDate, endDate *time.Time) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
				defer cancel()
				ccr := []models.Memory{}

				var errorStr string
				var memoryError error
				for _, emb := range embeddings {
					similarMessages, err := storage.SearchSimilarity(ctx, emb, threshold, limit, userID, conversationID, startDate, endDate)
					if err != nil {
						errorStr += err.Error() + " "
					}
					for _, msg := range similarMessages {
						ccr = append(ccr, msg)
						if len(ccr) >= limit {
							break
						}
					}
					if errorStr != "" {
						memoryError = errors.New(errorStr)
					}
				}
				results <- ccr
				errs <- memoryError
			}(cc.ConversationID, limit, queryEmbeddings, threshold, &userID, &conversationID, startDate, endDate)

			// Wait for all started goroutines
			wg.Wait()

			// Collect results
			msgs := <-results
			err := <-errs

			if err != nil {
				return util.HandleError(err)
			}

			util.LogInfo(fmt.Sprintf("Found %v semantically similar messages in current conversation", len(msgs)))
			// 1. First add current conversation messages
			if len(msgs) > 0 {
				cc.appendMemoriesToContext(msgs)
			}
		}
	}

	return nil
}

func (cc *ConversationContext) appendMemoriesToContext(memories []models.Memory) {
	// Add retrieved memories to the context
	for _, mem := range memories {
		if slices.ContainsFunc(cc.RetrievedMemories, func(m models.Memory) bool {
			return m.Source == mem.Source && m.SourceID == mem.SourceID
		}) {
			util.LogInfo("Memory already exists in context, skipping", logrus.Fields{"source_id": mem.SourceID, "source": mem.Source})
			continue // Skip if memory already exists
		}
		cc.RetrievedMemories = append(cc.RetrievedMemories, mem)
	}
}
