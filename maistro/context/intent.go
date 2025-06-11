package context

import (
	"context"
	"strings"

	"maistro/util" // replace with your actual import path for util package
)

type Intent struct {
	WebSearch    bool `json:"web_search"`
	Memory       bool `json:"memory"`
	DeepResearch bool `json:"deep_research"`
}

func (cc *ConversationContext) DetectIntent(ctx context.Context, query string) (*Intent, error) {
	// Check if the query is empty
	if strings.TrimSpace(query) == "" {
		util.LogWarning("Empty query detected")
		return nil, nil
	}
	cfg, err := GetUserConfig(cc.UserID)
	if err != nil {
		util.HandleError(err)
		return nil, err
	}

	intent := Intent{
		WebSearch:    false,
		Memory:       false,
		DeepResearch: false,
	}

	if cfg.WebSearch.Enabled {
		// Check if the query should trigger a web search only if web search is enabled
		intent.WebSearch = shouldSearchWeb(query)
	}

	if cfg.Memory.AlwaysRetrieve {
		intent.Memory = true
	} else if cfg.Memory.Enabled {
		intent.Memory = shouldRetrieveMemories(query)
	}

	return &intent, nil
}

// shouldSearchWeb determines if a query likely requires web search
func shouldSearchWeb(query string) bool {
	// Check for explicit web search indicators
	lowerQuery := strings.ToLower(query)
	explicitIndicators := []string{
		"search", "google", "look up", "find information", "search for",
		"what is the latest", "recent news", "current", "today's",
		"latest update", "website", "webpage", "url", "link",
		"http://", "https://", "www.", "online", "internet",
	}

	for _, indicator := range explicitIndicators {
		if strings.Contains(lowerQuery, indicator) {
			return true
		}
	}

	// Check for question formats that likely need external information
	questionIndicators := []string{
		"what is", "who is", "where is", "when did", "how does",
		"why does", "can you find", "what are", "is there",
		"tell me about", "explain", "define", "summarize",
	}

	for _, indicator := range questionIndicators {
		if strings.HasPrefix(lowerQuery, indicator) {
			return true
		}
	}

	// Check for date/time-sensitive queries
	timeIndicators := []string{
		"today", "yesterday", "this week", "this month", "this year",
		"latest", "newest", "recent", "current", "update",
	}

	for _, indicator := range timeIndicators {
		if strings.Contains(lowerQuery, indicator) {
			return true
		}
	}

	// Check for URLs in the query
	if strings.Contains(query, "http://") || strings.Contains(query, "https://") {
		return true
	}

	return false
}

// shouldRetrieveMemories determines if a query likely needs memory retrieval
func shouldRetrieveMemories(query string) bool {
	// Convert query to lowercase for case-insensitive matching
	lowercaseQuery := strings.ToLower(query)

	// List of keywords and phrases that suggest the user is asking about past information
	memoryTriggers := []string{
		"remember", "recall", "previous", "earlier", "before", "last time",
		"you said", "mentioned", "told me", "yesterday", "last week",
		"forgot", "remind me", "i asked", "we discussed", "we talked about",
		"what did i", "what did you", "did i tell", "did you tell",
	}

	// Check if any memory trigger is in the query
	for _, trigger := range memoryTriggers {
		if strings.Contains(lowercaseQuery, trigger) {
			util.LogInfo("Memory retrieval triggered by keyword", map[string]any{"trigger": trigger})
			return true
		}
	}

	// Question patterns that often benefit from memory retrieval
	questionPatterns := []string{
		"what was", "who was", "where was", "when was", "how was",
		"what were", "who were", "where were", "when were", "how were",
		"what did", "who did", "where did", "when did", "how did",
	}

	// Check for question patterns
	for _, pattern := range questionPatterns {
		if strings.Contains(lowercaseQuery, pattern) {
			// Check if the question appears to be about past interactions
			if strings.Contains(lowercaseQuery, "you") || strings.Contains(lowercaseQuery, "we") || strings.Contains(lowercaseQuery, "i") {
				util.LogInfo("Memory retrieval triggered by question pattern", map[string]any{"pattern": pattern})
				return true
			}
		}
	}

	return false
}
