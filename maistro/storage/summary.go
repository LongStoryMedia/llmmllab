package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"maistro/models"
	"maistro/util"
)

type summaryStore struct{}

// CreateSummary adds a new summary for a conversation
func (ss *summaryStore) CreateSummary(ctx context.Context, conversationID int, content string, level int, sourceIDs []int) (int, error) {
	// Convert source IDs to JSON
	sourceIDsJSON, err := json.Marshal(sourceIDs)
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to marshal source IDs: %w", err))
	}
	sanitizedContent := util.RemoveThinkTags(content)

	var summaryID int
	err = Pool.QueryRow(ctx, GetQuery("summary.create_summary"),
		conversationID, sanitizedContent, level, sourceIDsJSON).Scan(&summaryID)
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to create summary: %w", err))
	}

	// Invalidate the conversation summaries cache
	InvalidateConversationSummariesCache(ctx, conversationID)

	return summaryID, nil
}

// GetSummariesForConversation gets all summaries for a conversation
func (ss *summaryStore) GetSummariesForConversation(ctx context.Context, conversationID int) ([]models.Summary, error) {
	// Try to get from cache first
	if summaries, found := GetSummariesByConversationIDFromCache(ctx, conversationID); found {
		return summaries, nil
	}

	// Not in cache, get from database
	rows, err := Pool.Query(ctx, GetQuery("summary.get_summaries_for_conversation"), conversationID)
	if err != nil {
		return nil, fmt.Errorf("failed to query summaries for conversation %d: %w", conversationID, err)
	}
	defer rows.Close()

	var summaries []models.Summary
	for rows.Next() {
		var s models.Summary
		var sourceIDsJSON []byte

		if err := rows.Scan(&s.ID, &s.ConversationID, &s.Content, &s.Level, &sourceIDsJSON, &s.CreatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan summary row: %w", err)
		}

		// Parse source IDs JSON
		if err := json.Unmarshal(sourceIDsJSON, &s.SourceIds); err != nil {
			// If JSON parsing fails, initialize an empty slice
			s.SourceIds = make([]int, 0)
			util.LogWarning(fmt.Sprintf("Failed to parse source IDs JSON for summary %d: %v", s.ID, err))
		}

		summaries = append(summaries, s)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating summary rows: %w", err)
	}

	// Cache the results
	if err := CacheSummariesByConversationID(ctx, conversationID, summaries); err != nil {
		// Just log, don't fail on cache error
		util.LogWarning(fmt.Sprintf("Failed to cache summaries for conversation %d: %v", conversationID, err))
	}

	return summaries, nil
}

// GetRecentSummaries gets recent summaries for a conversation at a specific level
func (ss *summaryStore) GetRecentSummaries(ctx context.Context, conversationID int, level int, limit int) ([]models.Summary, error) {
	// This one doesn't use cache because it's more dynamic with limit parameter

	rows, err := Pool.Query(ctx, GetQuery("summary.get_recent_summaries"), conversationID, level, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent summaries: %w", err)
	}
	defer rows.Close()

	var summaries []models.Summary
	for rows.Next() {
		var s models.Summary
		var sourceIDsJSON []byte

		if err := rows.Scan(&s.ID, &s.ConversationID, &s.Content, &s.Level, &sourceIDsJSON, &s.CreatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan recent summary row: %w", err)
		}

		// Parse source IDs JSON
		if err := json.Unmarshal(sourceIDsJSON, &s.SourceIds); err != nil {
			// If JSON parsing fails, initialize an empty map
			s.SourceIds = make([]int, 0)
			util.LogWarning(fmt.Sprintf("Failed to parse source IDs JSON for recent summary %d: %v", s.ID, err))
		}

		summaries = append(summaries, s)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating recent summary rows: %w", err)
	}

	return summaries, nil
}

// DeleteSummariesForConversation deletes all summaries for a conversation
func (ss *summaryStore) DeleteSummariesForConversation(ctx context.Context, conversationID int) error {
	_, err := Pool.Exec(ctx, GetQuery("summary.delete_summaries"), conversationID)
	if err != nil {
		return fmt.Errorf("failed to delete summaries: %w", err)
	}

	// Invalidate the conversation summaries cache
	InvalidateConversationSummariesCache(ctx, conversationID)

	return nil
}

// GetSummary gets a single summary by ID
func (ss *summaryStore) GetSummary(ctx context.Context, summaryID int) (*models.Summary, error) {
	var s models.Summary
	var sourceIDsJSON []byte

	err := Pool.QueryRow(ctx, GetQuery("summary.get_summary"), summaryID).Scan(
		&s.ID, &s.ConversationID, &s.Content, &s.Level, &sourceIDsJSON, &s.CreatedAt)
	if err != nil {
		return nil, fmt.Errorf("failed to get summary: %w", err)
	}

	// Parse source IDs JSON
	if err := json.Unmarshal(sourceIDsJSON, &s.SourceIds); err != nil {
		// If JSON parsing fails, initialize an empty map
		s.SourceIds = make([]int, 0)
		util.LogWarning(fmt.Sprintf("Failed to parse source IDs JSON for summary %d: %v", s.ID, err))
	}

	return &s, nil
}
