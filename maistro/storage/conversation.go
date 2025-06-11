package storage

import (
	"context"
	"fmt"
	"maistro/models"
	"time"
)

type conversationStore struct{}

// CreateConversation starts a new conversation for a user
func (cs *conversationStore) CreateConversation(ctx context.Context, userID string, title string) (int, error) {
	// Ensure the user exists
	if err := EnsureUser(ctx, userID); err != nil {
		return 0, err
	}

	var conversationID int
	err := Pool.QueryRow(ctx, GetQuery("conversation.create_conversation"), userID, title).Scan(&conversationID)
	if err != nil {
		return 0, err
	}

	// Create a conversation object for caching
	conversation := &models.Conversation{
		ID:        conversationID,
		UserID:    userID,
		Title:     title,
		CreatedAt: time.Now(), // Approximate time until we fetch from DB
		UpdatedAt: time.Now(),
	}

	// Cache the new conversation
	if err := CacheConversation(ctx, conversation); err != nil {
		// Log but don't fail on cache error
	}

	// Invalidate the user's conversations list cache
	InvalidateUserConversationsCache(ctx, userID)

	return conversationID, nil
}

// GetUserConversations gets all conversations for a user, ordered by most recent
func (cs *conversationStore) GetUserConversations(ctx context.Context, userID string) ([]models.Conversation, error) {
	// Try to get from cache first
	if conversations, found := GetConversationsByUserIDFromCache(ctx, userID); found {
		return conversations, nil
	}

	// Not in cache, get from database using the query from our loader
	rows, err := Pool.Query(ctx, GetQuery("conversation.list_user_conversations"), userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var conversations []models.Conversation
	for rows.Next() {
		var conv models.Conversation
		if err := rows.Scan(&conv.ID, &conv.UserID, &conv.Title, &conv.CreatedAt, &conv.UpdatedAt); err != nil {
			return nil, err
		}
		conversations = append(conversations, conv)
	}

	// Check for errors from iterating over rows
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Cache the conversations list
	if err := CacheConversationsByUserID(ctx, userID, conversations); err != nil {
		// Just log, don't fail on cache error
	}

	return conversations, nil
}

// GetConversation retrieves a single conversation by ID
func (cs *conversationStore) GetConversation(ctx context.Context, conversationID int) (*models.Conversation, error) {
	// Try to get from cache first
	if conversation, found := GetConversationFromCache(ctx, conversationID); found {
		return conversation, nil
	}

	// Not in cache, get from database
	var conv models.Conversation
	err := Pool.QueryRow(ctx, GetQuery("conversation.get_conversation"), conversationID).Scan(
		&conv.ID, &conv.UserID, &conv.Title, &conv.CreatedAt, &conv.UpdatedAt,
	)

	if err != nil {
		return nil, err
	}

	// Cache for future use
	if err := CacheConversation(ctx, &conv); err != nil {
		// Just log, don't fail on cache error
	}

	return &conv, nil
}

// UpdateConversationTitle updates the title of a conversation
func (cs *conversationStore) UpdateConversationTitle(ctx context.Context, conversationID int, title string) error {
	_, err := Pool.Exec(ctx, GetQuery("conversation.update_title"), title, conversationID)
	if err != nil {
		return err
	}

	// If successful, invalidate the conversation cache
	InvalidateConversationCache(ctx, conversationID)

	// Also invalidate any user conversations lists that might include this conversation
	// Get the conversation to find the userID
	conversation, err := cs.GetConversation(ctx, conversationID)
	if err == nil {
		// We got the conversation after the update, so let's invalidate the user's list
		InvalidateUserConversationsCache(ctx, conversation.UserID)
	}

	return nil
}

// DeleteConversation deletes a conversation and all its messages using transaction
func (cs *conversationStore) DeleteConversation(ctx context.Context, conversationID int) error {
	// Get the conversation to find the userID before deletion
	conversation, err := cs.GetConversation(ctx, conversationID)
	if err != nil {
		return fmt.Errorf("failed to get conversation before deletion: %w", err)
	}

	userID := conversation.UserID

	// Start a transaction for atomicity
	tx, err := Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	// Defer a rollback in case anything fails
	defer tx.Rollback(ctx)

	// Delete the conversation (triggers will handle dependent records)
	_, err = tx.Exec(ctx, GetQuery("conversation.delete_conversation"), conversationID)
	if err != nil {
		return fmt.Errorf("failed to delete conversation: %w", err)
	}

	// Commit the transaction
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	// Invalidate caches
	InvalidateConversationCache(ctx, conversationID)
	InvalidateConversationMessagesCache(ctx, conversationID)
	InvalidateConversationSummariesCache(ctx, conversationID)
	InvalidateUserConversationsCache(ctx, userID)

	return nil
}
