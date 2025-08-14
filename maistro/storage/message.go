package storage

import (
	"context"
	"fmt"
	"maistro/models"
	"maistro/util"
)

type messageStore struct{}

// AddMessage adds a message to a conversation
func (ms *messageStore) AddMessage(ctx context.Context, message *models.Message, usrCfg *models.UserConfig) (int, error) {
	// Check if Pool is initialized
	if Pool == nil {
		return 0, util.HandleError(fmt.Errorf("database connection pool is not initialized (Pool is nil)"))
	}
	if message == nil {
		return 0, util.HandleError(fmt.Errorf("message cannot be nil"))
	}
	if message.ConversationID <= 0 {
		return 0, util.HandleError(fmt.Errorf("conversation ID must be greater than 0"))
	}
	if message.Role == "" {
		return 0, util.HandleError(fmt.Errorf("message role cannot be empty"))
	}
	if len(message.Content) == 0 {
		return 0, util.HandleError(fmt.Errorf("message content cannot be empty"))
	}

	// Start a transaction for atomicity
	tx, err := Pool.Begin(ctx)
	if err != nil {
		return 0, util.HandleError(err)
	}
	// Use defer with a named error return to ensure we correctly handle transaction state
	defer func() {
		if err != nil {
			tx.Rollback(ctx) // rollback on error
		}
	}()

	var messageID int
	// Use the SQL query from our loader
	err = tx.QueryRow(ctx, GetQuery("message.add_message"), message.ConversationID, message.Role, message.Content).Scan(&messageID)
	if err != nil {
		return 0, util.HandleError(err)
	}

	for _, c := range message.Content {
		err = tx.QueryRow(ctx, GetQuery("message.add_content"), messageID, c.Type, c.Text, c.URL).Scan(&messageID)
		if err != nil {
			return 0, util.HandleError(err)
		}
	}

	// Update the conversation's updated_at timestamp
	_, err = tx.Exec(ctx, GetQuery("conversation.update_conversation"), message.ConversationID)
	if err != nil {
		return 0, util.HandleError(err)
	}

	// Commit the transaction
	if err = tx.Commit(ctx); err != nil {
		return 0, util.HandleError(err)
	}

	// Invalidate the conversation's message cache
	InvalidateConversationMessagesCache(ctx, message.ConversationID)

	// Cache the new message
	if err := CacheMessage(ctx, message); err != nil {
		// Log but don't fail on cache error
		util.LogWarning("Failed to cache message", nil)
		// We'll just have a cache miss next time
	}

	return messageID, nil
}

// GetMessage gets a single message by ID
func (ms *messageStore) GetMessage(ctx context.Context, messageID int) (*models.Message, error) {
	// Try to get from cache first
	if msg, found := GetMessageFromCache(ctx, messageID); found {
		return msg, nil
	}

	// Not in cache, get from database using the query from our loader
	var msg models.Message
	err := Pool.QueryRow(ctx, GetQuery("message.get_message"), messageID).Scan(
		&msg.ID, &msg.ConversationID, &msg.Role, &msg.Content, &msg.CreatedAt)
	if err != nil {
		return nil, err
	}

	// Cache for future use
	if err := CacheMessage(ctx, &msg); err != nil {
		// Just log, don't fail on cache error
		util.LogWarning("Failed to cache message", nil)
	}

	return &msg, nil
}

// GetConversationHistory retrieves all messages for a conversation
func (ms *messageStore) GetConversationHistory(ctx context.Context, conversationID int) ([]models.Message, error) {
	// Try to get from cache first
	if messages, found := GetMessagesByConversationIDFromCache(ctx, conversationID); found && len(messages) > 0 {
		util.LogDebug("Cache hit for conversation messages", nil)
		return messages, nil
	}

	// Not in cache, get from database using the query from our loader
	rows, err := Pool.Query(ctx, GetQuery("message.get_conversation_history"), conversationID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []models.Message
	for rows.Next() {
		var msg models.Message
		if err := rows.Scan(&msg.ID, &msg.ConversationID, &msg.Role, &msg.Content, &msg.CreatedAt); err != nil {
			return nil, err
		}
		messages = append(messages, msg)
	}

	// Check for errors from iterating over rows
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Cache the message list
	if err := CacheMessagesByConversationID(ctx, conversationID, messages); err != nil {
		// Just log, don't fail on cache error
		util.LogWarning("Failed to cache messages for conversation", nil)
	}

	return messages, nil
}

// DeleteMessage deletes a message by ID
func (ms *messageStore) DeleteMessage(ctx context.Context, messageID int) error {
	// Start a transaction for atomicity
	tx, err := Pool.Begin(ctx)
	if err != nil {
		return util.HandleError(err)
	}
	defer func() {
		if err != nil {
			tx.Rollback(ctx) // rollback on error
		}
	}()

	// Delete the message using the query from our loader
	_, err = tx.Exec(ctx, GetQuery("message.delete_message"), messageID)
	if err != nil {
		return util.HandleError(err)
	}

	// Commit the transaction
	if err = tx.Commit(ctx); err != nil {
		return util.HandleError(err)
	}

	// Invalidate the cache
	InvalidateMessageCache(ctx, messageID)

	return nil
}
