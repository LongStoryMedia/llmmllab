package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"maistro/config"
	"maistro/models"
	"maistro/util"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/sirupsen/logrus"
)

// Redis client for storage caching
var redisClient *redis.Client

// Cache key prefixes for different object types
const (
	messageKeyPrefix      = "proxyllama:message:"
	summaryKeyPrefix      = "proxyllama:summary:"
	conversationKeyPrefix = "proxyllama:conversation:"
	messagesListPrefix    = "proxyllama:messages:"
	summariesListPrefix   = "proxyllama:summaries:"
)

// Helper to construct cache keys
func cacheKey(prefix string, id interface{}) string {
	return fmt.Sprintf("%s%v", prefix, id)
}

// InitStorageCache initializes the Redis client for storage caching
func InitStorageCache() error {
	conf := config.GetConfig(nil)

	// Create Redis client
	// Parse Redis timeout values from string to time.Duration
	readTimeout := time.Duration(conf.Redis.ConnectTimeout) * time.Second
	writeTimeout := readTimeout
	dialTimeout := readTimeout

	redisClient = redis.NewClient(&redis.Options{
		Addr:         fmt.Sprintf("%s:%d", conf.Redis.Host, conf.Redis.Port),
		Password:     conf.Redis.Password,
		DB:           conf.Redis.DB,
		PoolSize:     conf.Redis.PoolSize,
		MinIdleConns: conf.Redis.MinIdleConnections,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
		DialTimeout:  dialTimeout,
	})

	// Test the connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := redisClient.Ping(ctx).Result()
	if err != nil {
		logrus.Errorf("Failed to connect to Redis: %v", err)
		redisClient = nil // Ensure client is nil if connection fails
		return fmt.Errorf("failed to connect to Redis: %w", err)
	}

	redisClient.FlushAll(ctx) // Clear all keys for a fresh start

	logrus.Infof("Storage Redis cache initialized at %s:%d", conf.Redis.Host, conf.Redis.Port)

	// Start a background routine to monitor Redis health
	go startRedisHealthCheck()

	return nil
}

// startRedisHealthCheck periodically checks the Redis connection
func startRedisHealthCheck() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		if redisClient == nil {
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		_, err := redisClient.Ping(ctx).Result()
		cancel()

		if err != nil {
			logrus.Errorf("Redis health check failed: %v", err)
		}
	}
}

// IsStorageCacheEnabled checks if Redis storage caching is enabled and working
func IsStorageCacheEnabled() bool {
	if redisClient == nil {
		return false
	}

	// Quick ping to verify connection
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	_, err := redisClient.Ping(ctx).Result()
	return err == nil
}

// ========== Message Cache Operations ==========

// GetMessageFromCache attempts to retrieve a message from cache
func GetMessageFromCache(ctx context.Context, messageID int) (*models.Message, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	key := cacheKey(messageKeyPrefix, messageID)
	data, err := redisClient.Get(ctx, key).Bytes()
	if err != nil {
		if err != redis.Nil {
			logrus.Errorf("Redis error retrieving message %d: %v", messageID, err)
		}
		return nil, false
	}

	var message models.Message
	if err := json.Unmarshal(data, &message); err != nil {
		logrus.Errorf("Error deserializing message from Redis: %v", err)
		return nil, false
	}

	return &message, true
}

// CacheMessage stores a message in cache
func CacheMessage(ctx context.Context, message *models.Message) error {
	if !IsStorageCacheEnabled() || message == nil {
		return nil
	}

	conf := config.GetConfig(nil)
	data, err := json.Marshal(message)
	if err != nil {
		return util.HandleError(fmt.Errorf("failed to marshal message: %w", err))
	}

	ttl := conf.Redis.MessageTtl

	key := cacheKey(messageKeyPrefix, message.ID)
	if err := redisClient.Set(ctx, key, data, time.Duration(ttl)*time.Second).Err(); err != nil {
		return util.HandleError(fmt.Errorf("failed to cache message: %w", err))
	}

	// Also cache the message ID in the conversation's message list (as a Redis list)
	conversationMessagesKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:messages", message.ConversationID))
	return redisClient.RPush(ctx, conversationMessagesKey, message.ID).Err()
}

// InvalidateMessageCache removes a message from cache
func InvalidateMessageCache(ctx context.Context, messageID int) {
	if !IsStorageCacheEnabled() {
		return
	}

	key := cacheKey(messageKeyPrefix, messageID)
	redisClient.Del(ctx, key)
}

// InvalidateConversationMessagesCache removes all messages for a conversation from cache
func InvalidateConversationMessagesCache(ctx context.Context, conversationID int) {
	if !IsStorageCacheEnabled() {
		return
	}

	// Remove the message list for this conversation
	messagesListKey := cacheKey(messagesListPrefix, conversationID)
	redisClient.Del(ctx, messagesListKey)

	// Find all message keys for this conversation
	pattern := fmt.Sprintf("%s*", messageKeyPrefix)
	iter := redisClient.Scan(ctx, 0, pattern, 100).Iterator()

	// This is inefficient but works for now - ideally we'd index by conversation
	for iter.Next(ctx) {
		key := iter.Val()
		data, err := redisClient.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var msg models.Message
		if err := json.Unmarshal(data, &msg); err != nil {
			continue
		}

		if msg.ConversationID == conversationID {
			redisClient.Del(ctx, key)
		}
	}
}

// GetMessagesByConversationIDFromCache tries to get all messages for a conversation from cache
func GetMessagesByConversationIDFromCache(ctx context.Context, conversationID int) ([]models.Message, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	// Try to get the complete message list first
	messagesListKey := cacheKey(messagesListPrefix, conversationID)
	data, err := redisClient.Get(ctx, messagesListKey).Bytes()
	if err == nil {
		var messages []models.Message
		if err := json.Unmarshal(data, &messages); err == nil {
			return messages, true
		}
	}

	// Fall back to getting by message IDs from Redis list
	messageIDsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:messages", conversationID))
	idStrs, err := redisClient.LRange(ctx, messageIDsKey, 0, -1).Result()
	if err != nil {
		return nil, false
	}

	var messageIDs []int
	for _, s := range idStrs {
		var id int
		if _, err := fmt.Sscanf(s, "%d", &id); err == nil {
			messageIDs = append(messageIDs, id)
		}
	}

	var messages []models.Message
	for _, msgID := range messageIDs {
		if msg, found := GetMessageFromCache(ctx, msgID); found {
			messages = append(messages, *msg)
		} else {
			return nil, false
		}
	}

	return messages, true
}

// CacheMessagesByConversationID caches all messages for a conversation
func CacheMessagesByConversationID(ctx context.Context, conversationID int, messages []models.Message) error {
	if !IsStorageCacheEnabled() || len(messages) == 0 {
		return nil
	}

	conf := config.GetConfig(nil)
	ttl := time.Duration(conf.Redis.MessageTtl) * time.Second

	// Cache the full message list
	messagesListKey := cacheKey(messagesListPrefix, conversationID)
	messagesData, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("failed to marshal messages: %w", err)
	}

	if err := redisClient.Set(ctx, messagesListKey, messagesData, ttl).Err(); err != nil {
		return err
	}

	// Also cache each individual message and track IDs
	var messageIDs []interface{}
	for i := range messages {
		messageIDs = append(messageIDs, messages[i].ID)
		if err := CacheMessage(ctx, &messages[i]); err != nil {
			logrus.Errorf("Error caching message %d: %v", messages[i].ID, err)
		}
	}

	// Store the message ID list as a Redis list (overwrite existing)
	messageIDsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:messages", conversationID))
	if err := redisClient.Del(ctx, messageIDsKey).Err(); err != nil {
		return err
	}
	if len(messageIDs) > 0 {
		if err := redisClient.RPush(ctx, messageIDsKey, messageIDs...).Err(); err != nil {
			return err
		}
	}
	if ttl > 0 {
		redisClient.Expire(ctx, messageIDsKey, ttl)
	}
	return nil
}

// ========== Summary Cache Operations ==========

// GetSummaryFromCache attempts to retrieve a summary from cache
func GetSummaryFromCache(ctx context.Context, summaryID int) (*models.Summary, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	key := cacheKey(summaryKeyPrefix, summaryID)
	data, err := redisClient.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}

	var summary models.Summary
	if err := json.Unmarshal(data, &summary); err != nil {
		logrus.Errorf("Error deserializing summary from Redis: %v", err)
		return nil, false
	}

	return &summary, true
}

// CacheSummary stores a summary in cache
func CacheSummary(ctx context.Context, summary *models.Summary) error {
	if !IsStorageCacheEnabled() || summary == nil {
		return nil
	}

	conf := config.GetConfig(nil)
	data, err := json.Marshal(summary)
	if err != nil {
		return fmt.Errorf("failed to marshal summary: %w", err)
	}

	key := cacheKey(summaryKeyPrefix, summary.ID)
	ttl := time.Duration(conf.Redis.SummaryTtl) * time.Second
	return redisClient.Set(ctx, key, data, ttl).Err()
}

// InvalidateSummaryCache removes a summary from cache
func InvalidateSummaryCache(ctx context.Context, summaryID int) {
	if !IsStorageCacheEnabled() {
		return
	}

	key := cacheKey(summaryKeyPrefix, summaryID)
	redisClient.Del(ctx, key)
}

// GetSummariesByConversationIDFromCache tries to get all summaries for a conversation from cache
func GetSummariesByConversationIDFromCache(ctx context.Context, conversationID int) ([]models.Summary, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	// Try to get the complete summaries list first
	summariesListKey := cacheKey(summariesListPrefix, conversationID)
	data, err := redisClient.Get(ctx, summariesListKey).Bytes()
	if err == nil {
		var summaries []models.Summary
		if err := json.Unmarshal(data, &summaries); err == nil {
			return summaries, true
		}
	}

	// Fall back to getting by summary IDs
	summaryIDsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:summaries", conversationID))

	// Check if we have the summary ID list
	summaryIDsStr, err := redisClient.Get(ctx, summaryIDsKey).Result()
	if err != nil {
		return nil, false
	}

	var summaryIDs []int
	if err := json.Unmarshal([]byte(summaryIDsStr), &summaryIDs); err != nil {
		return nil, false
	}

	// Now get each summary from cache
	var summaries []models.Summary
	for _, sumID := range summaryIDs {
		if sum, found := GetSummaryFromCache(ctx, sumID); found {
			summaries = append(summaries, *sum)
		} else {
			// If any summary is missing, return false to fetch all from DB
			return nil, false
		}
	}

	return summaries, true
}

// CacheSummariesByConversationID caches all summaries for a conversation
func CacheSummariesByConversationID(ctx context.Context, conversationID int, summaries []models.Summary) error {
	if !IsStorageCacheEnabled() || len(summaries) == 0 {
		return nil
	}

	conf := config.GetConfig(nil)

	// Cache the full summaries list
	summariesListKey := cacheKey(summariesListPrefix, conversationID)
	summariesData, err := json.Marshal(summaries)
	if err != nil {
		return fmt.Errorf("failed to marshal summaries: %w", err)
	}

	ttl := time.Duration(conf.Redis.SummaryTtl) * time.Second
	if err := redisClient.Set(ctx, summariesListKey, summariesData, ttl).Err(); err != nil {
		return err
	}

	// Also cache each individual summary
	var summaryIDs []int
	for i := range summaries {
		summaryIDs = append(summaryIDs, summaries[i].ID)
		if err := CacheSummary(ctx, &summaries[i]); err != nil {
			logrus.Errorf("Error caching summary %d: %v", summaries[i].ID, err)
		}
	}

	// Cache the summary ID list for this conversation
	summaryIDsData, err := json.Marshal(summaryIDs)
	if err != nil {
		return fmt.Errorf("failed to marshal summary IDs: %w", err)
	}

	summaryIDsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:summaries", conversationID))
	return redisClient.Set(ctx, summaryIDsKey, summaryIDsData, ttl).Err()
}

// InvalidateConversationSummariesCache removes all summaries for a conversation from cache
func InvalidateConversationSummariesCache(ctx context.Context, conversationID int) {
	if !IsStorageCacheEnabled() {
		return
	}

	// Remove the summary lists
	summariesListKey := cacheKey(summariesListPrefix, conversationID)
	summaryIDsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:summaries", conversationID))
	redisClient.Del(ctx, summariesListKey, summaryIDsKey)

	// Find all summary keys for this conversation (less efficient but still works)
	pattern := fmt.Sprintf("%s*", summaryKeyPrefix)
	iter := redisClient.Scan(ctx, 0, pattern, 100).Iterator()

	for iter.Next(ctx) {
		key := iter.Val()
		data, err := redisClient.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var sum models.Summary
		if err := json.Unmarshal(data, &sum); err != nil {
			continue
		}

		if sum.ConversationID == conversationID {
			redisClient.Del(ctx, key)
		}
	}
}

// ========== Conversation Cache Operations ==========

// GetConversationFromCache attempts to retrieve a conversation from cache
func GetConversationFromCache(ctx context.Context, conversationID int) (*models.Conversation, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	key := cacheKey(conversationKeyPrefix, conversationID)
	data, err := redisClient.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}

	var conversation models.Conversation
	if err := json.Unmarshal(data, &conversation); err != nil {
		logrus.Errorf("Error deserializing conversation from Redis: %v", err)
		return nil, false
	}

	return &conversation, true
}

// CacheConversation stores a conversation in cache
func CacheConversation(ctx context.Context, conversation *models.Conversation) error {
	if !IsStorageCacheEnabled() || conversation == nil {
		return nil
	}

	conf := config.GetConfig(nil)
	data, err := json.Marshal(conversation)
	if err != nil {
		return fmt.Errorf("failed to marshal conversation: %w", err)
	}

	key := cacheKey(conversationKeyPrefix, conversation.ID)
	ttl := time.Duration(conf.Redis.ConversationTtl) * time.Second
	return redisClient.Set(ctx, key, data, ttl).Err()
}

// InvalidateConversationCache removes a conversation from cache
func InvalidateConversationCache(ctx context.Context, conversationID int) {
	if !IsStorageCacheEnabled() {
		return
	}

	key := cacheKey(conversationKeyPrefix, conversationID)
	redisClient.Del(ctx, key)

	// Also invalidate message and summary lists
	conversationMessagesKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:messages", conversationID))
	conversationSummariesKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%d:summaries", conversationID))
	redisClient.Del(ctx, conversationMessagesKey, conversationSummariesKey)
}

// GetConversationsByUserIDFromCache attempts to retrieve user's conversations from cache
func GetConversationsByUserIDFromCache(ctx context.Context, userID string) ([]models.Conversation, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	// We'll use a special key for storing the list of conversation IDs for a user
	userConversationsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%s:conversations", userID))

	// Check if we have the conversation ID list
	conversationIDsStr, err := redisClient.Get(ctx, userConversationsKey).Result()
	if err != nil {
		return nil, false
	}

	var conversationIDs []int
	if err := json.Unmarshal([]byte(conversationIDsStr), &conversationIDs); err != nil {
		return nil, false
	}

	// Now get each conversation from cache
	var conversations []models.Conversation
	for _, convID := range conversationIDs {
		if conv, found := GetConversationFromCache(ctx, convID); found {
			conversations = append(conversations, *conv)
		} else {
			// If any conversation is missing, return false to fetch all from DB
			return nil, false
		}
	}

	return conversations, len(conversations) > 0
}

// CacheConversationsByUserID caches all conversations for a user
func CacheConversationsByUserID(ctx context.Context, userID string, conversations []models.Conversation) error {
	if !IsStorageCacheEnabled() || len(conversations) == 0 {
		return nil
	}

	conf := config.GetConfig(nil)

	// First cache each individual conversation
	var conversationIDs []int
	for i := range conversations {
		conversationIDs = append(conversationIDs, conversations[i].ID)
		if err := CacheConversation(ctx, &conversations[i]); err != nil {
			logrus.Errorf("Error caching conversation %d: %v", conversations[i].ID, err)
		}
	}

	// Then cache the conversation ID list for this user
	conversationIDsData, err := json.Marshal(conversationIDs)
	if err != nil {
		return fmt.Errorf("failed to marshal conversation IDs: %w", err)
	}

	userConversationsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%s:conversations", userID))
	ttl := time.Duration(conf.Redis.ConversationTtl) * time.Second
	return redisClient.Set(ctx, userConversationsKey, conversationIDsData, ttl).Err()
}

// InvalidateUserConversationsCache removes all conversation records for a user from cache
func InvalidateUserConversationsCache(ctx context.Context, userID string) {
	if !IsStorageCacheEnabled() {
		return
	}

	// Remove the conversation ID list for this user
	userConversationsKey := cacheKey(conversationKeyPrefix, fmt.Sprintf("%s:conversations", userID))

	// Get the conversation IDs first so we can invalidate each one
	conversationIDsStr, err := redisClient.Get(ctx, userConversationsKey).Result()
	if err == nil {
		var conversationIDs []int
		if err := json.Unmarshal([]byte(conversationIDsStr), &conversationIDs); err == nil {
			// Invalidate each conversation
			for _, convID := range conversationIDs {
				InvalidateConversationCache(ctx, convID)
			}
		}
	}

	// Remove the user's conversation list
	redisClient.Del(ctx, userConversationsKey)
}

// CloseRedisCache shuts down the Redis client when the application exits
func CloseRedisCache() error {
	if redisClient != nil {
		return redisClient.Close()
	}
	return nil
}
