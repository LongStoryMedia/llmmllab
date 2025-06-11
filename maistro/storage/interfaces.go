// Package storage defines interfaces for the data layer abstraction
package storage

import (
	"context"
	"maistro/config"
	"maistro/models"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// MessageStore abstracts message-related operations
// (Add more methods as needed)
type MessageStore interface {
	AddMessage(ctx context.Context, conversationID int, role, content string, usrCfg *config.UserConfig) (int, error)
	GetMessage(ctx context.Context, messageID int) (*models.Message, error)
	GetConversationHistory(ctx context.Context, conversationID int) ([]models.Message, error)
}

// ConversationStore abstracts conversation-related operations
type ConversationStore interface {
	CreateConversation(ctx context.Context, userID string, title string) (int, error)
	GetUserConversations(ctx context.Context, userID string) ([]models.Conversation, error)
	GetConversation(ctx context.Context, conversationID int) (*models.Conversation, error)
	UpdateConversationTitle(ctx context.Context, conversationID int, title string) error
	DeleteConversation(ctx context.Context, conversationID int) error
}

// SummaryStore abstracts summary-related operations
type SummaryStore interface {
	CreateSummary(ctx context.Context, conversationID int, content string, level int, sourceIDs []int) (int, error)
	GetSummariesForConversation(ctx context.Context, conversationID int) ([]models.Summary, error)
	GetRecentSummaries(ctx context.Context, conversationID int, level int, limit int) ([]models.Summary, error)
	DeleteSummariesForConversation(ctx context.Context, conversationID int) error
	GetSummary(ctx context.Context, summaryID int) (*models.Summary, error)
}

// ModelProfileStore abstracts model profile operations
type ModelProfileStore interface {
	CreateModelProfile(ctx context.Context, profile *models.ModelProfile) (uuid.UUID, error)
	GetModelProfile(ctx context.Context, profileID uuid.UUID) (*models.ModelProfile, error)
	UpdateModelProfile(ctx context.Context, profile *models.ModelProfile) error
	DeleteModelProfile(ctx context.Context, profileID uuid.UUID) error
	ListModelProfilesByUser(ctx context.Context, userID string) ([]*models.ModelProfile, error)
}

// ResearchTaskStore abstracts research task operations
type ResearchTaskStore interface {
	SaveResearchTask(ctx context.Context, userID, query string, conversationID *int) (int, error)
	UpdateTaskStatus(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error)
	UpdateTask(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error)
	StoreResearchPlan(ctx context.Context, taskID int, plan *models.ResearchPlan) (time.Time, error)
	StoreFinalResult(ctx context.Context, taskID int, result *models.ResearchQuestionResult) (time.Time, error)
	SaveSubtask(ctx context.Context, subtask *models.ResearchSubtask) (int, error)
	UpdateSubtaskStatus(ctx context.Context, taskID, questionID int, status string, errorMsg *string) (int, time.Time, error)
	StoreGatheredInfo(ctx context.Context, taskID, questionID int, gatheredInfo []string, sources []string) (time.Time, error)
	StoreSynthesizedAnswer(ctx context.Context, taskID, questionID int, answer string) (time.Time, error)
	GetTaskByID(ctx context.Context, taskID int) (*models.ResearchTask, error)
	ListTasksByUserID(ctx context.Context, userID string, limit, offset int) ([]models.ResearchTask, error)
	GetSubtasksForTask(ctx context.Context, taskID string) ([]models.ResearchSubtask, error)
}

// MemoryStore abstracts memory-related operations
type MemoryStore interface {
	InitMemorySchema(ctx context.Context) error
	StoreMemory(ctx context.Context, userID, source, role string, sourceID int, embeddings [][]float32) error
	StoreMemoryWithTx(ctx context.Context, userID, source, role string, sourceID int, embeddings [][]float32, tx pgx.Tx) error
	DeleteMemory(ctx context.Context, id, userID string) error
	DeleteAllUserMemories(ctx context.Context, userID string) error
}

// UserConfigStore abstracts user config operations
type UserConfigStore interface {
	GetUserConfig(ctx context.Context, userID string) (*config.UserConfig, error)
	UpdateUserConfig(ctx context.Context, userID string, cfg *config.UserConfig) error
	GetAllUsers(ctx context.Context) ([]models.User, error)
}

var (
	// MessageStoreInstance is the global instance of the message store
	MessageStoreInstance MessageStore
	// ConversationStoreInstance is the global instance of the conversation store
	ConversationStoreInstance ConversationStore
	// SummaryStoreInstance is the global instance of the summary store
	SummaryStoreInstance SummaryStore
	// ModelProfileStoreInstance is the global instance of the model profile store
	ModelProfileStoreInstance ModelProfileStore
	// ResearchTaskStoreInstance is the global instance of the research task store
	ResearchTaskStoreInstance ResearchTaskStore
	// MemoryStoreInstance is the global instance of the memory store
	MemoryStoreInstance MemoryStore
	// UserConfigStoreInstance is the global instance of the user config store
	UserConfigStoreInstance UserConfigStore
)

// InitializeStorage initializes the global storage instances
func InitializeStorage() error {
	if MessageStoreInstance == nil {
		MessageStoreInstance = &messageStore{}
	}
	if MemoryStoreInstance == nil {
		MemoryStoreInstance = &memoryStore{}
	}
	if ConversationStoreInstance == nil {
		ConversationStoreInstance = &conversationStore{}
	}
	if SummaryStoreInstance == nil {
		SummaryStoreInstance = &summaryStore{}
	}
	if ModelProfileStoreInstance == nil {
		ModelProfileStoreInstance = &modelProfileStore{}
	}
	if ResearchTaskStoreInstance == nil {
		ResearchTaskStoreInstance = &researchTaskStore{}
	}
	if UserConfigStoreInstance == nil {
		UserConfigStoreInstance = &userConfigStore{}
	}

	return nil
}
