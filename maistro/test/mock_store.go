package test

import (
	"context"
	"maistro/models"
	"maistro/storage"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// mockMessageStore delegates read-only methods to the real MessageStore
// and stubs write methods for testing.
type mockMessageStore struct {
	real storage.MessageStore
}

func (ms *mockMessageStore) AddMessage(ctx context.Context, conversationID int, role models.MessageRole, content []models.MessageContent, usrCfg *models.UserConfig) (int, error) {
	return ms.real.AddMessage(ctx, conversationID, role, content, usrCfg)
}
func (ms *mockMessageStore) GetMessage(ctx context.Context, messageID int) (*models.Message, error) {
	return ms.real.GetMessage(ctx, messageID)
}
func (ms *mockMessageStore) GetConversationHistory(ctx context.Context, conversationID int) ([]models.Message, error) {
	return ms.real.GetConversationHistory(ctx, conversationID)
}
func (ms *mockMessageStore) DeleteMessage(ctx context.Context, messageID int) error {
	return ms.real.DeleteMessage(ctx, messageID)
}

// mockConversationStore delegates read-only methods to the real ConversationStore
// and stubs write methods for testing.
type mockConversationStore struct {
	real storage.ConversationStore
}

func (m *mockConversationStore) CreateConversation(ctx context.Context, userID string, title string) (int, error) {
	return m.real.CreateConversation(ctx, userID, title)
}
func (m *mockConversationStore) GetUserConversations(ctx context.Context, userID string) ([]models.Conversation, error) {
	return m.real.GetUserConversations(ctx, userID)
}
func (m *mockConversationStore) GetConversation(ctx context.Context, conversationID int) (*models.Conversation, error) {
	return m.real.GetConversation(ctx, conversationID)
}
func (m *mockConversationStore) UpdateConversationTitle(ctx context.Context, conversationID int, title string) error {
	return m.real.UpdateConversationTitle(ctx, conversationID, title)
}
func (m *mockConversationStore) DeleteConversation(ctx context.Context, conversationID int) error {
	return m.real.DeleteConversation(ctx, conversationID)
}

// mockSummaryStore delegates read-only methods to the real SummaryStore
// and stubs write methods for testing.
type mockSummaryStore struct {
	real storage.SummaryStore
}

func (m *mockSummaryStore) CreateSummary(ctx context.Context, conversationID int, content string, level int, sourceIDs []int) (int, error) {
	return m.real.CreateSummary(ctx, conversationID, content, level, sourceIDs)
}
func (m *mockSummaryStore) GetSummariesForConversation(ctx context.Context, conversationID int) ([]models.Summary, error) {
	return m.real.GetSummariesForConversation(ctx, conversationID)
}
func (m *mockSummaryStore) GetRecentSummaries(ctx context.Context, conversationID int, level int, limit int) ([]models.Summary, error) {
	return m.real.GetRecentSummaries(ctx, conversationID, level, limit)
}
func (m *mockSummaryStore) DeleteSummariesForConversation(ctx context.Context, conversationID int) error {
	return m.real.DeleteSummariesForConversation(ctx, conversationID)
}
func (m *mockSummaryStore) GetSummary(ctx context.Context, summaryID int) (*models.Summary, error) {
	return m.real.GetSummary(ctx, summaryID)
}

// mockModelProfileStore delegates read-only methods to the real ModelProfileStore
// and stubs write methods for testing.
type mockModelProfileStore struct {
	real storage.ModelProfileStore
}

func (m *mockModelProfileStore) CreateModelProfile(ctx context.Context, profile *models.ModelProfile) (uuid.UUID, error) {
	return m.real.CreateModelProfile(ctx, profile)
}
func (m *mockModelProfileStore) GetModelProfile(ctx context.Context, profileID uuid.UUID) (*models.ModelProfile, error) {
	switch profileID.String() {
	case "00000000-0000-0000-0000-000000000001":
		return &testDefaultPrimaryProfile, nil
	case "00000000-0000-0000-0000-000000000002":
		return &testDefaultSummarizationProfile, nil
	case "00000000-0000-0000-0000-000000000003":
		return &testDefaultMasterSummaryProfile, nil
	case "00000000-0000-0000-0000-000000000004":
		return &testDefaultBriefSummaryProfile, nil
	case "00000000-0000-0000-0000-000000000005":
		return &testDefaultKeyPointsProfile, nil
	case "00000000-0000-0000-0000-000000000006":
		return &testDefaultSelfCritiqueProfile, nil
	case "00000000-0000-0000-0000-000000000007":
		return &testDefaultImprovementProfile, nil
	case "00000000-0000-0000-0000-000000000008":
		return &testDefaultMemoryRetrievalProfile, nil
	case "00000000-0000-0000-0000-000000000009":
		return &testDefaultAnalysisProfile, nil
	case "00000000-0000-0000-0000-000000000010":
		return &testDefaultResearchTaskProfile, nil
	case "00000000-0000-0000-0000-000000000011":
		return &testDefaultResearchPlanProfile, nil
	case "00000000-0000-0000-0000-000000000012":
		return &testDefaultResearchConsolidationProfile, nil
	case "00000000-0000-0000-0000-000000000013":
		return &testDefaultResearchAnalysisProfile, nil
	case "00000000-0000-0000-0000-000000000014":
		return &testDefaultEmbeddingProfile, nil
	case "00000000-0000-0000-0000-000000000015":
		return &testDefaultFormattingProfile, nil
	default:
		return m.real.GetModelProfile(ctx, profileID)
	}
}
func (m *mockModelProfileStore) UpdateModelProfile(ctx context.Context, profile *models.ModelProfile) error {
	return m.real.UpdateModelProfile(ctx, profile)
}
func (m *mockModelProfileStore) DeleteModelProfile(ctx context.Context, profileID uuid.UUID) error {
	return m.real.DeleteModelProfile(ctx, profileID)
}
func (m *mockModelProfileStore) ListModelProfilesByUser(ctx context.Context, userID string) ([]*models.ModelProfile, error) {
	return m.real.ListModelProfilesByUser(ctx, userID)
}

// mockResearchTaskStore delegates read-only methods to the real ResearchTaskStore
// and stubs write methods for testing.
type mockResearchTaskStore struct {
	real storage.ResearchTaskStore
}

func (m *mockResearchTaskStore) SaveResearchTask(ctx context.Context, userID, query string, conversationID *int) (int, error) {
	return m.real.SaveResearchTask(ctx, userID, query, conversationID)
}
func (m *mockResearchTaskStore) UpdateTaskStatus(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error) {
	return m.real.UpdateTaskStatus(ctx, taskID, status, errorMsg)
}
func (m *mockResearchTaskStore) UpdateTask(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error) {
	return m.real.UpdateTask(ctx, taskID, status, errorMsg)
}
func (m *mockResearchTaskStore) StoreResearchPlan(ctx context.Context, taskID int, plan *models.ResearchPlan) (time.Time, error) {
	return m.real.StoreResearchPlan(ctx, taskID, plan)
}
func (m *mockResearchTaskStore) StoreFinalResult(ctx context.Context, taskID int, result *models.ResearchQuestionResult) (time.Time, error) {
	return m.real.StoreFinalResult(ctx, taskID, result)
}
func (m *mockResearchTaskStore) SaveSubtask(ctx context.Context, subtask *models.ResearchSubtask) (int, error) {
	return m.real.SaveSubtask(ctx, subtask)
}
func (m *mockResearchTaskStore) UpdateSubtaskStatus(ctx context.Context, taskID, questionID int, status string, errorMsg *string) (int, time.Time, error) {
	return m.real.UpdateSubtaskStatus(ctx, taskID, questionID, status, errorMsg)
}
func (m *mockResearchTaskStore) StoreGatheredInfo(ctx context.Context, taskID, questionID int, gatheredInfo []string, sources []string) (time.Time, error) {
	return m.real.StoreGatheredInfo(ctx, taskID, questionID, gatheredInfo, sources)
}
func (m *mockResearchTaskStore) StoreSynthesizedAnswer(ctx context.Context, taskID, questionID int, answer string) (time.Time, error) {
	return m.real.StoreSynthesizedAnswer(ctx, taskID, questionID, answer)
}
func (m *mockResearchTaskStore) GetTaskByID(ctx context.Context, taskID int) (*models.ResearchTask, error) {
	return m.real.GetTaskByID(ctx, taskID)
}
func (m *mockResearchTaskStore) ListTasksByUserID(ctx context.Context, userID string, limit, offset int) ([]models.ResearchTask, error) {
	return m.real.ListTasksByUserID(ctx, userID, limit, offset)
}
func (m *mockResearchTaskStore) GetSubtasksForTask(ctx context.Context, taskID string) ([]models.ResearchSubtask, error) {
	return m.real.GetSubtasksForTask(ctx, taskID)
}

// mockMemoryStore delegates read-only methods to the real MemoryStore
// and stubs write methods for testing.
type mockMemoryStore struct {
	real storage.MemoryStore
}

func (m *mockMemoryStore) InitMemorySchema(ctx context.Context) error {
	return m.real.InitMemorySchema(ctx)
}
func (m *mockMemoryStore) StoreMemory(ctx context.Context, userID, source string, role models.MessageRole, sourceID int, embeddings [][]float32) error {
	return m.real.StoreMemory(ctx, userID, source, role, sourceID, embeddings)
}
func (m *mockMemoryStore) StoreMemoryWithTx(ctx context.Context, userID, source string, role models.MessageRole, sourceID int, embeddings [][]float32, tx pgx.Tx) error {
	return m.real.StoreMemoryWithTx(ctx, userID, source, role, sourceID, embeddings, tx)
}
func (m *mockMemoryStore) DeleteMemory(ctx context.Context, id, userID string) error {
	return m.real.DeleteMemory(ctx, id, userID)
}
func (m *mockMemoryStore) DeleteAllUserMemories(ctx context.Context, userID string) error {
	return m.real.DeleteAllUserMemories(ctx, userID)
}
func (m *mockMemoryStore) SearchSimilarity(ctx context.Context, embeddings [][]float32, minSimilarity float32, limit int, userID *string, conversationID *int, startDate, endDate *time.Time) ([]models.Memory, error) {
	return m.real.SearchSimilarity(ctx, embeddings, minSimilarity, limit, userID, conversationID, startDate, endDate)
}

// mockUserConfigStore delegates read-only methods to the real UserConfigStore
// and stubs write methods for testing.
type mockUserConfigStore struct {
	real storage.UserConfigStore
}

func (m *mockUserConfigStore) GetUserConfig(ctx context.Context, userID string) (*models.UserConfig, error) {
	return m.real.GetUserConfig(ctx, userID)
}
func (m *mockUserConfigStore) UpdateUserConfig(ctx context.Context, userID string, cfg *models.UserConfig) error {
	return m.real.UpdateUserConfig(ctx, userID, cfg)
}
func (m *mockUserConfigStore) GetAllUsers(ctx context.Context) ([]models.User, error) {
	return m.real.GetAllUsers(ctx)
}

func InitMockStore() error {
	storage.InitializeStorage()

	storage.MessageStoreInstance = &mockMessageStore{real: storage.MessageStoreInstance}
	storage.ConversationStoreInstance = &mockConversationStore{real: storage.ConversationStoreInstance}
	storage.SummaryStoreInstance = &mockSummaryStore{real: storage.SummaryStoreInstance}
	storage.ModelProfileStoreInstance = &mockModelProfileStore{real: storage.ModelProfileStoreInstance}
	storage.ResearchTaskStoreInstance = &mockResearchTaskStore{real: storage.ResearchTaskStoreInstance}
	storage.MemoryStoreInstance = &mockMemoryStore{real: storage.MemoryStoreInstance}
	storage.UserConfigStoreInstance = &mockUserConfigStore{real: storage.UserConfigStoreInstance}
	return nil
}
