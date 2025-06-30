package test

import (
	"context"
	"maistro/config"
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

func (ms *mockMessageStore) AddMessage(ctx context.Context, conversationID int, role, content string, usrCfg *models.UserConfig) (int, error) {
	// Stubbed for testing
	return 1, nil
}
func (ms *mockMessageStore) GetMessage(ctx context.Context, messageID int) (*models.Message, error) {
	return ms.real.GetMessage(ctx, messageID)
}
func (ms *mockMessageStore) GetConversationHistory(ctx context.Context, conversationID int) ([]models.Message, error) {
	return ms.real.GetConversationHistory(ctx, conversationID)
}
func (ms *mockMessageStore) DeleteMessage(ctx context.Context, messageID int) error {
	// Stubbed for testing
	return nil
}

// mockConversationStore delegates read-only methods to the real ConversationStore
// and stubs write methods for testing.
type mockConversationStore struct {
	real storage.ConversationStore
}

func (m *mockConversationStore) CreateConversation(ctx context.Context, userID string, title string) (int, error) {
	return 1, nil
}
func (m *mockConversationStore) GetUserConversations(ctx context.Context, userID string) ([]models.Conversation, error) {
	return m.real.GetUserConversations(ctx, userID)
}
func (m *mockConversationStore) GetConversation(ctx context.Context, conversationID int) (*models.Conversation, error) {
	return &models.Conversation{
		ID:        conversationID,
		UserID:    "CgNsc20SBGxkYXA", // lsm
		Title:     "Test Conversation",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}, nil
}
func (m *mockConversationStore) UpdateConversationTitle(ctx context.Context, conversationID int, title string) error {
	return nil
}
func (m *mockConversationStore) DeleteConversation(ctx context.Context, conversationID int) error {
	return nil
}

// mockSummaryStore delegates read-only methods to the real SummaryStore
// and stubs write methods for testing.
type mockSummaryStore struct {
	real storage.SummaryStore
}

func (m *mockSummaryStore) CreateSummary(ctx context.Context, conversationID int, content string, level int, sourceIDs []int) (int, error) {
	return 1, nil
}
func (m *mockSummaryStore) GetSummariesForConversation(ctx context.Context, conversationID int) ([]models.Summary, error) {
	return m.real.GetSummariesForConversation(ctx, conversationID)
}
func (m *mockSummaryStore) GetRecentSummaries(ctx context.Context, conversationID int, level int, limit int) ([]models.Summary, error) {
	return m.real.GetRecentSummaries(ctx, conversationID, level, limit)
}
func (m *mockSummaryStore) DeleteSummariesForConversation(ctx context.Context, conversationID int) error {
	return nil
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
	return uuid.Nil, nil
}
func (m *mockModelProfileStore) GetModelProfile(ctx context.Context, profileID uuid.UUID) (*models.ModelProfile, error) {
	switch profileID.String() {
	case "00000000-0000-0000-0000-000000000001":
		return &config.DefaultPrimaryProfile, nil
	case "00000000-0000-0000-0000-000000000002":
		return &config.DefaultSummarizationProfile, nil
	case "00000000-0000-0000-0000-000000000003":
		return &config.DefaultMasterSummaryProfile, nil
	case "00000000-0000-0000-0000-000000000004":
		return &config.DefaultBriefSummaryProfile, nil
	case "00000000-0000-0000-0000-000000000005":
		return &config.DefaultKeyPointsProfile, nil
	case "00000000-0000-0000-0000-000000000006":
		return &config.DefaultSelfCritiqueProfile, nil
	case "00000000-0000-0000-0000-000000000007":
		return &config.DefaultImprovementProfile, nil
	case "00000000-0000-0000-0000-000000000008":
		return &config.DefaultMemoryRetrievalProfile, nil
	case "00000000-0000-0000-0000-000000000009":
		return &config.DefaultAnalysisProfile, nil
	case "00000000-0000-0000-0000-000000000010":
		return &config.DefaultResearchTaskProfile, nil
	case "00000000-0000-0000-0000-000000000011":
		return &config.DefaultResearchPlanProfile, nil
	case "00000000-0000-0000-0000-000000000012":
		return &config.DefaultResearchConsolidationProfile, nil
	case "00000000-0000-0000-0000-000000000013":
		return &config.DefaultResearchAnalysisProfile, nil
	case "00000000-0000-0000-0000-000000000014":
		return &config.DefaultEmbeddingProfile, nil
	case "00000000-0000-0000-0000-000000000015":
		return &config.DefaultFormattingProfile, nil
	default:
		return m.real.GetModelProfile(ctx, profileID)
	}
}
func (m *mockModelProfileStore) UpdateModelProfile(ctx context.Context, profile *models.ModelProfile) error {
	return nil
}
func (m *mockModelProfileStore) DeleteModelProfile(ctx context.Context, profileID uuid.UUID) error {
	return nil
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
	return 1, nil
}
func (m *mockResearchTaskStore) UpdateTaskStatus(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error) {
	return time.Now(), nil
}
func (m *mockResearchTaskStore) UpdateTask(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error) {
	return time.Now(), nil
}
func (m *mockResearchTaskStore) StoreResearchPlan(ctx context.Context, taskID int, plan *models.ResearchPlan) (time.Time, error) {
	return time.Now(), nil
}
func (m *mockResearchTaskStore) StoreFinalResult(ctx context.Context, taskID int, result *models.ResearchQuestionResult) (time.Time, error) {
	return time.Now(), nil
}
func (m *mockResearchTaskStore) SaveSubtask(ctx context.Context, subtask *models.ResearchSubtask) (int, error) {
	return 1, nil
}
func (m *mockResearchTaskStore) UpdateSubtaskStatus(ctx context.Context, taskID, questionID int, status string, errorMsg *string) (int, time.Time, error) {
	return 1, time.Now(), nil
}
func (m *mockResearchTaskStore) StoreGatheredInfo(ctx context.Context, taskID, questionID int, gatheredInfo []string, sources []string) (time.Time, error) {
	return time.Now(), nil
}
func (m *mockResearchTaskStore) StoreSynthesizedAnswer(ctx context.Context, taskID, questionID int, answer string) (time.Time, error) {
	return time.Now(), nil
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
	return nil
}
func (m *mockMemoryStore) StoreMemory(ctx context.Context, userID, source, role string, sourceID int, embeddings [][]float32) error {
	return nil
}
func (m *mockMemoryStore) StoreMemoryWithTx(ctx context.Context, userID, source, role string, sourceID int, embeddings [][]float32, tx pgx.Tx) error {
	return nil
}
func (m *mockMemoryStore) DeleteMemory(ctx context.Context, id, userID string) error {
	return nil
}
func (m *mockMemoryStore) DeleteAllUserMemories(ctx context.Context, userID string) error {
	return nil
}
func (m *mockMemoryStore) SearchSimilarity(ctx context.Context, embeddings [][]float32, minSimilarity float32, limit int, userID *string, conversationID *int, startDate, endDate *time.Time) ([]models.Memory, error) {
	// Stubbed for testing
	return []models.Memory{
		{
			SourceID:       1,
			Source:         "test_source",
			Similarity:     0.8,
			ConversationID: conversationID,
			CreatedAt:      time.Now(),
		},
	}, nil
}

// mockUserConfigStore delegates read-only methods to the real UserConfigStore
// and stubs write methods for testing.
type mockUserConfigStore struct {
	real storage.UserConfigStore
}

func (m *mockUserConfigStore) GetUserConfig(ctx context.Context, userID string) (*models.UserConfig, error) {
	return &models.UserConfig{
		Memory: &models.MemoryConfig{
			Limit:                   3,
			Enabled:                 true,
			AlwaysRetrieve:          true,
			EnableCrossUser:         false,
			SimilarityThreshold:     0.6,
			EnableCrossConversation: true,
		},
		UserID: "CgNsc20SBGxkYXA",
		Refinement: &models.RefinementConfig{
			EnableResponseCritique:  false,
			EnableResponseFiltering: false,
		},
		WebSearch: &models.WebSearchConfig{
			Enabled:        true,
			AutoDetect:     true,
			MaxResults:     3,
			IncludeResults: true,
		},
		Summarization: &models.SummarizationConfig{
			Enabled:                      true,
			MaxSummaryLevels:             3,
			EmbeddingDimension:           786,
			MessagesBeforeSummary:        4,
			SummaryWeightCoefficient:     0.5,
			SummariesBeforeConsolidation: 3,
		},
		ModelProfiles: &models.ModelProfileConfig{
			PrimaryProfileID:               uuid.MustParse("00000000-0000-0000-0000-000000000001"),
			AnalysisProfileID:              uuid.MustParse("00000000-0000-0000-0000-000000000009"),
			EmbeddingProfileID:             uuid.MustParse("00000000-0000-0000-0000-000000000014"),
			FormattingProfileID:            uuid.MustParse("00000000-0000-0000-0000-000000000015"),
			KeyPointsProfileID:             uuid.MustParse("00000000-0000-0000-0000-000000000005"),
			ImprovementProfileID:           uuid.MustParse("00000000-0000-0000-0000-000000000007"),
			BriefSummaryProfileID:          uuid.MustParse("00000000-0000-0000-0000-000000000004"),
			ResearchPlanProfileID:          uuid.MustParse("00000000-0000-0000-0000-000000000011"),
			ResearchTaskProfileID:          uuid.MustParse("00000000-0000-0000-0000-000000000010"),
			SelfCritiqueProfileID:          uuid.MustParse("00000000-0000-0000-0000-000000000006"),
			SummarizationProfileID:         uuid.MustParse("00000000-0000-0000-0000-000000000002"),
			MasterSummaryProfileID:         uuid.MustParse("00000000-0000-0000-0000-000000000003"),
			ImageGenerationProfileID:       uuid.MustParse("00000000-0000-0000-0000-000000000016"),
			MemoryRetrievalProfileID:       uuid.MustParse("00000000-0000-0000-0000-000000000008"),
			ResearchAnalysisProfileID:      uuid.MustParse("00000000-0000-0000-0000-000000000013"),
			ResearchConsolidationProfileID: uuid.MustParse("00000000-0000-0000-0000-000000000012"),
			ImageGenerationPromptProfileID: uuid.MustParse("00000000-0000-0000-0000-000000000017"),
		},
		ImageGeneration: &models.ImageGenerationConfig{
			Enabled:              true,
			MaxImageSize:         1280,
			RetentionHours:       1,
			StorageDirectory:     "/root/images",
			AutoPromptRefinement: false,
		},
	}, nil
}
func (m *mockUserConfigStore) UpdateUserConfig(ctx context.Context, userID string, cfg *models.UserConfig) error {
	return nil
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
