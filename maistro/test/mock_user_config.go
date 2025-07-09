package test

import (
	"maistro/models"

	"github.com/google/uuid"
)

var MockUserConfig = models.UserConfig{
	Memory: &models.MemoryConfig{
		Limit:                   3,
		Enabled:                 true,
		AlwaysRetrieve:          true,
		EnableCrossUser:         false,
		SimilarityThreshold:     0.6,
		EnableCrossConversation: true,
	},
	UserID: "test_user",
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
		StorageDirectory:     "./testdata",
		AutoPromptRefinement: false,
	},
}
