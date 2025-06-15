// Package config provides configuration handling
package config

import (
	"os"
	"sync"

	"maistro/models"
	"maistro/util"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

var (
	config                    *models.Config
	configOnce                sync.Once
	defaultModelProfileConfig = models.ModelProfileConfig{
		PrimaryProfileID:               DefaultPrimaryProfile.ID,
		SummarizationProfileID:         DefaultSummarizationProfile.ID,
		MasterSummaryProfileID:         DefaultMasterSummaryProfile.ID,
		BriefSummaryProfileID:          DefaultBriefSummaryProfile.ID,
		KeyPointsProfileID:             DefaultKeyPointsProfile.ID,
		ImprovementProfileID:           DefaultImprovementProfile.ID,
		MemoryRetrievalProfileID:       DefaultMemoryRetrievalProfile.ID,
		SelfCritiqueProfileID:          DefaultSelfCritiqueProfile.ID,
		AnalysisProfileID:              DefaultAnalysisProfile.ID,
		ResearchTaskProfileID:          DefaultResearchTaskProfile.ID,
		ResearchPlanProfileID:          DefaultResearchPlanProfile.ID,
		ResearchConsolidationProfileID: DefaultResearchConsolidationProfile.ID,
		ResearchAnalysisProfileID:      DefaultResearchAnalysisProfile.ID,
		EmbeddingProfileID:             DefaultEmbeddingProfile.ID,
		FormattingProfileID:            DefaultFormattingProfile.ID,
		ImageGenerationProfileID:       DefaultImageGenerationProfile.ID,
	}
)

// GetConfig loads configuration from config.yaml with environment variable overrides
func GetConfig(configFile *string) *models.Config {
	configOnce.Do(func() {
		// Use viper for all config loading and merging
		var filePath string
		if configFile != nil {
			filePath = *configFile
		} else if os.Getenv("LOCAL") == "true" {
			filePath = ".config.local.yaml"
		} else {
			filePath = ".config.yaml"
		}

		// Read the config file into a byte slice
		data, err := os.ReadFile(filePath)
		if err != nil {
			util.HandleFatalError(err, logrus.Fields{"error": "Failed to read configuration file"})
		}

		// Initialize config as a new struct before unmarshalling
		cfg := &models.Config{}
		if err := yaml.Unmarshal(data, cfg); err != nil {
			util.HandleFatalError(err, logrus.Fields{"error": "Failed to unmarshal configuration"})
		}
		cfg.ModelProfiles = defaultModelProfileConfig
		config = cfg
	})
	return config
}
