// Package config provides configuration handling
package config

import (
	"os"
	"strings"
	"sync"
	"time"

	"maistro/util"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"github.com/spf13/viper"
)

// Config represents the application configuration
type Config struct {
	Server   ServerConfig   `json:"server" mapstructure:"server"`
	Database DatabaseConfig `json:"database" mapstructure:"database"`
	// Ollama            InferenceService       `json:"ollama" mapstructure:"ollama"`
	Redis             RedisConfig            `json:"redis" mapstructure:"redis"`
	Auth              AuthConfig             `json:"auth" mapstructure:"auth"`
	PreferenceMap     map[string]UserConfig  `json:"-" mapstructure:"preferences_map"`
	InferenceServices InferenceServiceConfig `json:"inferenceServices" mapstructure:"inference_services"`

	// Default user preferences, will be copied to new users
	Summarization SummarizationConfig `json:"summarization" mapstructure:"summarization"`
	Memory        MemoryConfig        `json:"retrieval" mapstructure:"retrieval"`
	WebSearch     WebSearchConfig     `json:"webSearch" mapstructure:"web_search"`
	Preferences   PreferencesConfig   `json:"preferences" mapstructure:"preferences"`
	ModelProfiles ModelProfileConfig  `json:"modelProfiles,omitempty"`
	Refinement    RefinementConfig    `json:"refinement,omitempty" mapstructure:"refinement"`
	LogLevel      string              `json:"logLevel" mapstructure:"log_level"`
}

// InferenceServiceConfig contains configuration for inference services
type InferenceServiceConfig struct {
	Ollama          InferenceService `json:"ollama" mapstructure:"ollama"`
	StableDiffusion InferenceService `json:"stableDiffusion" mapstructure:"stable_diffusion"`
}

// UserConfig represents user-specific configuration
type UserConfig struct {
	UserID        string               `json:"userId,omitempty"`
	Summarization *SummarizationConfig `json:"summarization,omitempty"`
	Memory        *MemoryConfig        `json:"memory,omitempty"`
	WebSearch     *WebSearchConfig     `json:"webSearch,omitempty"`
	Preferences   *PreferencesConfig   `json:"preferences,omitempty"`
	ModelProfiles *ModelProfileConfig  `json:"modelProfiles,omitempty"`
	Refinement    *RefinementConfig    `json:"refinement,omitempty"`
}

// ServerConfig contains server configuration
type ServerConfig struct {
	Host    string `json:"host" mapstructure:"host"`
	Port    int    `json:"port" mapstructure:"port"`
	BaseURL string `json:"baseUrl" mapstructure:"base_url"`
}

// DatabaseConfig contains database configuration
type DatabaseConfig struct {
	Host             string `json:"host" mapstructure:"host"`
	Port             int    `json:"port" mapstructure:"port"`
	User             string `json:"user" mapstructure:"user"`
	Password         string `json:"password" mapstructure:"password"`
	DBName           string `json:"dbname" mapstructure:"dbname"`
	SSLMode          string `json:"sslmode" mapstructure:"sslmode"`
	ConnectionString string `json:"connectionString" mapstructure:"connection_string"`
}

// InferenceService contains ollama service configuration
type InferenceService struct {
	BaseURL string `json:"baseUrl" mapstructure:"base_url"`
}

// RedisConfig contains redis configuration
type RedisConfig struct {
	Host            string `json:"host" mapstructure:"host"`
	Port            int    `json:"port" mapstructure:"port"`
	Password        string `json:"password" mapstructure:"password"`
	DB              int    `json:"db" mapstructure:"db"`
	Enabled         bool   `json:"enabled" mapstructure:"enabled"`
	ConversationTTL string `json:"conversationTtl" mapstructure:"conversation_ttl"`
	MessageTTL      string `json:"messageTtl" mapstructure:"message_ttl"`
	SummaryTTL      string `json:"summaryTtl" mapstructure:"summary_ttl"`
	PoolSize        int    `json:"poolSize" mapstructure:"pool_size"`
	MinIdleConns    int    `json:"minIdleConnections" mapstructure:"min_idle_connections"`
	ConnectTimeout  string `json:"connectTimeout" mapstructure:"connect_timeout"`
}

// AuthConfig contains authentication configuration
type AuthConfig struct {
	JWKSUri      string `json:"jwksUri" mapstructure:"jwks_uri"`
	Audience     string `json:"audience" mapstructure:"audience"`
	ClientID     string `json:"clientId" mapstructure:"client_id"`
	ClientSecret string `json:"clientSecret" mapstructure:"client_secret"`
}

// WebSearchConfig contains web search settings
type WebSearchConfig struct {
	Enabled        bool `json:"enabled,omitempty" mapstructure:"enabled"`
	AutoDetect     bool `json:"autoDetect,omitempty" mapstructure:"auto_detect"`
	MaxResults     int  `json:"maxResults,omitempty" mapstructure:"max_results"`
	IncludeResults bool `json:"includeResults,omitempty" mapstructure:"include_results"`
}

// SummarizationConfig contains conversation summarization settings
type SummarizationConfig struct {
	Enabled                      bool    `json:"enabled,omitempty" mapstructure:"enabled"`
	MessagesBeforeSummary        int     `json:"messagesBeforeSummary,omitempty" mapstructure:"messages_before_summary"`
	SummariesBeforeConsolidation int     `json:"summariesBeforeConsolidation,omitempty" mapstructure:"summaries_before_consolidation"`
	EmbeddingModel               string  `json:"embeddingModel,omitempty" mapstructure:"embedding_model"`
	EmbeddingDimension           int     `json:"embeddingDimension,omitempty" mapstructure:"embedding_dimension"`
	MaxSummaryLevels             int     `json:"maxSummaryLevels,omitempty" mapstructure:"max_summary_levels"`
	SummaryWeightCoefficient     float64 `json:"summaryWeightCoefficient,omitempty" mapstructure:"summary_weight_coefficient"`
}

// RefinementConfig contains response refinement settings
type RefinementConfig struct {
	EnableResponseFiltering bool `json:"enableResponseFiltering,omitempty" mapstructure:"enable_response_filtering"`
	EnableResponseCritique  bool `json:"enableResponseCritique,omitempty" mapstructure:"enable_response_critique"`
}

// MemoryConfig contains memory retrieval settings
type MemoryConfig struct {
	Enabled                 bool    `json:"enabled,omitempty" mapstructure:"enabled"`
	Limit                   int     `json:"limit,omitempty" mapstructure:"limit"`
	ProfileID               string  `json:"profileId,omitempty" mapstructure:"profile_id"`
	EnableCrossUser         bool    `json:"enableCrossUser,omitempty" mapstructure:"enable_cross_user"`
	EnableCrossConversation bool    `json:"enableCrossConversation,omitempty" mapstructure:"enable_cross_conversation"`
	SimilarityThreshold     float64 `json:"similarityThreshold,omitempty" mapstructure:"similarity_threshold"`
	AlwaysRetrieve          bool    `json:"alwaysRetrieve,omitempty" mapstructure:"always_retrieve"`
}

// PreferencesConfig contains user preferences
type PreferencesConfig struct {
	DefaultProfileID string `json:"defaultProfileId,omitempty" mapstructure:"default_profile_id"`
	Theme            string `json:"theme,omitempty" mapstructure:"theme"`
	FontSize         int    `json:"fontSize,omitempty" mapstructure:"font_size"`
	NotificationsOn  bool   `json:"notificationsOn,omitempty" mapstructure:"notifications_on"`
	Language         string `json:"language,omitempty" mapstructure:"language"`
}

type ModelProfileConfig struct {
	PrimaryProfileID               uuid.UUID `json:"primaryProfileId,omitempty" mapstructure:"primary_profile_id"`
	SummarizationProfileID         uuid.UUID `json:"summarizationProfileId,omitempty" mapstructure:"summarization_profile_id"`
	MasterSummaryProfileID         uuid.UUID `json:"masterSummaryProfileId,omitempty" mapstructure:"master_summary_profile_id"`
	BriefSummaryProfileID          uuid.UUID `json:"briefSummaryProfileId,omitempty" mapstructure:"brief_summary_profile_id"`
	KeyPointsProfileID             uuid.UUID `json:"keyPointsProfileId,omitempty" mapstructure:"key_points_profile_id"`
	ImprovementProfileID           uuid.UUID `json:"improvementProfileId,omitempty" mapstructure:"improvement_profile_id"`
	AnalysisProfileID              uuid.UUID `json:"analysisProfileId,omitempty" mapstructure:"analysis_profile_id"`
	MemoryRetrievalProfileID       uuid.UUID `json:"memoryRetrievalProfileId,omitempty" mapstructure:"memory_retrieval_profile_id"`
	SelfCritiqueProfileID          uuid.UUID `json:"selfCritiqueProfileId,omitempty" mapstructure:"self_critique_profile_id"`
	ResearchTaskProfileID          uuid.UUID `json:"researchTaskProfileId,omitempty" mapstructure:"research_task_profile_id"`
	ResearchPlanProfileID          uuid.UUID `json:"researchPlanProfileId,omitempty" mapstructure:"research_plan_profile_id"`
	ResearchConsolidationProfileID uuid.UUID `json:"researchConsolidationProfileId,omitempty" mapstructure:"research_consolidation_profile_id"`
	ResearchAnalysisProfileID      uuid.UUID `json:"researchAnalysisProfileId,omitempty" mapstructure:"research_analysis_profile_id"`
	EmbeddingProfileID             uuid.UUID `json:"embeddingProfileId,omitempty" mapstructure:"embedding_profile_id"`
	FormattingProfileID            uuid.UUID `json:"formattingProfileId,omitempty" mapstructure:"formatting_profile_id"`
	ImageGenerationProfileID       uuid.UUID `json:"imageGenerationProfileId,omitempty" mapstructure:"image_generation_profile_id"`
}

var (
	config                    Config
	configOnce                sync.Once
	defaultModelProfileConfig = ModelProfileConfig{
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
func GetConfig(configFile *string) Config {
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
		v := viper.New()
		v.SetConfigFile(filePath)

		// Enable env var overrides (e.g. SERVER_HOST, DATABASE_PORT, etc)
		v.AutomaticEnv()
		v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

		// Try to read config file (optional)
		_ = v.ReadInConfig() // ignore error, use defaults if missing

		// Unmarshal into config struct
		if err := v.Unmarshal(&config); err != nil {
			util.LogWarning("Warning: could not unmarshal config", logrus.Fields{"error": err})
		}

		config.ModelProfiles = defaultModelProfileConfig
	})
	return config
}

// LoadConfig loads the configuration from a file
func LoadConfig(path string) (*Config, error) {
	viper.SetConfigFile(path)

	// Set default values
	viper.SetDefault("server.host", "0.0.0.0")
	viper.SetDefault("server.port", 8000)
	viper.SetDefault("server.base_url", "http://localhost:8000")

	// Set default database config
	viper.SetDefault("database.host", "localhost")
	viper.SetDefault("database.port", 5432)
	viper.SetDefault("database.user", "postgres")
	viper.SetDefault("database.password", "postgres")
	viper.SetDefault("database.dbname", "proxyllama")
	viper.SetDefault("database.sslmode", "disable")

	// Set default Ollama config
	viper.SetDefault("ollama.base_url", "http://localhost:11434")

	// Set default Redis config
	viper.SetDefault("redis.host", "localhost")
	viper.SetDefault("redis.port", 6379)
	viper.SetDefault("redis.enabled", false)
	viper.SetDefault("redis.conversation_ttl", "168h") // 7 days
	viper.SetDefault("redis.message_ttl", "168h")      // 7 days
	viper.SetDefault("redis.summary_ttl", "720h")      // 30 days
	viper.SetDefault("redis.pool_size", 10)
	viper.SetDefault("redis.min_idle_connections", 3)
	viper.SetDefault("redis.connect_timeout", "5s")

	// Set default summarization config
	viper.SetDefault("summarization.enabled", true)
	viper.SetDefault("summarization.messages_before_summary", 10)
	viper.SetDefault("summarization.summaries_before_consolidation", 5)
	viper.SetDefault("summarization.summary_model", "qwen3:0.6b")
	viper.SetDefault("summarization.critique_model", "qwen3:0.6b")
	viper.SetDefault("summarization.embedding_model", "qwen3:0.6b")
	viper.SetDefault("summarization.embedding_dimension", 768)
	viper.SetDefault("summarization.enable_rag", false)
	viper.SetDefault("summarization.enable_response_filtering", true)
	viper.SetDefault("summarization.enable_response_critique", false)
	viper.SetDefault("summarization.max_summary_levels", 3)
	viper.SetDefault("summarization.summary_weight_coefficient", 0.7)

	// Set default retrieval config
	viper.SetDefault("retrieval.enabled", true)
	viper.SetDefault("retrieval.limit", 5)
	viper.SetDefault("retrieval.enable_cross_conversation", false)
	viper.SetDefault("retrieval.similarity_threshold", 0.7)
	viper.SetDefault("retrieval.always_retrieve", false)

	// Set default web search config
	viper.SetDefault("web_search.enabled", false)
	viper.SetDefault("web_search.auto_detect", true)
	viper.SetDefault("web_search.max_results", 3)
	viper.SetDefault("web_search.include_results", true)

	// Set default user preferences
	viper.SetDefault("preferences.default_model", "qwen3:0.6b")
	viper.SetDefault("preferences.theme", "light")
	viper.SetDefault("preferences.font_size", 14)
	viper.SetDefault("preferences.notifications_on", true)
	viper.SetDefault("preferences.language", "en")

	// Load environment variables
	viper.AutomaticEnv()

	// Replace config values with environment variables if they exist
	if host := os.Getenv("SERVER_HOST"); host != "" {
		viper.Set("server.host", host)
	}
	if port := os.Getenv("SERVER_PORT"); port != "" {
		viper.Set("server.port", port)
	}
	if baseURL := os.Getenv("SERVER_BASE_URL"); baseURL != "" {
		viper.Set("server.base_url", baseURL)
	}

	// Read the config file
	err := viper.ReadInConfig()
	if err != nil {
		return nil, err
	}

	var cfg Config
	err = viper.Unmarshal(&cfg)
	if err != nil {
		return nil, err
	}

	return &cfg, nil
}

// GetUserConfig returns the configuration for a specific user
func (c *Config) GetUserConfig(userID string) UserConfig {
	if userConfig, ok := c.PreferenceMap[userID]; ok {
		return userConfig
	}

	// Return a copy of the default preferences for new users
	return UserConfig{
		UserID:        userID,
		Summarization: &c.Summarization,
		Memory:        &c.Memory,
		WebSearch:     &c.WebSearch,
		Preferences:   &c.Preferences,
	}
}

// UpdateUserConfig updates the configuration for a specific user
func (c *Config) UpdateUserConfig(userConfig UserConfig) {
	if c.PreferenceMap == nil {
		c.PreferenceMap = make(map[string]UserConfig)
	}
	c.PreferenceMap[userConfig.UserID] = userConfig
}

// SaveConfig saves the current configuration back to file
func SaveConfig(cfg *Config, path string) error {
	viper.SetConfigFile(path)

	// Transfer the config struct back to viper
	err := viper.ReadInConfig()
	if err != nil {
		return err
	}

	// Write the configuration to file
	return viper.WriteConfigAs(path)
}

// GetRedisConfigDurations returns the Redis TTL durations converted from strings
func (c *Config) GetRedisConfigDurations() (conversationTTL, messageTTL, summaryTTL, connectTimeout time.Duration, err error) {
	conversationTTL, err = time.ParseDuration(c.Redis.ConversationTTL)
	if err != nil {
		return
	}

	messageTTL, err = time.ParseDuration(c.Redis.MessageTTL)
	if err != nil {
		return
	}

	summaryTTL, err = time.ParseDuration(c.Redis.SummaryTTL)
	if err != nil {
		return
	}

	connectTimeout, err = time.ParseDuration(c.Redis.ConnectTimeout)
	return
}
