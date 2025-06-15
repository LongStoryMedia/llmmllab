package config

import (
	"maistro/models"
	"maistro/util"
	"reflect"
	"testing"
)

func TestMergeWithDefaultConfig_UserConfigPreferred(t *testing.T) {
	base := GetConfig(nil)
	user := models.UserConfig{
		Summarization: &models.SummarizationConfig{Enabled: true},
		Memory:        &models.MemoryConfig{Enabled: true},
		WebSearch:     &models.WebSearchConfig{Enabled: false},
		Preferences:   &models.PreferencesConfig{FontSize: util.IntPtr(43)},
		ModelProfiles: &models.ModelProfileConfig{PrimaryProfileID: base.ModelProfiles.PrimaryProfileID},
	}

	MergeWithDefaultConfig(&user)

	if user.Summarization.Enabled != true {
		t.Errorf("Expected user Summarization.Enabled to be preferred")
	}
	if user.Memory.Enabled != true {
		t.Errorf("Expected user Retrieval.Enabled to be preferred")
	}
	if user.WebSearch.Enabled != false {
		t.Errorf("Expected user WebSearch.Enabled to be preferred")
	}
}

func TestMergeWithDefaultConfig_DefaultsUsedIfNil(t *testing.T) {
	base := GetConfig(nil)
	user := models.UserConfig{
		UserID:        "test_user",
		Summarization: nil,
		Memory:        nil,
		WebSearch:     nil,
		Preferences:   nil,
		ModelProfiles: nil,
	}

	MergeWithDefaultConfig(&user)

	if !reflect.DeepEqual(base.Summarization, *user.Summarization) || user.Summarization == nil {
		t.Errorf("Expected default Summarization to be used")
	}
	if !reflect.DeepEqual(base.Memory, *user.Memory) || user.Memory == nil {
		t.Errorf("Expected default Retrieval to be used")
	}
	if !reflect.DeepEqual(base.WebSearch, *user.WebSearch) || user.WebSearch == nil {
		t.Errorf("Expected default WebSearch to be used")
	}
	if !reflect.DeepEqual(base.Preferences, *user.Preferences) || user.Preferences == nil {
		t.Errorf("Expected default Preferences to be used")
	}
	if !reflect.DeepEqual(base.ModelProfiles, *user.ModelProfiles) || user.ModelProfiles == nil {
		t.Errorf("Expected default ModelProfiles to be used")
	}
}
