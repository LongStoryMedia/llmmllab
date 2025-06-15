package config

import (
	"maistro/models"
	"maistro/util"
	"reflect"
)

func MergeWithDefaultConfig(userConfig *models.UserConfig) {
	base := GetConfig(nil)

	// Fill in nil fields in userConfig from base config
	if userConfig.ModelProfiles == nil {
		userConfig.ModelProfiles = &base.ModelProfiles
	}

	// Iterate over each field in userConfig.ModelProfiles
	// and set it to the corresponding field in base.ModelProfiles if it's nil
	util.StructForEach(userConfig.ModelProfiles, func(fieldName string, fieldValue any) {
		if fieldValue == nil {
			// If the field is nil, set it to the corresponding base.ModelProfiles
			baseVal := reflect.ValueOf(base.ModelProfiles)
			baseField := baseVal.FieldByName(fieldName)
			if baseField.IsValid() {
				userConfigVal := reflect.ValueOf(userConfig.ModelProfiles).Elem()
				userConfigField := userConfigVal.FieldByName(fieldName)
				if userConfigField.IsValid() && userConfigField.CanSet() {
					userConfigField.Set(baseField)
				}
			}
		}
	})

	if userConfig.Summarization == nil {
		userConfig.Summarization = &base.Summarization
	}
	if userConfig.Memory == nil {
		userConfig.Memory = &base.Memory
	}
	if userConfig.WebSearch == nil {
		userConfig.WebSearch = &base.WebSearch
	}
	if userConfig.Preferences == nil {
		userConfig.Preferences = &base.Preferences
	}
	if userConfig.Refinement == nil {
		userConfig.Refinement = &base.Refinement
	}
	if userConfig.ImageGeneration == nil {
		userConfig.ImageGeneration = &base.ImageGeneration
	}
}
