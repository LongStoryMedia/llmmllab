package handlers

import (
	"maistro/auth"
	"maistro/context"
	"maistro/models"
	"maistro/util"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// GetUserConfig returns the user's configuration, merged with system defaults
func GetUserConfig(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	// Use context package cache-aware function
	effectiveConfig, err := context.GetUserConfig(userID)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve user configuration")
	}

	return c.JSON(effectiveConfig)
}

// UpdateUserConfig updates a user's configuration settings
func UpdateUserConfig(c *fiber.Ctx) error {
	cfg := getUserConfig(c)
	// Parse the incoming user config
	var newUserConfig models.UserConfig
	if err := c.BodyParser(&newUserConfig); err != nil {
		return handleApiError(err, fiber.StatusBadRequest, "Invalid configuration format")
	}

	// Preserve important fields and merge configurations
	newUserConfig.UserID = cfg.UserID

	// Merge in existing fields if they weren't included in the update
	// Summarization config
	if newUserConfig.Summarization == nil && cfg.Summarization != nil {
		newUserConfig.Summarization = cfg.Summarization
		util.LogDebug("Preserving existing summarization config", logrus.Fields{"userID": cfg.UserID})
	}

	// Memory config
	if newUserConfig.Memory == nil && cfg.Memory != nil {
		newUserConfig.Memory = cfg.Memory
		util.LogDebug("Preserving existing memory config", logrus.Fields{"userID": cfg.UserID})
	}

	// Web search config
	if newUserConfig.WebSearch == nil && cfg.WebSearch != nil {
		newUserConfig.WebSearch = cfg.WebSearch
		util.LogDebug("Preserving existing web search config", logrus.Fields{"userID": cfg.UserID})
	}

	// Preferences config
	if newUserConfig.Preferences == nil && cfg.Preferences != nil {
		newUserConfig.Preferences = cfg.Preferences
		util.LogDebug("Preserving existing preferences config", logrus.Fields{"userID": cfg.UserID})
	}

	// Refinement config
	if newUserConfig.Refinement == nil && cfg.Refinement != nil {
		newUserConfig.Refinement = cfg.Refinement
		util.LogDebug("Preserving existing refinement config", logrus.Fields{"userID": cfg.UserID})
	}

	// ImageGeneration config
	if newUserConfig.ImageGeneration == nil && cfg.ImageGeneration != nil {
		newUserConfig.ImageGeneration = cfg.ImageGeneration
		util.LogDebug("Preserving existing image generation config", logrus.Fields{"userID": cfg.UserID})
	}

	// Special handling for ModelProfiles to ensure they aren't lost if partial update
	if newUserConfig.ModelProfiles != nil {
		// Updating model profiles for user
		util.LogDebug("Updating model profiles for user", logrus.Fields{"userID": cfg.UserID})
	} else if cfg.ModelProfiles != nil {
		// If the request doesn't include model profiles but we have existing ones, preserve them
		newUserConfig.ModelProfiles = cfg.ModelProfiles
		util.LogDebug("Preserving existing model profiles for user", logrus.Fields{"userID": cfg.UserID})
	}

	// Use context package cache-aware function to update
	if err := context.SetUserConfig(&newUserConfig); err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to update user configuration")
	}

	util.LogInfo("User configuration updated", logrus.Fields{"userID": cfg.UserID})

	// Return the new config
	return c.JSON(newUserConfig)
}
