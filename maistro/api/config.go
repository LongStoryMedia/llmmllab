package api

import (
	"maistro/auth"
	"maistro/config"
	"maistro/context"
	"maistro/util"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// RegisterConfigRoutes adds configuration management endpoints
func RegisterConfigRoutes(app *fiber.App) {
	app.Get("/api/config", GetUserConfig)
	app.Put("/api/config", UpdateUserConfig)
}

// GetUserConfig returns the user's configuration, merged with system defaults
func GetUserConfig(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	// Use context package cache-aware function
	effectiveConfig, err := context.GetUserConfig(userID)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to retrieve user configuration")
	}

	return c.JSON(effectiveConfig)
}

// UpdateUserConfig updates a user's configuration settings
func UpdateUserConfig(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	// Parse the incoming user config
	var newUserConfig config.UserConfig
	if err := c.BodyParser(&newUserConfig); err != nil {
		return handleError(err, fiber.StatusBadRequest, "Invalid configuration format")
	}

	// Get current user config to ensure we don't lose existing settings
	currentConfig, err := context.GetUserConfig(userID)
	if err != nil {
		// If we can't get current config, we'll just use what was provided
		currentConfig = &config.UserConfig{UserID: userID}
	}

	// Preserve important fields
	newUserConfig.UserID = userID

	// Special handling for ModelProfiles to ensure they aren't lost if partial update
	if newUserConfig.ModelProfiles != nil {
		// Updating model profiles for user
	} else if currentConfig.ModelProfiles != nil {
		// If the request doesn't include model profiles but we have existing ones, preserve them
		newUserConfig.ModelProfiles = currentConfig.ModelProfiles
		// Preserving existing model profiles for user
	}

	// Use context package cache-aware function to update
	if err := context.SetUserConfig(&newUserConfig); err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to update user configuration")
	}

	util.LogInfo("User configuration updated", logrus.Fields{"userID": userID})

	// Return the new config
	return c.JSON(newUserConfig)
}
