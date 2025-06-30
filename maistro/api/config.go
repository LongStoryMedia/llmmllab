// Deprecated: Use the routes package instead
package api

// import (
// 	"maistro/auth"
// 	"maistro/context"
// 	"maistro/models"
// 	"maistro/util"

// 	"github.com/gofiber/fiber/v2"
// 	"github.com/sirupsen/logrus"
// )

// // Deprecated: RegisterConfigRoutes adds configuration management endpoints
// func RegisterConfigRoutes(app *fiber.App) {
// 	app.Get("/api/config", GetUserConfig)
// 	app.Put("/api/config", UpdateUserConfig)
// }

// // GetUserConfig returns the user's configuration, merged with system defaults
// func GetUserConfig(c *fiber.Ctx) error {
// 	userID := c.UserContext().Value(auth.UserIDKey).(string)

// 	// Use context package cache-aware function
// 	effectiveConfig, err := context.GetUserConfig(userID)
// 	if err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to retrieve user configuration")
// 	}

// 	return c.JSON(effectiveConfig)
// }

// // UpdateUserConfig updates a user's configuration settings
// func UpdateUserConfig(c *fiber.Ctx) error {
// 	userID := c.UserContext().Value(auth.UserIDKey).(string)
// 	// Parse the incoming user config
// 	var newUserConfig models.UserConfig
// 	if err := c.BodyParser(&newUserConfig); err != nil {
// 		return handleError(err, fiber.StatusBadRequest, "Invalid configuration format")
// 	}

// 	// Get current user config to ensure we don't lose existing settings
// 	currentConfig, err := context.GetUserConfig(userID)
// 	if err != nil {
// 		// If we can't get current config, we'll just use what was provided
// 		currentConfig = &models.UserConfig{UserID: userID}
// 	}

// 	// Preserve important fields and merge configurations
// 	newUserConfig.UserID = userID

// 	// Merge in existing fields if they weren't included in the update
// 	// Summarization config
// 	if newUserConfig.Summarization == nil && currentConfig.Summarization != nil {
// 		newUserConfig.Summarization = currentConfig.Summarization
// 		util.LogDebug("Preserving existing summarization config", logrus.Fields{"userID": userID})
// 	}

// 	// Memory config
// 	if newUserConfig.Memory == nil && currentConfig.Memory != nil {
// 		newUserConfig.Memory = currentConfig.Memory
// 		util.LogDebug("Preserving existing memory config", logrus.Fields{"userID": userID})
// 	}

// 	// Web search config
// 	if newUserConfig.WebSearch == nil && currentConfig.WebSearch != nil {
// 		newUserConfig.WebSearch = currentConfig.WebSearch
// 		util.LogDebug("Preserving existing web search config", logrus.Fields{"userID": userID})
// 	}

// 	// Preferences config
// 	if newUserConfig.Preferences == nil && currentConfig.Preferences != nil {
// 		newUserConfig.Preferences = currentConfig.Preferences
// 		util.LogDebug("Preserving existing preferences config", logrus.Fields{"userID": userID})
// 	}

// 	// Refinement config
// 	if newUserConfig.Refinement == nil && currentConfig.Refinement != nil {
// 		newUserConfig.Refinement = currentConfig.Refinement
// 		util.LogDebug("Preserving existing refinement config", logrus.Fields{"userID": userID})
// 	}

// 	// ImageGeneration config
// 	if newUserConfig.ImageGeneration == nil && currentConfig.ImageGeneration != nil {
// 		newUserConfig.ImageGeneration = currentConfig.ImageGeneration
// 		util.LogDebug("Preserving existing image generation config", logrus.Fields{"userID": userID})
// 	}

// 	// Special handling for ModelProfiles to ensure they aren't lost if partial update
// 	if newUserConfig.ModelProfiles != nil {
// 		// Updating model profiles for user
// 		util.LogDebug("Updating model profiles for user", logrus.Fields{"userID": userID})
// 	} else if currentConfig.ModelProfiles != nil {
// 		// If the request doesn't include model profiles but we have existing ones, preserve them
// 		newUserConfig.ModelProfiles = currentConfig.ModelProfiles
// 		util.LogDebug("Preserving existing model profiles for user", logrus.Fields{"userID": userID})
// 	}

// 	// Use context package cache-aware function to update
// 	if err := context.SetUserConfig(&newUserConfig); err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to update user configuration")
// 	}

// 	util.LogInfo("User configuration updated", logrus.Fields{"userID": userID})

// 	// Return the new config
// 	return c.JSON(newUserConfig)
// }
