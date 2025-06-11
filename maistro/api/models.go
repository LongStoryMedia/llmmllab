package api

import (
	"maistro/auth"
	"maistro/config"
	"maistro/models"
	"maistro/storage"

	"maistro/util"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

func RegisterModelProfileRoutes(app *fiber.App) {
	app.Get("/api/model-profiles", ListModelProfiles)
	app.Get("/api/model-profiles/:id", GetModelProfile)
	app.Post("/api/model-profiles", CreateModelProfile)
	app.Put("/api/model-profiles/:id", UpdateModelProfile)
	app.Delete("/api/model-profiles/:id", DeleteModelProfile)
}

// ListModelProfiles returns all model profiles for the authenticated user
func ListModelProfiles(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	profiles, err := storage.ModelProfileStoreInstance.ListModelProfilesByUser(c.UserContext(), userID)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to retrieve model profiles")
	}

	// Append static default profiles (from context/models.go)
	defaultProfiles := []models.ModelProfile{
		config.DefaultSummarizationProfile,
		config.DefaultMasterSummaryProfile,
		config.DefaultBriefSummaryProfile,
		config.DefaultKeyPointsProfile,
		config.DefaultSelfCritiqueProfile,
		config.DefaultPrimaryProfile,
		config.DefaultMemoryRetrievalProfile,
		config.DefaultImprovementProfile,
		config.DefaultAnalysisProfile,
		config.DefaultResearchTaskProfile,
		config.DefaultResearchPlanProfile,
		config.DefaultResearchConsolidationProfile,
		config.DefaultResearchAnalysisProfile,
		config.DefaultEmbeddingProfile,
		config.DefaultFormattingProfile,
	}
	// Convert to pointer slice for consistency
	for _, def := range defaultProfiles {
		profiles = append(profiles, &def)
	}

	util.LogInfo("User model profiles retrieved", logrus.Fields{
		"userId":       userID,
		"profileCount": len(profiles),
	})

	return c.JSON(profiles)
}

// GetModelProfile returns a specific model profile
func GetModelProfile(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)
	profileID := c.Params("id")

	if err := uuid.Validate(profileID); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid model profile ID")
	}

	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(c.UserContext(), uuid.MustParse(profileID))
	if err != nil {
		return fiber.NewError(fiber.StatusNotFound, "Model profile not found")
	}

	// Verify ownership
	if profile.UserID != userID {
		return fiber.NewError(fiber.StatusForbidden, "Access denied")
	}

	return c.JSON(profile)
}

// CreateModelProfile creates a new model profile
func CreateModelProfile(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	var profile models.ModelProfile
	if err := c.BodyParser(&profile); err != nil {
		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	profile.UserID = userID

	_, err := storage.ModelProfileStoreInstance.CreateModelProfile(c.UserContext(), &profile)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to create model profile")
	}

	return c.JSON(profile)
}

// UpdateModelProfile updates an existing model profile
func UpdateModelProfile(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	var profile models.ModelProfile
	if err := c.BodyParser(&profile); err != nil {
		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	profile.UserID = userID

	if err := storage.ModelProfileStoreInstance.UpdateModelProfile(c.UserContext(), &profile); err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to update model profile")
	}

	return c.JSON(profile)
}

// DeleteModelProfile deletes a model profile
func DeleteModelProfile(c *fiber.Ctx) error {
	profileID := c.Params("id")

	if err := uuid.Validate(profileID); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid model profile ID")
	}

	if err := storage.ModelProfileStoreInstance.DeleteModelProfile(c.UserContext(), uuid.MustParse(profileID)); err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to delete model profile")
	}

	return c.SendStatus(fiber.StatusNoContent)
}
