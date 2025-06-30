// Deprecated: Use the routes package instead
package api

// import (
// 	"context"
// 	"encoding/json"
// 	"fmt"
// 	"io"
// 	"maistro/auth"
// 	"maistro/config"
// 	"maistro/models"
// 	"maistro/proxy"
// 	"maistro/storage"
// 	"net/http"
// 	"sync"

// 	"maistro/util"

// 	"github.com/gofiber/fiber/v2"
// 	"github.com/google/uuid"
// 	"github.com/sirupsen/logrus"
// )

// // Deprecated: Use the routes package instead
// func RegisterModelProfileRoutes(app *fiber.App) {
// 	app.Get("/api/model-profiles", ListModelProfiles)
// 	app.Get("/api/model-profiles/:id", GetModelProfile)
// 	app.Post("/api/model-profiles", CreateModelProfile)
// 	app.Put("/api/model-profiles/:id", UpdateModelProfile)
// 	app.Delete("/api/model-profiles/:id", DeleteModelProfile)
// 	app.Get("/api/models", ListModels)
// }

// func getInferenceModels(ctx context.Context, url string) ([]models.Model, error) {
// 	// Get Models from ollama
// 	res, err := proxy.ProxyRequest(ctx, http.MethodGet, url, false, nil)
// 	if err != nil {
// 		return nil, handleError(err, fiber.StatusInternalServerError, fmt.Sprintf("Failed to retrieve models from %v", url))
// 	}
// 	defer res.Body.Close()

// 	if res.StatusCode != http.StatusOK {
// 		return nil, fiber.NewError(fiber.StatusInternalServerError, fmt.Sprintf("Failed to retrieve models from %v: %v", url, res.StatusCode))
// 	}

// 	var modelsResponse []byte
// 	if modelsResponse, err = io.ReadAll(res.Body); err != nil {
// 		return nil, handleError(err, fiber.StatusInternalServerError, "Failed to read models response")
// 	}

// 	var modelsData struct {
// 		Models []models.Model `json:"models"`
// 	}
// 	if err := json.Unmarshal(modelsResponse, &modelsData); err != nil {
// 		return nil, handleError(err, fiber.StatusInternalServerError, "Failed to parse models response")
// 	}

// 	return modelsData.Models, nil
// }

// func ListModels(c *fiber.Ctx) error {
// 	// Retrieve the user ID from the context
// 	conf := config.GetConfig(nil)
// 	var modelList []models.Model

// 	errChan := make(chan error, 2)
// 	wg := sync.WaitGroup{}

// 	// Get models from Ollama and Stable Diffusion in parallel
// 	wg.Add(1)
// 	go func(ctx context.Context, mdls *[]models.Model) {
// 		defer wg.Done()
// 		// Get Models from Ollama
// 		m, err := getInferenceModels(ctx, conf.InferenceServices.Ollama.BaseURL+"/api/tags")
// 		if err != nil {
// 			errChan <- err
// 			return
// 		}
// 		util.LogInfo("Ollama models retrieved", logrus.Fields{"modelCount": len(m)})
// 		*mdls = append(*mdls, m...)
// 	}(c.UserContext(), &modelList)

// 	wg.Add(1)
// 	go func(ctx context.Context, mdls *[]models.Model) {
// 		defer wg.Done()
// 		// Get Models from Stable Diffusion
// 		m, err := getInferenceModels(ctx, conf.InferenceServices.StableDiffusion.BaseURL+"/models")
// 		if err != nil {
// 			errChan <- err
// 			return
// 		}
// 		util.LogInfo("Stable Diffusion models retrieved", logrus.Fields{"modelCount": len(m)})
// 		*mdls = append(*mdls, m...)
// 	}(c.UserContext(), &modelList)

// 	wg.Wait()
// 	select {
// 	case err := <-errChan:
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to retrieve models")
// 	default:
// 	}

// 	return c.JSON(modelList)
// }

// // ListModelProfiles returns all model profiles for the authenticated user
// func ListModelProfiles(c *fiber.Ctx) error {
// 	userID := c.UserContext().Value(auth.UserIDKey).(string)

// 	profiles, err := storage.ModelProfileStoreInstance.ListModelProfilesByUser(c.UserContext(), userID)
// 	if err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to retrieve model profiles")
// 	}
// 	// Convert to pointer slice for consistency
// 	for _, def := range config.DefaultModelProfiles {
// 		profiles = append(profiles, &def)
// 	}

// 	util.LogInfo("User model profiles retrieved", logrus.Fields{
// 		"userId":       userID,
// 		"profileCount": len(profiles),
// 	})

// 	return c.JSON(profiles)
// }

// // GetModelProfile returns a specific model profile
// func GetModelProfile(c *fiber.Ctx) error {
// 	userID := c.UserContext().Value(auth.UserIDKey).(string)
// 	profileID := c.Params("id")

// 	if err := uuid.Validate(profileID); err != nil {
// 		return fiber.NewError(fiber.StatusBadRequest, "Invalid model profile ID")
// 	}

// 	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(c.UserContext(), uuid.MustParse(profileID))
// 	if err != nil {
// 		return fiber.NewError(fiber.StatusNotFound, "Model profile not found")
// 	}

// 	// Verify ownership
// 	if profile.UserID != userID {
// 		return fiber.NewError(fiber.StatusForbidden, "Access denied")
// 	}

// 	return c.JSON(profile)
// }

// // CreateModelProfile creates a new model profile
// func CreateModelProfile(c *fiber.Ctx) error {
// 	userID := c.UserContext().Value(auth.UserIDKey).(string)

// 	var profile models.ModelProfile
// 	if err := c.BodyParser(&profile); err != nil {
// 		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
// 	}

// 	profile.UserID = userID

// 	_, err := storage.ModelProfileStoreInstance.CreateModelProfile(c.UserContext(), &profile)
// 	if err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to create model profile")
// 	}

// 	return c.JSON(profile)
// }

// // UpdateModelProfile updates an existing model profile
// func UpdateModelProfile(c *fiber.Ctx) error {
// 	userID := c.UserContext().Value(auth.UserIDKey).(string)

// 	var profile models.ModelProfile
// 	if err := c.BodyParser(&profile); err != nil {
// 		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
// 	}

// 	profile.UserID = userID

// 	if err := storage.ModelProfileStoreInstance.UpdateModelProfile(c.UserContext(), &profile); err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to update model profile")
// 	}

// 	return c.JSON(profile)
// }

// // DeleteModelProfile deletes a model profile
// func DeleteModelProfile(c *fiber.Ctx) error {
// 	profileID := c.Params("id")

// 	if err := uuid.Validate(profileID); err != nil {
// 		return fiber.NewError(fiber.StatusBadRequest, "Invalid model profile ID")
// 	}

// 	if err := storage.ModelProfileStoreInstance.DeleteModelProfile(c.UserContext(), uuid.MustParse(profileID)); err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to delete model profile")
// 	}

// 	return c.SendStatus(fiber.StatusNoContent)
// }
