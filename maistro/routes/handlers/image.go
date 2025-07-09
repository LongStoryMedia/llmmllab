package handlers

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"maistro/auth"
	"maistro/config"
	pxcx "maistro/context"
	"maistro/models"
	svc "maistro/services"
	"maistro/storage"
	"maistro/util"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// generateImageToken creates a secure token for accessing an image
// Format: base64(expiry_timestamp:user_id:filename:hmac_signature)
func generateImageToken(userID, filename string, expiryMinutes int) string {
	conf := config.GetConfig(nil)
	secret := conf.Internal.APIKey

	// Set expiry time
	expiry := time.Now().Add(time.Duration(expiryMinutes) * time.Minute).Unix()

	// Create the message to sign (expiry:user_id:filename)
	message := fmt.Sprintf("%d:%s:%s", expiry, userID, filename)

	// Create HMAC signature
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(message))
	signature := h.Sum(nil)

	// Combine message and signature
	token := fmt.Sprintf("%s:%s", message, base64.StdEncoding.EncodeToString(signature))

	// Base64 encode the whole thing
	return base64.URLEncoding.EncodeToString([]byte(token))
}

// validateImageToken validates a token and returns userID and filename if valid
func validateImageToken(token string) (userID, filename string, valid bool) {
	conf := config.GetConfig(nil)
	secret := conf.Internal.APIKey

	// Decode the token
	tokenBytes, err := base64.URLEncoding.DecodeString(token)
	if err != nil {
		return "", "", false
	}

	// Split into parts
	parts := strings.Split(string(tokenBytes), ":")
	if len(parts) != 4 {
		return "", "", false
	}

	// Extract parts
	expiryStr, userID, filename, signatureB64 := parts[0], parts[1], parts[2], parts[3]

	// Check expiry
	expiry, err := strconv.ParseInt(expiryStr, 10, 64)
	if err != nil {
		return "", "", false
	}

	if time.Now().Unix() > expiry {
		return "", "", false // Token expired
	}

	// Verify signature
	message := fmt.Sprintf("%s:%s:%s", expiryStr, userID, filename)
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(message))
	expectedSignature := base64.StdEncoding.EncodeToString(h.Sum(nil))

	if signatureB64 != expectedSignature {
		return "", "", false // Invalid signature
	}

	return userID, filename, true
}

// createSignedImageURL creates a URL with a secure token for accessing an image
func createSignedImageURL(baseURL, userID, filename string, isDownload bool) string {
	// Generate a token valid for 60 minutes
	token := generateImageToken(userID, filename, 60)

	// Create the URL path
	path := "view"
	if isDownload {
		path = "download"
	}

	return fmt.Sprintf("%s/static/images/%s/%s?token=%s",
		baseURL, path, filename, token)
}

// DownloadImage serves the locally stored image as a downloadable attachment
func DownloadImage(c *fiber.Ctx) error {
	filename := c.Params("filename")
	token := c.Query("token")

	// Validate token
	userID, tokenFilename, valid := validateImageToken(token)
	if !valid {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "Invalid or expired token",
		})
	}

	// Verify filename matches the one in the token
	if filename != tokenFilename {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "Token doesn't match requested image",
		})
	}

	conf := config.GetConfig(nil)

	// Path to the locally stored image
	filePath := filepath.Join(conf.ImageGeneration.StorageDirectory, userID, filename)

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return handleApiError(err, fiber.StatusNotFound, "Image not found")
	}

	// Serve the file as an attachment for download
	c.Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Set("Content-Type", "application/octet-stream")
	return c.SendFile(filePath)
}

// ServeImage serves the locally stored image for direct viewing (e.g. in <img> tags)
func ServeImage(c *fiber.Ctx) error {
	filename := c.Params("filename")
	token := c.Query("token")

	// Validate token
	userID, tokenFilename, valid := validateImageToken(token)
	if !valid {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "Invalid or expired token",
		})
	}

	// Verify filename matches the one in the token
	if filename != tokenFilename {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "Token doesn't match requested image",
		})
	}

	conf := config.GetConfig(nil)

	// Path to the locally stored image
	filePath := filepath.Join(conf.ImageGeneration.StorageDirectory, userID, filename)

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return handleApiError(err, fiber.StatusNotFound, "Image not found")
	}

	// Serve the file directly with appropriate content type
	return c.SendFile(filePath)
}

// GetUserImages retrieves all images for the current user
func GetUserImages(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)
	// Parse query parameters
	limit := 50 // Default limit
	offset := 0 // Default offset
	var conversationID *int

	if c.Query("limit") != "" {
		parsedLimit, err := strconv.Atoi(c.Query("limit"))
		if err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	if c.Query("offset") != "" {
		parsedOffset, err := strconv.Atoi(c.Query("offset"))
		if err == nil && parsedOffset >= 0 {
			offset = parsedOffset
		}
	}

	if c.Query("conversation_id") != "" {
		parsedConversationID, err := strconv.Atoi(c.Query("conversation_id"))
		if err == nil {
			conversationID = &parsedConversationID
		}
	}

	// Get images from storage
	images, err := storage.ImageStoreInstance.ListImages(c.Context(), userID, conversationID, &limit, &offset)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve images")
	}

	// Add view_url and download_url to each image with secure tokens
	conf := config.GetConfig(nil)
	baseURL := conf.Server.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080" // Fallback for local development
	}

	for i := range images {
		viewURL := createSignedImageURL(baseURL, userID, images[i].Filename, false)
		downloadURL := createSignedImageURL(baseURL, userID, images[i].Filename, true)

		images[i].ViewURL = util.StrPtr(viewURL)
		images[i].DownloadURL = util.StrPtr(downloadURL)
	}

	return c.JSON(images)
}

func GenerateImage(c *fiber.Ctx) error {
	cfg := getUserConfig(c)
	var req models.ImageGenerateRequest
	if err := c.BodyParser(&req); err != nil {
		return handleApiError(err, fiber.StatusBadRequest, "Invalid request body")
	}
	p := req.Prompt
	np := req.NegativePrompt
	if np == nil {
		np = util.StrPtr("anime, cartoon, sketch, drawing, lowres, bad anatomy, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username")
	}

	if cfg.ImageGeneration.AutoPromptRefinement {
		fp, err := storage.ModelProfileStoreInstance.GetModelProfile(c.UserContext(), cfg.ModelProfiles.ImageGenerationPromptProfileID)
		if err != nil {
			return handleApiError(err, fiber.StatusInternalServerError, "Failed to get model profile")
		}

		ctx, cancel := context.WithTimeout(c.UserContext(), time.Minute)
		defer cancel()
		p, err = pxcx.FmtQuery(ctx, fp, req.Prompt, pxcx.ImgGeneratePrompt)
		if err != nil {
			return util.HandleError(err)
		}
	}

	// Prepare the image generation request
	imgReq := models.ImageGenerateRequest{
		Prompt:         p,
		NegativePrompt: np,
		Model:          "stabilityai-stable-diffusion-3.5-medium",
	}

	svc.GetInferenceService().GenerateImage(c.UserContext(), cfg.UserID, -99, imgReq)
	return c.JSON(fiber.Map{
		"status":  "success",
		"message": "Image generation request submitted successfully",
	})
}

func EditImage(c *fiber.Ctx) error {
	cfg := getUserConfig(c)
	var req models.ImageGenerateRequest
	if err := c.BodyParser(&req); err != nil {
		return handleApiError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	// Validate image ID (should be provided in the request)
	if req.ImageID == nil {
		return handleApiError(nil, fiber.StatusBadRequest, "No image ID provided")
	}

	// Look up the image in storage
	imageID := *req.ImageID
	userID := cfg.UserID
	imageMetadata, err := storage.ImageStoreInstance.GetImageByID(c.Context(), userID, imageID)
	if err != nil {
		return handleApiError(err, fiber.StatusNotFound, "Image not found")
	}

	// Validate image metadata
	if imageMetadata == nil {
		return handleApiError(nil, fiber.StatusNotFound, "Image not found")
	}

	// Get the configuration
	conf := config.GetConfig(nil)

	// Path to the locally stored image
	filePath := filepath.Join(conf.ImageGeneration.StorageDirectory, userID, imageMetadata.Filename)

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return handleApiError(err, fiber.StatusNotFound, "Image file not found on disk")
	}

	// Open the image file to be uploaded
	file, err := os.Open(filePath)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to open image file")
	}
	defer file.Close()

	url := fmt.Sprintf("%s/store-image", conf.InferenceServices.StableDiffusion.BaseURL)

	// Create a buffer to store the form data
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	// Create form file field
	part, err := writer.CreateFormFile("image", imageMetadata.Filename)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to create form file")
	}

	// Copy the file content to the form field
	if _, err = io.Copy(part, file); err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to copy file content")
	}

	// Close the writer to finalize the form data
	if err = writer.Close(); err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to close multipart writer")
	}

	// Create the request to upload the image
	httpReq, err := http.NewRequestWithContext(c.Context(), "POST", url, &body)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to create upload request")
	}

	httpReq.Header.Set("Content-Type", writer.FormDataContentType())

	// Send the upload request
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to upload image to inference service")
	}
	defer resp.Body.Close()

	// Check the response status
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return handleApiError(fmt.Errorf("inference service returned status %d: %s", resp.StatusCode, string(respBody)),
			fiber.StatusInternalServerError, "Failed to store image in inference service")
	}

	// Process prompt and negative prompt
	p := req.Prompt
	np := req.NegativePrompt
	if np == nil {
		np = util.StrPtr("anime, cartoon, sketch, drawing, lowres, bad anatomy, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username")
	}

	if cfg.ImageGeneration.AutoPromptRefinement {
		fp, err := storage.ModelProfileStoreInstance.GetModelProfile(c.UserContext(), cfg.ModelProfiles.ImageGenerationPromptProfileID)
		if err != nil {
			return handleApiError(err, fiber.StatusInternalServerError, "Failed to get model profile")
		}

		ctx, cancel := context.WithTimeout(c.UserContext(), time.Minute)
		defer cancel()
		p, err = pxcx.FmtQuery(ctx, fp, req.Prompt, pxcx.ImgGeneratePrompt)
		if err != nil {
			return util.HandleError(err)
		}
	}

	// Create the final request with all parameters
	imgReq := models.ImageGenerateRequest{
		Prompt:         p,
		NegativePrompt: np,
		Width:          req.Width,
		Height:         req.Height,
		InferenceSteps: req.InferenceSteps,
		GuidanceScale:  req.GuidanceScale,
		LowMemoryMode:  req.LowMemoryMode,
		ImageID:        imageMetadata.ID,
		Model:          req.Model,
		Filename:       util.StrPtr(imageMetadata.Filename),
	}

	// Add the unique filename we used when uploading the image
	// This replaces the URL parameter that was used before
	util.LogInfo("Image uploaded to inference service", logrus.Fields{
		"userId":  userID,
		"imageId": imageID,
	})

	// Modify the imgReq to include the uploaded filename

	// Send the edit request to the inference service
	svc.GetInferenceService().EditImage(c.UserContext(), cfg.UserID, -99, imgReq)

	return c.JSON(fiber.Map{
		"status":  "success",
		"message": "Image uploaded and edit request submitted successfully",
	})
}

// DeleteImage removes an image from the user's storage
func DeleteImage(c *fiber.Ctx) error {
	imageIDStr := c.Params("imageID")
	if imageIDStr == "" {
		return handleApiError(nil, fiber.StatusBadRequest, "Image ID is required")
	}

	imageID, err := strconv.Atoi(imageIDStr)
	if err != nil {
		return handleApiError(err, fiber.StatusBadRequest, "Invalid image ID format")
	}

	userID := c.UserContext().Value(auth.UserIDKey).(string)

	// Delete the image from storage
	if err := storage.ImageStoreInstance.DeleteImage(c.Context(), imageID); err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to delete image")
	}

	util.LogInfo("Image deleted successfully", logrus.Fields{
		"userID":  userID,
		"imageID": imageID,
	})

	return c.JSON(fiber.Map{
		"status":  "success",
		"message": "Image deleted successfully",
	})
}
