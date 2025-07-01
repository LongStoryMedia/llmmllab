package svc

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"maistro/config"
	pxcx "maistro/context"
	"maistro/models"
	"maistro/proxy"
	"maistro/storage"
	"maistro/util"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

type ImageService interface {
	// GenerateImage generates an image based on the provided prompt and returns the image URL.
	GenerateImage(ctx context.Context, userID string, conversationID int, originalRequest models.ImageGenerateRequest) (*models.ImageMetadata, error)
	SaveImage(igr *models.ImageGenerateResponse, conversationID int, userID string) (*models.ImageMetadata, error)
}

type imageService struct{}

var (
	ImgSvc = &imageService{}
)

func GetImgSvc() ImageService {
	return ImgSvc
}

// GenerateImage generates an image based on the provided prompt and model.
func (s *imageService) GenerateImage(ctx context.Context, userID string, conversationID int, originalRequest models.ImageGenerateRequest) (*models.ImageMetadata, error) {
	conf := config.GetConfig(nil)
	// Create a modified proxyStableDiffusionRequest with WebSocket notification
	targetURL := fmt.Sprintf("%s%s", conf.InferenceServices.StableDiffusion.BaseURL, "/generate-image")

	// Log the request
	util.LogInfo("Proxying image generation request to Stable Diffusion API", logrus.Fields{
		"url":    targetURL,
		"userId": userID,
	})

	body, err := json.Marshal(originalRequest)
	if err != nil {
		return nil, s.sendImageGenerationFailureNotification(conversationID, userID, fmt.Sprintf("Failed to marshal request body: %v", err))
	}

	resp, err := proxy.ProxyRequest(ctx, http.MethodPost, targetURL, false, body)
	if err != nil {
		return nil, s.sendImageGenerationFailureNotification(conversationID, userID, fmt.Sprintf("Failed to contact Stable Diffusion API: %v", err))
	}
	defer resp.Body.Close()

	util.LogDebug("Received response from Stable Diffusion API", logrus.Fields{
		"statusCode": resp.StatusCode,
		"userId":     userID,
	})

	// Read the response body
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, s.sendImageGenerationFailureNotification(conversationID, userID, fmt.Sprintf("Failed to read Stable Diffusion API response: %v", err))
	}

	// If the API returns an error, pass it through
	if resp.StatusCode >= 400 {
		util.LogWarning("Stable Diffusion API error", logrus.Fields{
			"statusCode": resp.StatusCode,
		})

		return nil, s.sendImageGenerationFailureNotification(conversationID, userID, fmt.Sprintf("Stable Diffusion API error: %d", resp.StatusCode))
	}

	util.LogDebug("Stable Diffusion API response body", logrus.Fields{
		"userId": userID,
	})

	// Parse the successful response to extract image details for the WebSocket notification
	var sdResponse models.ImageGenerateResponse
	var imgMetadata models.ImageMetadata

	if err := json.Unmarshal(responseBody, &sdResponse); err == nil && sdResponse.Download != "" {
		// Extract the image filename from the download URL
		// Format is typically /download/filename.png
		imageID := ""
		if len(sdResponse.Download) > 10 { // Check if URL is long enough
			imageID = sdResponse.Download[10:] // Skip the "/download/" prefix
		}

		util.LogDebug("Image generation successful", logrus.Fields{
			"imageID":     imageID,
			"downloadURL": sdResponse.Download,
			"userId":      userID,
		})

		// get current conversation context
		cc, err := pxcx.GetCachedConversation(userID, conversationID)
		if err != nil {
			util.LogWarning("Failed to get conversation context", logrus.Fields{
				"userId": userID,
				"error":  err.Error(),
			})

			return nil, s.sendImageGenerationFailureNotification(cc.ConversationID, userID, "Failed to retrieve conversation context")
		}

		imgMetadata, err := s.SaveImage(&sdResponse, cc.ConversationID, userID)
		if err != nil {
			util.LogWarning("Failed to add image to conversation context", logrus.Fields{
				"userId": userID,
				"error":  err.Error(),
			})

			return nil, s.sendImageGenerationFailureNotification(cc.ConversationID, userID, "Failed to add image to conversation context")
		}
		// Send success notification via WebSocket
		s.sendImageGenerationSuccessNotification(cc.ConversationID, userID, imgMetadata)
	}

	// Return the response to the client
	return &imgMetadata, nil
}

// sendImageGenerationFailureNotification sends a WebSocket notification for failed image generation
func (s *imageService) sendImageGenerationFailureNotification(conversationID int, userID, errorMessage string) error {
	return GetSocketService().SendError(models.SocketStageTypeGeneratingImage, conversationID, userID, errorMessage)
}

// sendImageGenerationSuccessNotification sends a WebSocket notification for successful image generation
func (s *imageService) sendImageGenerationSuccessNotification(conversationID int, userID string, payload *models.ImageMetadata) {
	GetSocketService().SendCompletion(models.SocketStageTypeGeneratingImage, conversationID, userID, "Image generation completed", payload)
}

func (s *imageService) SaveImage(igr *models.ImageGenerateResponse, conversationID int, userID string) (*models.ImageMetadata, error) {
	conf := config.GetConfig(nil)
	basename := strings.TrimPrefix(igr.Download, "/download/")
	thumbName := fmt.Sprintf("thumbnail_%s", basename)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute*5) // Set a timeout for the operation
	defer cancel()

	// Ensure user exists in the database before proceeding
	if err := storage.EnsureUser(ctx, userID); err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to ensure user exists: %w", err))
	}
	util.LogDebug("Adding image to conversation context", logrus.Fields{
		"userID":   userID,
		"filename": basename,
		"thumb":    thumbName,
	})

	// download the image to the local storage
	url := conf.InferenceServices.StableDiffusion.BaseURL + igr.Download

	// Ensure the directory exists
	err := os.MkdirAll(fmt.Sprintf("%s/%s", conf.ImageGeneration.StorageDirectory, userID), os.ModePerm)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to create directory: %w", err))
	}
	util.LogDebug("Creating image file", logrus.Fields{
		"storageDir": conf.ImageGeneration.StorageDirectory,
		"userID":     userID,
		"filename":   basename,
	})

	out, err := os.Create(fmt.Sprintf("%s/%s/%s", conf.ImageGeneration.StorageDirectory, userID, basename))
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to create file: %w", err))
	}
	defer out.Close()

	res, err := http.Get(url)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to download image: %w", err))
	}
	defer res.Body.Close()

	util.LogDebug("Downloading image", logrus.Fields{
		"url":      url,
		"userID":   userID,
		"filename": basename,
	})

	if res.StatusCode != http.StatusOK {
		return nil, util.HandleError(fmt.Errorf("failed to download image: %s", res.Status))
	}

	_, err = io.Copy(out, res.Body)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to read response body: %w", err))
	}
	util.LogDebug("Image downloaded successfully", logrus.Fields{
		"url":      url,
		"userID":   userID,
		"filename": basename,
	})

	// create the thumbnail
	img, err := os.Open(fmt.Sprintf("%s/%s/%s", conf.ImageGeneration.StorageDirectory, userID, basename))
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to open image file: %w", err))
	}
	defer img.Close()

	util.LogDebug("Creating thumbnail for image", logrus.Fields{
		"storageDir": conf.ImageGeneration.StorageDirectory,
		"userID":     userID,
		"filename":   basename,
		"thumb":      thumbName,
	})

	t, err := os.Create(fmt.Sprintf("%s/%s/%s", conf.ImageGeneration.StorageDirectory, userID, thumbName))
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to create thumbnail file: %w", err))
	}
	defer t.Close()

	util.LogDebug("Encoding thumbnail image", logrus.Fields{
		"storageDir": conf.ImageGeneration.StorageDirectory,
		"userID":     userID,
		"filename":   basename,
		"thumb":      thumbName,
	})

	err = png.Encode(t, image.NewRGBA(image.Rect(0, 0, 128, 128))) // Create a new image for the thumbnail
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to store image: %w", err))
	}

	imageMetadata := models.ImageMetadata{
		ID:             util.IntPtr(0), // Placeholder, will be set later
		Filename:       basename,
		Format:         "png",     // Assuming PNG format for generated images
		Thumbnail:      thumbName, // Placeholder for thumbnail
		CreatedAt:      time.Now(),
		ConversationID: &conversationID,
		UserID:         userID,
		ViewURL:        util.StrPtr(fmt.Sprintf("/static/images/view/%s", basename)),
		DownloadURL:    util.StrPtr(fmt.Sprintf("/static/images/%s", basename)),
	}

	// Store the image in the storage
	id, err := storage.ImageStoreInstance.StoreImage(ctx, userID, &imageMetadata)
	if err != nil {
		return &imageMetadata, util.HandleError(fmt.Errorf("failed to store image: %w", err))
	}

	util.LogDebug("Image stored successfully", logrus.Fields{
		"userID":   userID,
		"imageID":  id,
		"filename": basename,
		"thumb":    thumbName,
	})

	imageMetadata.ID = &id // Set the ID returned from the storage

	// Log the image URLs for debugging
	util.LogDebug("Image URLs generated", logrus.Fields{
		"viewURL":     imageMetadata.ViewURL,
		"downloadURL": imageMetadata.DownloadURL,
	})

	s.sendImageGenerationSuccessNotification(conversationID, userID, &imageMetadata)
	return &imageMetadata, nil
}
