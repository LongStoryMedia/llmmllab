// filepath: /Users/scott.long/workspace/llmmllab/maistro/proxy/image.go
package proxy

import (
	"context"
	"encoding/base64"
	"fmt"
	"maistro/models"
	"maistro/util"
)

// GenerateImage sends a request to generate an image based on the provided prompt and parameters
func GenerateImageFromPrompt(ctx context.Context, request models.ImageGenerateRequest) (*models.ImageGenerateResponse, error) {
	// Log the image generation attempt
	util.LogInfo(fmt.Sprintf("Generating image for prompt: %s", request.Prompt), nil)

	// Convert to StableDiffusionGenerateRequest
	sdRequest := StableDiffusionGenerateRequest{
		Prompt:         request.Prompt,
		Width:          request.Width,
		Height:         request.Height,
		InferenceSteps: request.InferenceSteps,
		GuidanceScale:  request.GuidanceScale,
		LowMemoryMode:  request.LowMemoryMode,
		NegativePrompt: request.NegativePrompt,
	}

	// Use the existing imago.go functionality to generate the image
	sdResponse, err := GenerateImage(ctx, sdRequest)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to generate image: %w", err))
	}

	if sdResponse == nil {
		return nil, util.HandleError(fmt.Errorf("received nil response from image generation"))
	}

	// Return the data in our response format
	return &models.ImageGenerateResponse{
		Image:    sdResponse.Image,
		Download: sdResponse.Download,
	}, nil
}

// GetImage retrieves an image by its filename
func GetImage(ctx context.Context, filename string) ([]byte, string, error) {
	// Use the existing imago.go functionality to download the image
	imageData, contentType, err := DownloadImage(ctx, filename)
	if err != nil {
		return nil, "", util.HandleError(fmt.Errorf("failed to download image: %w", err))
	}

	return imageData, contentType, nil
}

// GetImageAsBase64 retrieves an image and returns it as a base64 string
func GetImageAsBase64(ctx context.Context, filename string) (string, string, error) {
	// Download the image
	imageData, contentType, err := GetImage(ctx, filename)
	if err != nil {
		return "", "", err
	}

	// Convert to base64
	base64Data := base64.StdEncoding.EncodeToString(imageData)
	return base64Data, contentType, nil
}
