package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maistro/config"
	"maistro/util"
	"net/http"
	"time"

	"github.com/sirupsen/logrus"
)

// StableDiffusionGenerateRequest represents a request to generate an image
type StableDiffusionGenerateRequest struct {
	Prompt         string   `json:"prompt"`
	Width          *int     `json:"width,omitempty"`
	Height         *int     `json:"height,omitempty"`
	InferenceSteps *int     `json:"inference_steps,omitempty"`
	GuidanceScale  *float32 `json:"guidance_scale,omitempty"`
	LowMemoryMode  *bool    `json:"low_memory_mode,omitempty"`
	NegativePrompt *string  `json:"negative_prompt,omitempty"`
}

// StableDiffusionGenerateResponse represents the response from the image generation API
type StableDiffusionGenerateResponse struct {
	Image    string `json:"image"`    // base64-encoded image
	Download string `json:"download"` // path to download the image
}

// GenerateImage sends a request to the Stable Diffusion API to generate an image
// from the given prompt and parameters
func GenerateImage(ctx context.Context, request StableDiffusionGenerateRequest) (*StableDiffusionGenerateResponse, error) {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return nil, fmt.Errorf("stable diffusion base URL not configured")
	}

	// Validate the request
	if request.Prompt == "" {
		return nil, fmt.Errorf("prompt cannot be empty")
	}

	// Log the request parameters
	util.LogInfo("Sending image generation request to Stable Diffusion API", logrus.Fields{
		"prompt":         request.Prompt,
		"width":          request.Width,
		"height":         request.Height,
		"inferenceSteps": request.InferenceSteps,
		"guidanceScale":  request.GuidanceScale,
	})

	// Prepare the request body
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error marshaling image generation request: %w", err))
	}

	// Create the request
	url := fmt.Sprintf("%s/generate-image", conf.InferenceServices.StableDiffusion.BaseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error creating image generation request: %w", err))
	}
	req.Header.Set("Content-Type", "application/json")

	// Create HTTP client with an extended timeout for image generation
	client := &http.Client{
		Timeout: 2 * time.Minute, // Images might take longer to generate
	}

	// Execute the request
	resp, err := client.Do(req)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error calling stable diffusion API: %w", err))
	}
	defer resp.Body.Close()

	// Check for non-200 responses
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, util.HandleError(fmt.Errorf("stable diffusion API returned status %d: %s", resp.StatusCode, string(bodyBytes)))
	}

	// Read and parse the response
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error reading image generation response: %w", err))
	}

	var response StableDiffusionGenerateResponse
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		return nil, util.HandleError(fmt.Errorf("error parsing image generation response: %w", err))
	}

	util.LogInfo("Successfully generated image", logrus.Fields{
		"downloadPath": response.Download,
	})

	return &response, nil
}

// DownloadImage sends a request to download a generated image by its filename
func DownloadImage(ctx context.Context, filename string) ([]byte, string, error) {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return nil, "", fmt.Errorf("stable diffusion base URL not configured")
	}

	// Create the request
	url := fmt.Sprintf("%s/download/%s", conf.InferenceServices.StableDiffusion.BaseURL, filename)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", util.HandleError(fmt.Errorf("error creating image download request: %w", err))
	}

	// Create HTTP client
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Execute the request
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", util.HandleError(fmt.Errorf("error calling stable diffusion API for image download: %w", err))
	}
	defer resp.Body.Close()

	// Check for non-200 responses
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, "", util.HandleError(fmt.Errorf("stable diffusion API returned status %d: %s", resp.StatusCode, string(bodyBytes)))
	}

	// Get content type
	contentType := resp.Header.Get("Content-Type")

	// Read the image data
	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", util.HandleError(fmt.Errorf("error reading image data: %w", err))
	}

	return imageData, contentType, nil
}

// GetStableDiffusionModels retrieves the list of available Stable Diffusion models
func GetStableDiffusionModels(ctx context.Context) ([]byte, error) {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return nil, fmt.Errorf("stable diffusion base URL not configured")
	}

	return sendStableDiffusionRequest(ctx, "/models/", http.MethodGet, nil)
}

// SetActiveStableDiffusionModel sets the active Stable Diffusion model
func SetActiveStableDiffusionModel(ctx context.Context, modelID string) error {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return fmt.Errorf("stable diffusion base URL not configured")
	}

	path := fmt.Sprintf("/models/active/%s", modelID)
	_, err := sendStableDiffusionRequest(ctx, path, http.MethodPut, nil)
	return err
}

// GetStableDiffusionLoras retrieves the list of available LoRAs
func GetStableDiffusionLoras(ctx context.Context) ([]byte, error) {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return nil, fmt.Errorf("stable diffusion base URL not configured")
	}

	return sendStableDiffusionRequest(ctx, "/loras/", http.MethodGet, nil)
}

// ActivateStableDiffusionLora activates a LoRA for use with image generation
func ActivateStableDiffusionLora(ctx context.Context, loraID string) error {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return fmt.Errorf("stable diffusion base URL not configured")
	}

	path := fmt.Sprintf("/loras/%s/activate", loraID)
	_, err := sendStableDiffusionRequest(ctx, path, http.MethodPut, nil)
	return err
}

// DeactivateStableDiffusionLora deactivates a LoRA
func DeactivateStableDiffusionLora(ctx context.Context, loraID string) error {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return fmt.Errorf("stable diffusion base URL not configured")
	}

	path := fmt.Sprintf("/loras/%s/deactivate", loraID)
	_, err := sendStableDiffusionRequest(ctx, path, http.MethodPut, nil)
	return err
}

// SetStableDiffusionLoraWeight sets the weight for a specific LoRA
func SetStableDiffusionLoraWeight(ctx context.Context, loraID string, weight float32) error {
	conf := config.GetConfig(nil)
	if conf.InferenceServices.StableDiffusion.BaseURL == "" {
		return fmt.Errorf("stable diffusion base URL not configured")
	}

	reqBody, err := json.Marshal(map[string]float32{"weight": weight})
	if err != nil {
		return util.HandleError(fmt.Errorf("error marshaling weight request: %w", err))
	}

	path := fmt.Sprintf("/loras/%s/weight", loraID)
	_, err = sendStableDiffusionRequest(ctx, path, http.MethodPut, reqBody)
	return err
}

// Helper function to send requests to the Stable Diffusion API
func sendStableDiffusionRequest(ctx context.Context, path string, method string, body []byte) ([]byte, error) {
	conf := config.GetConfig(nil)
	url := conf.InferenceServices.StableDiffusion.BaseURL + path

	var reqBody io.Reader
	if body != nil {
		reqBody = bytes.NewReader(body)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error creating request: %w", err))
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error sending request: %w", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, util.HandleError(fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(bodyBytes)))
	}

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("error reading response: %w", err))
	}

	return responseBody, nil
}
