package proxy

import (
	"context"
	"maistro/models"
	"os"
	"testing"
	"time"
)

func Test_GenerateImageFromPrompt(t *testing.T) {
	// Skip this test if SKIP_IMAGE_TESTS is set (useful for CI environments)
	if os.Getenv("SKIP_IMAGE_TESTS") != "" {
		t.Skip("Skipping image generation test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Set up test parameters
	width := 512
	height := 512
	steps := 20
	guidance := float32(7.5)
	lowMemory := false

	request := models.ImageGenerateRequest{
		Prompt:         "a colorful painting of mountains at sunset",
		Width:          &width,
		Height:         &height,
		InferenceSteps: &steps,
		GuidanceScale:  &guidance,
		LowMemoryMode:  &lowMemory,
	}

	response, err := GenerateImageFromPrompt(ctx, request)
	if err != nil {
		t.Fatalf("Failed to generate image: %v", err)
	}

	if response.Image == "" {
		t.Fatal("Empty image data received")
	}

	if response.Download == "" {
		t.Fatal("No download URL received")
	}

	t.Logf("Successfully generated image with download URL: %s", response.Download)
}

func Test_GetImageAsBase64(t *testing.T) {
	// Skip this test if SKIP_IMAGE_TESTS is set
	if os.Getenv("SKIP_IMAGE_TESTS") != "" {
		t.Skip("Skipping image download test")
	}

	// This test depends on Test_GenerateImageFromPrompt, so we'll first generate an image
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	request := models.ImageGenerateRequest{
		Prompt: "a digital artwork of a landscape with mountains and lakes",
	}

	response, err := GenerateImageFromPrompt(ctx, request)
	if err != nil {
		t.Fatalf("Failed to generate image for download test: %v", err)
	}

	// Extract the filename from the download URL
	// The path should be something like "/download/filename.png"
	downloadURL := response.Download
	filename := ""
	if len(downloadURL) > 10 {
		filename = downloadURL[10:] // Skip "/download/"
	}

	if filename == "" {
		t.Fatal("Failed to extract filename from download URL")
	}

	// Now test the GetImageAsBase64 functionality
	base64Data, contentType, err := GetImageAsBase64(ctx, filename)
	if err != nil {
		t.Fatalf("Failed to get image as base64: %v", err)
	}

	if base64Data == "" {
		t.Fatal("Empty base64 data received")
	}

	if contentType != "image/png" && contentType != "image/jpeg" {
		t.Fatalf("Unexpected content type: %s", contentType)
	}

	t.Logf("Successfully retrieved base64 image with content type: %s and length: %d", contentType, len(base64Data))
}
