package proxy

import (
	"context"
	"maistro/config"
	"os"
	"testing"
	"time"
)

func Test_GenerateImage(t *testing.T) {
	// Skip this test if SKIP_IMAGE_TESTS is set (useful for CI environments)
	if os.Getenv("SKIP_IMAGE_TESTS") != "" {
		t.Skip("Skipping image generation test")
	}

	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Set up test parameters
	width := 512
	height := 512
	steps := 20
	guidance := float32(7.5)
	lowMemory := false

	request := StableDiffusionGenerateRequest{
		Prompt:         "a beautiful sunset over mountains, photorealistic",
		Width:          &width,
		Height:         &height,
		InferenceSteps: &steps,
		GuidanceScale:  &guidance,
		LowMemoryMode:  &lowMemory,
	}

	response, err := GenerateImage(ctx, request)
	if err != nil {
		t.Fatalf("Failed to generate image: %v", err)
	}

	if response.Image == "" {
		t.Fatal("Empty image data received")
	}

	if response.Download == "" {
		t.Fatal("No download path received")
	}

	t.Logf("Successfully generated image with download path: %s", response.Download)
}

func Test_DownloadImage(t *testing.T) {
	// Skip this test if SKIP_IMAGE_TESTS is set
	if os.Getenv("SKIP_IMAGE_TESTS") != "" {
		t.Skip("Skipping image download test")
	}

	// This test depends on Test_GenerateImage, so we'll first generate an image
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	request := StableDiffusionGenerateRequest{
		Prompt: "a cat wearing sunglasses, digital art",
	}

	response, err := GenerateImage(ctx, request)
	if err != nil {
		t.Fatalf("Failed to generate image for download test: %v", err)
	}

	// Extract the filename from the download path
	// The path should be something like "/download/filename.png"
	downloadPath := response.Download
	filename := ""
	if len(downloadPath) > 10 {
		filename = downloadPath[10:] // Skip "/download/"
	}

	if filename == "" {
		t.Fatal("Failed to extract filename from download path")
	}

	// Now test the download functionality
	imageData, contentType, err := DownloadImage(ctx, filename)
	if err != nil {
		t.Fatalf("Failed to download image: %v", err)
	}

	if len(imageData) == 0 {
		t.Fatal("Downloaded empty image data")
	}

	if contentType != "image/png" && contentType != "image/jpeg" {
		t.Fatalf("Unexpected content type: %s", contentType)
	}

	t.Logf("Successfully downloaded image with content type: %s and size: %d bytes", contentType, len(imageData))
}
