package context

import (
	"context"
	"fmt"
	"image"
	"image/png"
	"io"
	"maistro/config"
	"maistro/models"
	"maistro/storage"
	"maistro/util"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

var (
	// Used to ensure only one cleanup is running at a time
	cleanupMutex sync.Mutex
	// Track if the cleanup routine has been started
	cleanupRoutineStarted bool = false
)

func (cc *conversationContext) AddImage(ctx context.Context, imageMetadata *models.ImageMetadata, downloadPath string) (int, error) {
	conf := config.GetConfig(nil)
	basename := strings.TrimPrefix(downloadPath, "/download/")
	thumbName := fmt.Sprintf("thumbnail_%s", basename)

	// Ensure user exists in the database before proceeding
	if err := storage.EnsureUser(ctx, cc.userID); err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to ensure user exists: %w", err))
	}
	util.LogDebug("Adding image to conversation context", logrus.Fields{
		"userID":   cc.userID,
		"imageID":  imageMetadata.ID,
		"filename": basename,
		"thumb":    thumbName,
	})

	// download the image to the local storage
	url := conf.InferenceServices.StableDiffusion.BaseURL + downloadPath

	// Ensure the directory exists
	err := os.MkdirAll(fmt.Sprintf("%s/%s", conf.ImageGeneration.StorageDirectory, cc.userID), os.ModePerm)
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to create directory: %w", err))
	}
	util.LogDebug("Creating image file", logrus.Fields{
		"storageDir": conf.ImageGeneration.StorageDirectory,
		"userID":     cc.userID,
		"filename":   basename,
	})

	out, err := os.Create(fmt.Sprintf("%s/%s/%s", conf.ImageGeneration.StorageDirectory, cc.userID, basename))
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to create file: %w", err))
	}
	defer out.Close()

	res, err := http.Get(url)
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to download image: %w", err))
	}
	defer res.Body.Close()

	util.LogDebug("Downloading image", logrus.Fields{
		"url":      url,
		"userID":   cc.userID,
		"filename": basename,
	})

	if res.StatusCode != http.StatusOK {
		return 0, util.HandleError(fmt.Errorf("failed to download image: %s", res.Status))
	}

	_, err = io.Copy(out, res.Body)
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to read response body: %w", err))
	}
	util.LogDebug("Image downloaded successfully", logrus.Fields{
		"url":      url,
		"userID":   cc.userID,
		"filename": basename,
	})

	// create the thumbnail
	img, err := os.Open(fmt.Sprintf("%s/%s/%s", conf.ImageGeneration.StorageDirectory, cc.userID, basename))
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to open image file: %w", err))
	}
	defer img.Close()

	util.LogDebug("Creating thumbnail for image", logrus.Fields{
		"storageDir": conf.ImageGeneration.StorageDirectory,
		"userID":     cc.userID,
		"filename":   basename,
		"thumb":      thumbName,
	})

	t, err := os.Create(fmt.Sprintf("%s/%s/%s", conf.ImageGeneration.StorageDirectory, cc.userID, thumbName))
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to create thumbnail file: %w", err))
	}
	defer t.Close()

	util.LogDebug("Encoding thumbnail image", logrus.Fields{
		"storageDir": conf.ImageGeneration.StorageDirectory,
		"userID":     cc.userID,
		"filename":   basename,
		"thumb":      thumbName,
	})

	err = png.Encode(t, image.NewRGBA(image.Rect(0, 0, 128, 128))) // Create a new image for the thumbnail
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to store image: %w", err))
	}

	imageMetadata.Thumbnail = thumbName // Set the thumbnail name in the metadata
	imageMetadata.Format = "png"        // Set the format of the image
	imageMetadata.Filename = basename   // Set the filename in the metadata

	// Store the image in the storage
	id, err := storage.ImageStoreInstance.StoreImage(ctx, cc.userID, imageMetadata)
	if err != nil {
		return 0, util.HandleError(fmt.Errorf("failed to store image: %w", err))
	}

	util.LogDebug("Image stored successfully", logrus.Fields{
		"userID":   cc.userID,
		"imageID":  id,
		"filename": basename,
		"thumb":    thumbName,
	})

	imageMetadata.ID = &id                        // Set the ID returned from the storage
	cc.images = append(cc.images, *imageMetadata) // Update the conversation context with the new image

	return id, nil
}

// StartImageCleanupRoutine starts a background goroutine that periodically cleans up old images
// based on the RetentionHours configuration. It ensures only one cleanup routine is running.
func StartImageCleanupRoutine() {
	cleanupMutex.Lock()
	defer cleanupMutex.Unlock()

	if cleanupRoutineStarted {
		return // Already running
	}

	conf := config.GetConfig(nil)
	if !conf.ImageGeneration.Enabled {
		return // Image generation is disabled
	}

	// create the storage directory if it doesn't exist
	if err := os.MkdirAll(conf.ImageGeneration.StorageDirectory, os.ModePerm); err != nil {
		util.HandleError(err)
		return
	}

	go func() {
		// Start with a slightly random delay to avoid scheduling many tasks at exactly the same time
		time.Sleep(time.Duration(30+time.Now().Second()) * time.Second)

		// Run immediately on startup
		CleanupOldImages()

		// Set up a ticker to run cleanup periodically (every 1 hour is reasonable)
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()

		for range ticker.C {
			CleanupOldImages()
		}
	}()

	cleanupRoutineStarted = true
	util.LogInfo("Image cleanup routine started", logrus.Fields{
		"retentionHours": config.GetConfig(nil).ImageGeneration.RetentionHours,
	})
}

// CleanupOldImages deletes image files that are older than the RetentionHours configuration
func CleanupOldImages() {
	cleanupMutex.Lock()
	defer cleanupMutex.Unlock()

	conf := config.GetConfig(nil)
	if !conf.ImageGeneration.Enabled || conf.ImageGeneration.StorageDirectory == "" {
		return // Nothing to do
	}

	// Calculate the cutoff time
	retentionPeriod := time.Duration(conf.ImageGeneration.RetentionHours) * time.Hour
	cutoffTime := time.Now().Add(-retentionPeriod)

	// Log the cleanup operation start
	util.LogInfo("Starting cleanup of old images", logrus.Fields{
		"retentionHours": conf.ImageGeneration.RetentionHours,
		"cutoffTime":     cutoffTime.Format(time.RFC3339),
	})

	storage.ImageStoreInstance.DeleteImagesOlderThan(context.Background(), cutoffTime) // Delete from database

	// Walk through the storage directory
	err := filepath.Walk(conf.ImageGeneration.StorageDirectory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err // Skip this file/directory on error
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Check if the file is old enough to be deleted
		if info.ModTime().Before(cutoffTime) {
			// It's an old file, delete it
			err := os.Remove(path)
			if err != nil {
				util.LogWarning(fmt.Sprintf("Failed to delete old image: %s", path), logrus.Fields{
					"error": err.Error(),
				})
				return nil // Continue with other files
			}

			util.LogDebug(fmt.Sprintf("Deleted old image: %s", path), logrus.Fields{
				"modTime": info.ModTime().Format(time.RFC3339),
			})
		}

		return nil
	})

	if err != nil {
		util.LogWarning("Error during image cleanup", logrus.Fields{
			"error": err.Error(),
		})
	} else {
		util.LogInfo("Image cleanup completed successfully", nil)
	}
}
