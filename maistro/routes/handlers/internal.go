package handlers

import (
	"maistro/config"
	"maistro/util"
	"os"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// InternalGetUserImage fetches a user's image for internal services without requiring user auth
// This is used by the inference service to get images for editing
func InternalGetUserImage(c *fiber.Ctx) error {
	userID := c.Params("userID")
	filename := c.Params("filename")
	conf := config.GetConfig(nil)

	// Validate user ID and filename to prevent directory traversal attacks
	if userID == "" || filename == "" ||
		filepath.IsAbs(userID) || filepath.IsAbs(filename) ||
		filepath.Base(userID) != userID || filepath.Base(filename) != filename {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"success": false,
			"error":   "Invalid user ID or filename",
		})
	}

	// Path to the locally stored image
	filePath := filepath.Join(conf.ImageGeneration.StorageDirectory, userID, filename)

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"success": false,
			"error":   "Image not found",
		})
	}

	// Add security headers to prevent browser caching and embedding
	c.Set("Cache-Control", "no-store, no-cache, must-revalidate")
	c.Set("Pragma", "no-cache")
	c.Set("X-Content-Type-Options", "nosniff")

	// Log the access
	util.LogInfo("Internal API accessed image", logrus.Fields{
		"userID":   userID,
		"filename": filename,
	})

	// Serve the file directly with appropriate content type
	return c.SendFile(filePath)
}
