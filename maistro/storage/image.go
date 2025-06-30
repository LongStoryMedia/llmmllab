package storage

import (
	"context"
	"fmt"
	"maistro/models"
	"maistro/util"
)

// ImageStore abstracts image-related operations
type imageStore struct{}

func (s *imageStore) StoreImage(ctx context.Context, userID string, image *models.ImageMetadata) (int, error) {
	// Check if Pool is initialized
	if Pool == nil {
		return 0, util.HandleError(fmt.Errorf("database connection pool is not initialized (Pool is nil)"))
	}
	// Use the SQL query from our loader to insert the image metadata
	var imageID int
	// filename, thumbnail, format, width, height, conversation_id, user_id
	err := Pool.QueryRow(ctx, GetQuery("images.add_image"), image.Filename, image.Thumbnail, image.Format, image.Width, image.Height, image.ConversationID, image.UserID).
		Scan(&imageID)
	if err != nil {
		return 0, util.HandleError(err)
	}

	return imageID, nil // Return image ID and nil error
}

func (s *imageStore) ListImages(ctx context.Context, userID string, conversationID, limit, offset *int) ([]models.ImageMetadata, error) {
	// Check if Pool is initialized
	if Pool == nil {
		return nil, util.HandleError(fmt.Errorf("database connection pool is not initialized (Pool is nil)"))
	}

	// Use the SQL query from our loader to list images for the user
	rows, err := Pool.Query(ctx, GetQuery("images.list_images"), userID, conversationID, limit, offset)
	if err != nil {
		return nil, util.HandleError(err)
	}
	defer rows.Close()
	var images []models.ImageMetadata
	for rows.Next() {
		var image models.ImageMetadata
		err := rows.Scan(&image.ID, &image.Filename, &image.Thumbnail, &image.Format, &image.Width, &image.Height, &image.ConversationID, &image.UserID)
		if err != nil {
			return nil, util.HandleError(err)
		}
		images = append(images, image)
	}
	if err := rows.Err(); err != nil {
		return nil, util.HandleError(err)
	}

	return images, nil // Return the list of images and nil error
}

func (s *imageStore) DeleteImage(ctx context.Context, imageID int) error {
	// Check if Pool is initialized
	if Pool == nil {
		return util.HandleError(fmt.Errorf("database connection pool is not initialized (Pool is nil)"))
	}

	// Use the SQL query from our loader to delete the image
	_, err := Pool.Exec(ctx, GetQuery("images.delete_image"), imageID)
	if err != nil {
		return util.HandleError(err)
	}
	return nil // Return nil error for now
}
