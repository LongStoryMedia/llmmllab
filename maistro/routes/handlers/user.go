package handlers

import (
	"maistro/auth"
	"maistro/middleware"
	"maistro/storage"

	"github.com/gofiber/fiber/v2"
)

func GetUsers(c *fiber.Ctx) error {
	users, err := storage.UserConfigStoreInstance.GetAllUsers(c.UserContext())
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve users")
	}

	return c.JSON(users)
}

// GetConversationsForUser returns all conversations for a specific user
func GetConversationsForUser(c *fiber.Ctx) error {
	uid := c.Params(string(middleware.UIDPKey), "")
	if uid == "" {
		return fiber.NewError(fiber.StatusBadRequest, "User ID is required")
	}

	if !auth.CanAccess(c, uid) {
		return fiber.NewError(fiber.StatusForbidden, "Access denied")
	}

	conversations, err := storage.ConversationStoreInstance.GetUserConversations(c.UserContext(), uid)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve conversations for user")
	}

	return c.JSON(conversations)
}
