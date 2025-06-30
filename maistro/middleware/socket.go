package middleware

import (
	"context"
	"maistro/auth"

	"github.com/gofiber/contrib/websocket"
	"github.com/gofiber/fiber/v2"
)

func SocketAuth(c *fiber.Ctx) error {
	// Check for token in query parameters
	token := c.Query("token")
	if token == "" {
		return fiber.ErrForbidden
	}

	// Validate the token and get user ID
	userID, err := auth.ValidateAndGetUserID(token)
	if err != nil {
		return fiber.ErrUnauthorized
	}

	ctx := context.WithValue(c.UserContext(), UIDPKey, userID)

	cid, err := c.ParamsInt(string(CIDPKey), 0)
	if err == nil && cid > 0 {
		ctx = context.WithValue(ctx, CIDPKey, cid)
		c.Locals(string(CIDPKey), cid)
	}

	c.SetUserContext(ctx)
	c.Locals(string(UIDPKey), userID)

	// Only allow WebSocket upgrade requests
	if websocket.IsWebSocketUpgrade(c) {
		return c.Next()
	}

	return fiber.ErrUpgradeRequired
}
