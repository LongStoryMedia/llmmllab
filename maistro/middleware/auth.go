package middleware

import (
	"context"
	"maistro/auth"
	"maistro/util"
	"net/http"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

type Actions struct {
	Read   bool `json:"read"`
	Write  bool `json:"write"`
	Delete bool `json:"delete"`
}

type What struct {
	Conversations Actions `json:"conversations"`
	Messages      Actions `json:"messages"`
	Images        Actions `json:"images"`
	Models        Actions `json:"models"`
	Settings      Actions `json:"settings"`
	Research      Actions `json:"research"`
}

type Can struct {
	Others What `json:"others"`
	My     What `json:"my"`
}

func WithAuth(c *fiber.Ctx) error {
	authHeader := c.Get("Authorization")
	if authHeader == "" {
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}

	tokenStr := strings.TrimPrefix(authHeader, "Bearer ")
	result, err := auth.ValidateToken(tokenStr, c.UserContext())
	if err != nil {
		util.LogWarning("Token validation failed", logrus.Fields{"error": err})
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}

	if result == nil {
		util.LogWarning("Token validation failed", logrus.Fields{"error": err})
		return c.Status(http.StatusForbidden).JSON(fiber.Map{
			"error": "Forbidden",
		})
	}

	// Set up context with auth information
	ctx := context.WithValue(c.Context(), auth.UserIDKey, result.UserID)
	ctx = context.WithValue(ctx, auth.TokenClaimsKey, result.Claims)
	if result.IsAdmin {
		ctx = context.WithValue(ctx, auth.IsAdminKey, true)
	}

	// Update context and local values
	c.SetUserContext(ctx)
	c.Locals(auth.UserIDKey, result.UserID)
	c.Locals(auth.TokenClaimsKey, result.Claims)
	c.Locals(auth.IsAdminKey, result.IsAdmin)
	c.Set("X-User-ID", result.UserID)

	// Generate and set request ID
	rid := uuid.New().String()
	c.Set("X-Request-ID", rid)
	c.Locals("request_id", rid)

	util.LogInfo("Request authorized", logrus.Fields{
		string(auth.UserIDKey): result.UserID,
	})
	return c.Next()
}
