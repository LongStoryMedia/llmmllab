package auth

import (
	"context"
	"maistro/config"
	"maistro/util"
	"net/http"
	"strings"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// contextKey is a custom type for context keys to avoid collisions
type contextKey string

// Key for user ID in context
const UserIDKey contextKey = "user_id"
const TokenClaimsKey contextKey = "token_claims"
const IsAdminKey contextKey = "is_admin"

func NewValidator(ctx context.Context, jwksUri string) keyfunc.Keyfunc {
	k, err := keyfunc.NewDefaultCtx(ctx, []string{jwksUri}) // Context is used to end the refresh goroutine.
	if err != nil {
		// Log context, then fatal
		util.LogInfo("Failed to create a keyfunc.Keyfunc from the server's URL.", logrus.Fields{"error": err})
		logrus.Fatal("Failed to create a keyfunc.Keyfunc from the server's URL.")
	}
	return k
}

func WithAuth(c *fiber.Ctx) error {
	authHeader := c.Get("Authorization")
	conf := config.GetConfig(nil)

	k := NewValidator(c.UserContext(), conf.Auth.JWKSUri)

	if authHeader == "" {
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}
	tokenStr := strings.TrimPrefix(authHeader, "Bearer ")
	token, err := jwt.Parse(tokenStr, k.Keyfunc)
	if err != nil {
		util.LogWarning("Failed to parse token", logrus.Fields{"error": err})
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}
	if !token.Valid {
		util.LogWarning("Invalid token")
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}
	claims := token.Claims.(jwt.MapClaims)
	userID, ok := claims["sub"].(string)
	if !ok {
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}

	ctx := context.WithValue(c.Context(), UserIDKey, userID)
	ctx = context.WithValue(ctx, TokenClaimsKey, claims)
	groups, ok := claims["groups"]
	if ok {
		groupsInterface, ok := groups.([]any)
		if ok {
			for _, group := range groupsInterface {
				if groupStr, ok := group.(string); ok && groupStr == "admins" {
					ctx = context.WithValue(ctx, IsAdminKey, true)
					break
				}
			}
		}
	}

	c.SetUserContext(ctx)
	c.Locals(UserIDKey, userID)
	c.Locals(TokenClaimsKey, claims)
	c.Locals(IsAdminKey, ctx.Value(IsAdminKey) != nil)
	c.Set("X-User-ID", userID)

	// Generate and set request ID
	rid := uuid.New().String()
	c.Set("X-Request-ID", rid)
	c.Locals("request_id", rid)
	util.LogInfo("Request authorized", logrus.Fields{
		string(UserIDKey): userID,
	})
	return c.Next()
}

func CanAccess(c *fiber.Ctx, targetUserID string) bool {
	userID := c.UserContext().Value(UserIDKey).(string)
	return userID != targetUserID && c.UserContext().Value(IsAdminKey) == nil
}
