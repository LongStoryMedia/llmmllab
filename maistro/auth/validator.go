package auth

import (
	"context"
	"errors"
	"maistro/config"
	"maistro/models"
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
		util.HandleFatalError(err, logrus.Fields{"error": "Failed to create a keyfunc.Keyfunc from the server's URL."})
	}
	return k
}

// ValidateToken handles the common token validation logic
// Returns user ID, claims, and error if validation fails
func ValidateToken(tokenStr string, ctx context.Context) (*models.TokenValidationResult, error) {
	conf := config.GetConfig(nil)
	k := NewValidator(ctx, conf.Auth.JwksURI)

	token, err := jwt.Parse(tokenStr, k.Keyfunc)
	if err != nil {
		util.LogWarning("Failed to parse token", logrus.Fields{"error": err})
		return nil, errors.New("invalid token")
	}

	if !token.Valid {
		util.LogWarning("Invalid token", nil)
		return nil, errors.New("invalid token")
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return nil, errors.New("invalid token claims")
	}

	userID, ok := claims["sub"].(string)
	if !ok {
		return nil, errors.New("user ID not found in token")
	}

	// Check for admin status
	isAdmin := false
	groups, ok := claims["groups"]
	if ok {
		groupsInterface, ok := groups.([]any)
		if ok {
			for _, group := range groupsInterface {
				if groupStr, ok := group.(string); ok && groupStr == "admins" {
					isAdmin = true
					break
				}
			}
		}
	}

	return &models.TokenValidationResult{
		UserID:  userID,
		Claims:  claims,
		IsAdmin: isAdmin,
	}, nil
}

// ValidateAndGetUserID validates a JWT token from a string and returns the user ID
// This is used especially for WebSocket connections that pass token via query param
func ValidateAndGetUserID(tokenStr string) (string, error) {
	result, err := ValidateToken(tokenStr, context.Background())
	if err != nil {
		return "", err
	}
	if result.UserID == "" {
		return "", errors.New("user ID not found in token")
	}

	return result.UserID, nil
}

func WithAuth(c *fiber.Ctx) error {
	authHeader := c.Get("Authorization")
	if authHeader == "" {
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}

	tokenStr := strings.TrimPrefix(authHeader, "Bearer ")
	result, err := ValidateToken(tokenStr, c.UserContext())

	if result == nil {
		util.LogWarning("Token validation failed", logrus.Fields{"error": err})
		return c.Status(http.StatusForbidden).JSON(fiber.Map{
			"error": "Forbidden",
		})
	}

	if err != nil {
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{
			"error": "Unauthorized",
		})
	}

	// Set up context with auth information
	ctx := context.WithValue(c.Context(), UserIDKey, result.UserID)
	ctx = context.WithValue(ctx, TokenClaimsKey, result.Claims)
	if result.IsAdmin {
		ctx = context.WithValue(ctx, IsAdminKey, true)
	}

	// Update context and local values
	c.SetUserContext(ctx)
	c.Locals(UserIDKey, result.UserID)
	c.Locals(TokenClaimsKey, result.Claims)
	c.Locals(IsAdminKey, result.IsAdmin)
	c.Set("X-User-ID", result.UserID)

	// Generate and set request ID
	rid := uuid.New().String()
	c.Set("X-Request-ID", rid)
	c.Locals("request_id", rid)

	util.LogInfo("Request authorized", logrus.Fields{
		string(UserIDKey): result.UserID,
	})
	return c.Next()
}

func CanAccess(c *fiber.Ctx, targetUserID string) bool {
	userID := c.UserContext().Value(UserIDKey).(string)
	isAdmin := c.UserContext().Value(IsAdminKey) != nil || c.Locals(IsAdminKey) != nil

	return isAdmin || userID == targetUserID
}
