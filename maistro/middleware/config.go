package middleware

import (
	"context"
	"maistro/auth"
	"maistro/config"
	pxcx "maistro/context"

	"github.com/gofiber/fiber/v2"
)

func Config(c *fiber.Ctx) error {
	cfg, err := pxcx.GetUserConfig(c.UserContext().Value(auth.UserIDKey).(string))
	if err != nil {
		return fiber.NewError(fiber.StatusInternalServerError, "Failed to retrieve user configuration")
	}

	ctx := context.WithValue(c.UserContext(), config.CfgKey, cfg)
	c.SetUserContext(ctx)

	return c.Next()
}
