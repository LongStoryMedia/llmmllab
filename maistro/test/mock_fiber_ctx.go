package test

import (
	"context"
	"maistro/auth"

	"github.com/gofiber/fiber/v2"
)

// NewMockApp returns a new mock Fiber application with a sample context.
func NewMockApp() *fiber.App {
	app := fiber.New()

	app.Use(func(c *fiber.Ctx) error {
		ctx := context.WithValue(c.Context(), auth.UserIDKey, "CgNsc20SBGxkYXA") // lsm
		ctx = context.WithValue(ctx, auth.IsAdminKey, true)
		c.SetUserContext(ctx)
		c.Set("Access-Control-Allow-Origin", "*")
		c.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		c.Set("Access-Control-Max-Age", "3600")
		return c.Next()
	})

	return app
}

// MockReq sets up a mock request on the given fiber.Ctx for testing.
// You can specify method, path, body, headers, and query params.
func MockReq(c *fiber.Ctx, method, path string, body []byte, headers map[string]string, queryParams map[string]string) error {
	c.Request().Header.SetMethod(method)
	c.Request().SetRequestURI(path)
	if body != nil {
		c.Request().SetBody(body)
	}
	for k, v := range headers {
		c.Request().Header.Set(k, v)
	}
	if len(queryParams) > 0 {
		q := c.Request().URI().QueryArgs()
		for k, v := range queryParams {
			q.Set(k, v)
		}
		c.Request().URI().SetQueryString(q.String())
	}
	return nil
}
