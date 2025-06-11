package api

import (
	"maistro/auth"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
)

// RegisterAllRoutes registers all the API routes with the fiber app
func RegisterAllRoutes(app *fiber.App) {
	app.Use(logger.New(
		logger.Config{
			Format:     "${time} ${status} - ${latency} ${method} ${path}\n",
			TimeFormat: "2006-01-02 15:04:05",
			TimeZone:   "Local",
		},
	)).Use(func(c *fiber.Ctx) error {
		c.Set("Access-Control-Allow-Origin", "*")
		c.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		c.Set("Access-Control-Max-Age", "3600")
		return c.Next()
	}).Options("/*", func(c *fiber.Ctx) error {
		return c.SendStatus(fiber.StatusNoContent)
	}).Use(auth.WithAuth)

	RegisterConversationRoutes(app)
	RegisterResearchRoutes(app)
	RegisterWebSearchRoutes(app)
	RegisterConfigRoutes(app)
	RegisterModelProfileRoutes(app)
	RegisterStableDiffusionRoutes(app)

	// Setup reverse proxy handler with chunk processing
	app.All("/*", OllamaHandler)
}
