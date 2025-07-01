package routes

import (
	"maistro/middleware"
	"maistro/routes/handlers"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
)

var (
	wsConf = websocket.Config{
		HandshakeTimeout: 10 * time.Second,
	}
)

func RegisterRoutes(app *fiber.App) {
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
	})

	app.Get("/health", func(c *fiber.Ctx) error {
		// Simple health check endpoint
		return c.SendStatus(fiber.StatusNoContent)
	})

	socket := app.Group("/ws")
	socket.Get("/chat/:"+string(middleware.CIDPKey), middleware.SocketAuth, websocket.New(handlers.HandleChatSocket, wsConf))
	socket.Get("/image", middleware.SocketAuth, websocket.New(handlers.HandleImageSocket, wsConf))
	socket.Get("/status", middleware.SocketAuth, websocket.New(handlers.HandleStatusSocket, wsConf))

	// Internal API endpoints - these don't use the standard auth middleware
	// but are secured with an API key for internal service-to-service communication
	internal := app.Group("/internal")
	internal.Use(middleware.InternalServiceAuth) // Custom middleware to verify internal service requests
	internal.Get("/images/:userID/:filename", handlers.InternalGetUserImage)

	// Public image endpoints with token-based security
	// These don't use auth middleware so they can be used in img tags
	static := app.Group("/static")
	publicImages := static.Group("/images")
	publicImages.Get("/view/:filename", handlers.ServeImage)
	publicImages.Get("/download/:filename", handlers.DownloadImage)

	api := app.Group("/api")
	api.Use(middleware.WithAuth, middleware.Config)

	convo := api.Group("/conversations")
	convo.Get("/", handlers.GetUserConversations)
	convo.Post("/", handlers.CreateConversation)
	convo.Get("/:"+string(middleware.CIDPKey), middleware.Convo, middleware.Session, handlers.GetConversation)
	convo.Delete("/:"+string(middleware.CIDPKey), middleware.Convo, middleware.Session, handlers.DeleteConversation)
	convo.Get("/:"+string(middleware.CIDPKey)+"/messages", middleware.Convo, middleware.Session, handlers.GetConversationMessages)
	convo.Post("/:"+string(middleware.CIDPKey)+"/messages", middleware.Convo, middleware.Session, handlers.ChatHandler)
	convo.Post("/:"+string(middleware.CIDPKey)+"/pause", middleware.Convo, middleware.Session, handlers.Pause)
	convo.Post("/:"+string(middleware.CIDPKey)+"/resume", middleware.Convo, middleware.Session, handlers.Resume)
	convo.Delete("/:"+string(middleware.CIDPKey)+"/cancel", middleware.Convo, middleware.Session, handlers.Cancel)

	users := api.Group("/users")
	users.Get("/", handlers.GetUsers)
	users.Get("/:"+string(middleware.UIDPKey)+"/conversations", handlers.GetConversationsForUser)

	models := api.Group("/models")
	models.Get("/", handlers.ListModels)
	models.Get("/profiles", handlers.ListModelProfiles)
	models.Post("/profiles", handlers.CreateModelProfile)
	models.Get("/profiles/:"+string(middleware.MPPKey), handlers.GetModelProfile)
	models.Put("/profiles/:"+string(middleware.MPPKey), handlers.UpdateModelProfile)
	models.Delete("/profiles/:"+string(middleware.MPPKey), handlers.DeleteModelProfile)
	models.Put("/image/:model", handlers.SetActiveImageModel)

	// Protected image endpoints requiring authentication
	images := api.Group("/images")
	images.Get("/", handlers.GetUserImages)
	images.Post("/generate", handlers.GenerateImage)
	images.Post("/edit", handlers.EditImage)
	images.Delete("/:imageID", handlers.DeleteImage)

	config := api.Group("/config")
	config.Get("/", handlers.GetUserConfig)
	config.Put("/", handlers.UpdateUserConfig)
}
