package main

import (
	"fmt"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"

	"maistro/api"
	"maistro/config"
	"maistro/context"
	"maistro/storage"
	"maistro/util"
)

func main() {
	conf := config.GetConfig(nil)

	// Set logrus log level from config
	switch conf.LogLevel {
	case "debug":
		logrus.SetLevel(logrus.DebugLevel)
	case "info":
		logrus.SetLevel(logrus.InfoLevel)
	case "warn":
		logrus.SetLevel(logrus.WarnLevel)
	case "error":
		logrus.SetLevel(logrus.ErrorLevel)
	default:
		logrus.SetLevel(logrus.InfoLevel)
	}
	fmtter := new(logrus.TextFormatter)
	fmtter.TimestampFormat = "2006-01-02 15:04:05"
	logrus.SetFormatter(fmtter)
	fmtter.FullTimestamp = true

	psqlconn := fmt.Sprintf(
		"postgresql://%s:%s@%s:%d/%s?sslmode=%s",
		conf.Database.User,
		conf.Database.Password,
		conf.Database.Host,
		conf.Database.Port,
		conf.Database.Dbname,
		conf.Database.Sslmode,
	)

	if err := storage.InitializeStorage(); err != nil {
		util.HandleFatalError(err)
	}

	if err := storage.InitDB(psqlconn); err != nil {
		util.HandleFatalErrorAtCallLevel(err, 1)
	}
	util.LogInfo("Connected to PostgreSQL database", logrus.Fields{
		"connection": psqlconn,
	})

	// Initialize research schema
	// storage.InitResearchSchema(ctx)

	// Initialize Redis for storage caching
	if err := storage.InitStorageCache(); err != nil {
		util.LogWarning("Failed to initialize Redis storage cache", logrus.Fields{
			"error": err,
		})
	} else if conf.Redis.Enabled {
		util.LogInfo("Redis storage cache initialized successfully")
		// Clean up Redis connections when the application exits
		defer storage.CloseRedisCache()
	}

	// Initialize the conversation cache with configured settings
	if conf.Redis.Enabled {
		util.LogInfo("Initializing conversation cache with Redis", logrus.Fields{
			"host": conf.Redis.Host,
			"port": conf.Redis.Port,
			"ttl":  conf.Redis.ConversationTtl,
		})
	} else {
		util.LogInfo("Redis disabled, using in-memory cache", logrus.Fields{
			"ttl": conf.Redis.ConversationTtl,
		})
	}
	duration := time.Duration(conf.Redis.ConversationTtl) * time.Second

	context.InitCache(duration)
	if conf.ImageGeneration.Enabled {
		context.StartImageCleanupRoutine()
	}
	// Create a new Fiber app
	app := fiber.New(fiber.Config{
		DisableStartupMessage: false,
		// Add streaming capability
		StreamRequestBody: true,
	})

	// Router
	api.RegisterAllRoutes(app)

	// Start the server
	addr := fmt.Sprintf("%s:%d", conf.Server.Host, conf.Server.Port)
	util.LogInfo(fmt.Sprintf("Starting server on %s", addr))
	if err := app.Listen(addr); err != nil {
		util.HandleFatalError(err)
	}
}
