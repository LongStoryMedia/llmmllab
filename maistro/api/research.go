package api

import (
	"github.com/gofiber/fiber/v2"

	"maistro/auth"
	"maistro/research"
)

// RegisterResearchRoutes adds research-related endpoints to the app
func RegisterResearchRoutes(app *fiber.App) {
	app.Post("/api/research/task", CreateResearchTask)
	app.Get("/api/research/tasks", ListResearchTasks)
	app.Get("/api/research/task/:taskID", GetResearchTask)
	app.Delete("/api/research/task/:taskID", CancelResearchTask)
}

// CreateResearchTask creates a new research task
func CreateResearchTask(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	var req research.CreateResearchRequest
	if err := c.BodyParser(&req); err != nil {
		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	// Validate request
	if req.Query == "" {
		return fiber.NewError(fiber.StatusBadRequest, "Query is required")
	}

	// Convert conversationID to pointer if present
	var conversationID *int
	if req.ConversationID != 0 {
		conversationID = &req.ConversationID
	}

	// Start the research task
	taskID, err := research.StartResearchTask(c.UserContext(), userID, req.Query, req.Model, conversationID)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to start research task")
	}

	// Return the task
	return c.Status(fiber.StatusCreated).JSON(fiber.Map{
		"task_id": taskID,
		"message": "Research task started successfully",
	})
}

// ListResearchTasks returns all research tasks for the user
func ListResearchTasks(c *fiber.Ctx) error {
	userID := c.UserContext().Value(auth.UserIDKey).(string)

	tasks, err := research.GetUserResearchTasks(c.UserContext(), userID)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to get research tasks")
	}

	// Return the tasks
	return c.JSON(tasks)
}

// GetResearchTask returns a specific research task
func GetResearchTask(c *fiber.Ctx) error {
	taskID, err := c.ParamsInt("taskID")
	if err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid task ID")
	}

	if taskID == 0 {
		return fiber.NewError(fiber.StatusBadRequest, "Task ID is required")
	}

	task, err := research.GetResearchTask(c.UserContext(), taskID)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to get research task")
	}

	// Return the task
	return c.JSON(task)
}

// CancelResearchTask cancels a specific research task
func CancelResearchTask(c *fiber.Ctx) error {
	taskID, err := c.ParamsInt("taskID")
	if err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid task ID")
	}

	if taskID == 0 {
		return fiber.NewError(fiber.StatusBadRequest, "Task ID is required")
	}

	if err := research.CancelResearchTask(c.UserContext(), taskID); err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to cancel research task")
	}

	// Return success
	return c.SendStatus(fiber.StatusOK)
}
