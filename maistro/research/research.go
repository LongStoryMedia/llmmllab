package research

import (
	"context"
	"fmt"
	"time"

	"maistro/models"
	"maistro/storage"
)

// CreateResearchRequest is the request to create a new research task
type CreateResearchRequest struct {
	Query          string `json:"query"`
	Model          string `json:"model,omitempty"`
	ConversationID int    `json:"conversation_id,omitempty"`
}

// StartResearchTask starts a new research task
func StartResearchTask(ctx context.Context, userID, query, model string, conversationID *int) (*int, error) {
	// Save the task to the database
	taskID, err := storage.ResearchTaskStoreInstance.SaveResearchTask(ctx, userID, query, conversationID)
	if err != nil {
		return nil, fmt.Errorf("failed to save research task: %w", err)
	}

	// Start the research process asynchronously using the deep research orchestrator
	go PerformDeepResearch(context.Background(), taskID, userID, query, conversationID)

	return &taskID, nil
}

// GetResearchTask gets a research task by ID
func GetResearchTask(ctx context.Context, taskID int) (*models.ResearchTask, error) {
	return storage.ResearchTaskStoreInstance.GetTaskByID(ctx, taskID)
}

// GetUserResearchTasks gets all research tasks for a user
func GetUserResearchTasks(ctx context.Context, userID string) ([]*models.ResearchTask, error) {
	tasks, err := storage.ResearchTaskStoreInstance.ListTasksByUserID(ctx, userID, 100, 0) // Default limit 100, offset 0
	if err != nil {
		return nil, err
	}

	// Convert []storage.ResearchTask to []*models.ResearchTask
	result := make([]*models.ResearchTask, len(tasks))
	for i, task := range tasks {
		result[i] = &models.ResearchTask{
			ID:             task.ID,
			UserID:         task.UserID,
			Query:          task.Query,
			Model:          task.Model,
			Status:         task.Status,
			ConversationID: task.ConversationID,
			CreatedAt:      task.CreatedAt,
			UpdatedAt:      task.UpdatedAt,
			CompletedAt:    task.CompletedAt,
			ErrorMessage:   task.ErrorMessage,
		}
	}

	return result, nil
}

// CancelResearchTask cancels a research task
func CancelResearchTask(ctx context.Context, taskID int) error {
	// Get the task first to verify ownership
	task, err := storage.ResearchTaskStoreInstance.GetTaskByID(ctx, taskID)
	if err != nil {
		return fmt.Errorf("failed to get research task: %w", err)
	}

	// Only allow cancellation if the task is not already completed or failed
	if task.Status == models.ResearchTaskStatusCOMPLETED || task.Status == models.ResearchTaskStatusFAILED {
		return fmt.Errorf("cannot cancel a task that is already %s", task.Status)
	}

	// Update the status to canceled
	task.Status = models.ResearchTaskStatusCANCELED
	task.UpdatedAt = time.Now()
	errMessage := "Task was canceled by the user"

	_, err = storage.ResearchTaskStoreInstance.UpdateTask(ctx, taskID, string(models.ResearchTaskStatusCANCELED), &errMessage)
	return err
}
