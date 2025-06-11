package storage

import (
	"context"
	"fmt"
	"maistro/models"
	"time"

	"github.com/jackc/pgx/v5"
)

// Research task statuses
const (
	TaskStatusPending     = "PENDING"
	TaskStatusPlanning    = "PLANNING"
	TaskStatusGathering   = "GATHERING"
	TaskStatusAnalyzing   = "ANALYZING"
	TaskStatusSummarizing = "SUMMARIZING"
	TaskStatusCompleted   = "COMPLETED"
	TaskStatusFailed      = "FAILED"
	TaskStatusCanceled    = "CANCELED"
)

// Subtask statuses
const (
	SubtaskStatusPending   = "PENDING"
	SubtaskStatusRunning   = "RUNNING"
	SubtaskStatusGathering = "GATHERING"
	SubtaskStatusAnalyzing = "ANALYZING"
	SubtaskStatusCompleted = "COMPLETED"
	SubtaskStatusFailed    = "FAILED"
)

// researchTaskStore implements the ResearchTaskStore interface
type researchTaskStore struct{}

// SaveResearchTask inserts a new research task
func (rts *researchTaskStore) SaveResearchTask(ctx context.Context, userID, query string, conversationID *int) (int, error) {
	var taskID int
	err := Pool.QueryRow(ctx, GetQuery("research.save_research_task"),
		userID, conversationID, query).Scan(&taskID)
	if err != nil {
		return -1, err
	}
	return taskID, nil
}

// UpdateTaskStatus updates the status of a research task
func (rts *researchTaskStore) UpdateTaskStatus(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error) {
	var updatedAt time.Time
	err := Pool.QueryRow(ctx, GetQuery("research.update_task_status"),
		taskID, status, errorMsg).Scan(&updatedAt)
	return updatedAt, err
}

// UpdateTask updates a task with completed_at when applicable
func (rts *researchTaskStore) UpdateTask(ctx context.Context, taskID int, status string, errorMsg *string) (time.Time, error) {
	var updatedAt time.Time
	err := Pool.QueryRow(ctx, GetQuery("research.update_task"),
		taskID, status, errorMsg).Scan(&updatedAt)
	return updatedAt, err
}

// StoreResearchPlan stores the plan for a research task
func (rts *researchTaskStore) StoreResearchPlan(ctx context.Context, taskID int, plan *models.ResearchPlan) (time.Time, error) {
	var updatedAt time.Time
	err := Pool.QueryRow(ctx, GetQuery("research.store_plan"),
		taskID, plan).Scan(&updatedAt)
	return updatedAt, err
}

// StoreFinalResult stores the final research result
func (rts *researchTaskStore) StoreFinalResult(ctx context.Context, taskID int, result *models.ResearchQuestionResult) (time.Time, error) {
	var updatedAt time.Time
	err := Pool.QueryRow(ctx, GetQuery("research.store_final_result"),
		taskID, result).Scan(&updatedAt)
	return updatedAt, err
}

// SaveSubtask saves a research subtask
func (rts *researchTaskStore) SaveSubtask(ctx context.Context, subtask *models.ResearchSubtask) (int, error) {
	tx, err := Pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer func() {
		if err != nil {
			tx.Rollback(ctx)
		}
	}()

	var id int
	err = tx.QueryRow(ctx, GetQuery("research.save_subtask"),
		subtask.TaskID, subtask.QuestionID, subtask.Status,
		subtask.CreatedAt, subtask.UpdatedAt).Scan(&id)

	if err != nil {
		return 0, err
	}

	err = tx.Commit(ctx)
	if err != nil {
		return 0, err
	}

	return id, nil
}

// UpdateSubtaskStatus updates the status of a research subtask
func (rts *researchTaskStore) UpdateSubtaskStatus(ctx context.Context, taskID, questionID int,
	status string, errorMsg *string) (int, time.Time, error) {

	var id int
	var updatedAt time.Time

	err := Pool.QueryRow(ctx, GetQuery("research.update_subtask_status"),
		taskID, status, errorMsg, questionID).Scan(&id, &updatedAt)

	return id, updatedAt, err
}

// StoreGatheredInfo stores gathered information for a subtask
func (rts *researchTaskStore) StoreGatheredInfo(ctx context.Context, taskID, questionID int,
	gatheredInfo []string, sources []string) (time.Time, error) {

	var updatedAt time.Time
	err := Pool.QueryRow(ctx, GetQuery("research.store_gathered_info"),
		taskID, questionID, gatheredInfo, sources).Scan(&updatedAt)

	return updatedAt, err
}

// StoreSynthesizedAnswer stores the synthesized answer for a subtask
func (rts *researchTaskStore) StoreSynthesizedAnswer(ctx context.Context, taskID, questionID int,
	answer string) (time.Time, error) {

	var updatedAt time.Time
	err := Pool.QueryRow(ctx, GetQuery("research.store_synthesized_answer"),
		taskID, questionID, answer).Scan(&updatedAt)

	return updatedAt, err
}

// GetTaskByID retrieves a research task by its ID
func (rts *researchTaskStore) GetTaskByID(ctx context.Context, taskID int) (*models.ResearchTask, error) {
	var task models.ResearchTask

	err := Pool.QueryRow(ctx, GetQuery("research.get_task_by_id"), taskID).Scan(
		&task.ID, &task.UserID, &task.Query, &task.Model, &task.ConversationID,
		&task.Status, &task.ErrorMessage, &task.Plan, &task.Results,
		&task.CreatedAt, &task.UpdatedAt, &task.CompletedAt)

	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("research task not found: %v", taskID)
		}
		return nil, err
	}

	return &task, nil
}

// ListTasksByUserID lists all research tasks for a user
func (rts *researchTaskStore) ListTasksByUserID(ctx context.Context, userID string, limit, offset int) ([]models.ResearchTask, error) {
	if limit <= 0 {
		limit = 10
	}

	rows, err := Pool.Query(ctx, GetQuery("research.list_tasks_by_user"),
		userID, limit, offset)

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tasks []models.ResearchTask
	for rows.Next() {
		var task models.ResearchTask
		err := rows.Scan(
			&task.ID, &task.UserID, &task.Query, &task.Model, &task.ConversationID,
			&task.Status, &task.ErrorMessage, &task.CreatedAt, &task.UpdatedAt,
			&task.CompletedAt)

		if err != nil {
			return nil, err
		}
		tasks = append(tasks, task)
	}

	if err = rows.Err(); err != nil {
		return nil, err
	}

	return tasks, nil
}

// GetSubtasksForTask retrieves all subtasks for a research task
func (rts *researchTaskStore) GetSubtasksForTask(ctx context.Context, taskID string) ([]models.ResearchSubtask, error) {
	rows, err := Pool.Query(ctx, GetQuery("research.get_subtasks_for_task"), taskID)

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var subtasks []models.ResearchSubtask
	for rows.Next() {
		var subtask models.ResearchSubtask
		err := rows.Scan(
			&subtask.ID, &subtask.TaskID, &subtask.QuestionID, &subtask.Status,
			&subtask.GatheredInfo, &subtask.InformationSources, &subtask.SynthesizedAnswer,
			&subtask.ErrorMessage, &subtask.CreatedAt, &subtask.UpdatedAt)

		if err != nil {
			return nil, err
		}
		subtasks = append(subtasks, subtask)
	}

	if err = rows.Err(); err != nil {
		return nil, err
	}

	return subtasks, nil
}
