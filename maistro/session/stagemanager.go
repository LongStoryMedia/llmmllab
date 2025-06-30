package session

import (
	"errors"
	"fmt"
	"maistro/models"
	"maistro/util"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// StageStatus represents the current execution status of a stage
type StageStatus string

type SessionID string

type SIDKey string

const (
	StageStatusRunning   StageStatus = "running"
	StageStatusPaused    StageStatus = "paused"
	StageStatusCanceled  StageStatus = "canceled"
	StageStatusCompleted StageStatus = "completed"
	StageStatusFailed    StageStatus = "failed"
	SessionCtxKey        SIDKey      = "session_state"
)

// RollbackFunc is a function that can be called to rollback actions
type RollbackFunc func() error

// StageState represents the state of a stage in a session
type StageState struct {
	SessionID      SessionID
	StageType      models.SocketStageType
	Status         StageStatus
	Progress       int
	Message        string
	StartTime      time.Time
	UpdateTime     time.Time
	EndTime        *time.Time
	UserID         string
	ConversationID int
	mutex          sync.RWMutex
}

type SessionState struct {
	mutex          sync.RWMutex
	Stages         map[models.SocketStageType]*StageState // SessionID -> StageType -> State
	Status         StageStatus
	UserID         string
	ConversationID int
	RollbackFunc   RollbackFunc
	CurrentRequest *models.ChatReq // Current request being processed
}

// StageManagerRegistry is the global registry for stage states
type StageManagerRegistry struct {
	States map[SessionID]*SessionState // SessionID -> StageType -> State
	mutex  sync.RWMutex
}

var (
	// GlobalStageManager is the singleton instance of StageManagerRegistry
	GlobalStageManager = &StageManagerRegistry{
		States: make(map[SessionID]*SessionState),
	}
)

func (smr *StageManagerRegistry) GetSessionState(userID string, conversationID int) *SessionState {
	smr.mutex.RLock()
	defer smr.mutex.RUnlock()
	sessionID := GetSessionID(userID, conversationID)

	state, exists := smr.States[sessionID]
	if !exists {
		state = &SessionState{
			Stages:         make(map[models.SocketStageType]*StageState),
			mutex:          sync.RWMutex{},
			UserID:         userID,
			ConversationID: conversationID,
			Status:         StageStatusRunning,
		}
		smr.States[sessionID] = state
	}

	return state
}

// CleanupSession removes all states for a session
func (smr *StageManagerRegistry) CleanupSession(sessionID SessionID) {
	smr.mutex.Lock()
	defer smr.mutex.Unlock()

	delete(smr.States, sessionID)
}

// Pause pauses the execution of a stage
func (ss *SessionState) Pause() error {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()
	ss.Status = StageStatusPaused
	// Send pause notification
	return nil
}

// Resume resumes the execution of a stage
func (ss *SessionState) Resume() error {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()
	ss.Status = StageStatusRunning
	// Send resume notification
	return nil
}

// Cancel cancels the execution of a stage
func (ss *SessionState) Cancel() error {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()

	if ss.Status == StageStatusCompleted || ss.Status == StageStatusFailed || ss.Status == StageStatusCanceled {
		return nil // Already canceled
	}
	ss.Status = StageStatusCanceled

	// Send cancel notification
	return nil
}

func (ss *SessionState) AddRollbackFunc(rollback RollbackFunc) {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()
	ss.RollbackFunc = rollback
}

// IsCanceled checks if the stage has been canceled
func (ss *SessionState) IsCanceled() bool {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Status == StageStatusCanceled
}

// IsPaused checks if the stage is paused
func (ss *SessionState) IsPaused() bool {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Status == StageStatusPaused
}

// IsRunning checks if the stage is currently running
func (ss *SessionState) IsRunning() bool {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Status == StageStatusRunning
}

// IsCompleted checks if the stage has been completed
func (ss *SessionState) IsCompleted() bool {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Status == StageStatusCompleted
}

// IsFailed checks if the stage has failed
func (ss *SessionState) IsFailed() bool {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Status == StageStatusFailed
}

// Checkpoint blocks until the stage is resumed or canceled
func (ss *SessionState) Checkpoint() bool {
	if ss.IsCanceled() || ss.IsCompleted() || ss.IsFailed() {
		return false // Stage is done, no need to wait
	} else if ss.IsRunning() {
		return true // Stage is running, no need to wait
	} else {
		time.Sleep(100 * time.Millisecond) // Give a brief pause before checking again
		return ss.Checkpoint()             // Recursively check again
	}
}

// startStage initializes a new stage state
func (ss *SessionState) startStage(sessionID SessionID, stageType models.SocketStageType, userID string, conversationID int, initialMessage string, rollbackFunc RollbackFunc) *StageState {
	// Check if the stage is already running
	if state, exists := ss.Stages[stageType]; exists {
		return state
	}
	util.LogDebug("Starting new stage", logrus.Fields{
		"sessionID":      sessionID,
		"stageType":      stageType,
		"userID":         userID,
		"conversationID": conversationID,
		"initialMessage": initialMessage,
	})

	// Create new state
	now := time.Now()
	state := &StageState{
		SessionID:      sessionID,
		StageType:      stageType,
		Status:         StageStatusRunning,
		Progress:       0,
		Message:        initialMessage,
		StartTime:      now,
		UpdateTime:     now,
		EndTime:        nil,
		UserID:         userID,
		ConversationID: conversationID,
	}
	ss.Stages[stageType] = state

	// Send initial status message

	return state
}

// GetStage retrieves the state for a specific stage, or starts a new one if it doesn't exist
func (ss *SessionState) GetStage(stageType models.SocketStageType) *StageState {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	sessionID := GetSessionID(ss.UserID, ss.ConversationID)

	state, exists := ss.Stages[stageType]
	if !exists {
		state = ss.startStage(sessionID, stageType, ss.UserID, ss.ConversationID, "Stage started", ss.RollbackFunc)
	}

	return state
}

// UpdateProgress updates the progress of a stage
func (ss *StageState) UpdateProgress(progress int, message string) error {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()

	if ss.Status == StageStatusCanceled {
		return errors.New("cannot update progress on canceled stage")
	}

	if ss.Status == StageStatusCompleted || ss.Status == StageStatusFailed {
		return errors.New("cannot update progress on completed or failed stage")
	}

	// Update state
	ss.Progress = progress
	if message != "" {
		ss.Message = message
	}
	ss.UpdateTime = time.Now()

	return nil
}

// Complete marks a stage as completed
func (ss *StageState) Complete(message string, payload ...any) {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()

	if ss.Status == StageStatusCanceled || ss.Status == StageStatusFailed || ss.Status == StageStatusCompleted {
		util.LogWarning(fmt.Sprintf("cannot complete stage with status %s", ss.Status))
		return
	}

	ss.Status = StageStatusCompleted
	ss.Progress = 100
	now := time.Now()
	ss.UpdateTime = now
	ss.EndTime = &now
	if message != "" {
		ss.Message = message
	}

	util.LogInfo("Stage completed", logrus.Fields{
		"sessionID": ss.SessionID,
		"stageType": ss.StageType,
		"userID":    ss.UserID,
		"duration":  now.Sub(ss.StartTime).String(),
	})

	return
}

// Fail marks a stage as failed
func (ss *StageState) Fail(message string, err error) error {
	ss.mutex.Lock()
	defer ss.mutex.Unlock()

	if ss.Status == StageStatusCanceled || ss.Status == StageStatusFailed || ss.Status == StageStatusCompleted {
		return fmt.Errorf("cannot fail stage with status %s", ss.Status)
	}

	ss.Status = StageStatusFailed
	now := time.Now()
	ss.UpdateTime = now
	ss.EndTime = &now
	if message != "" {
		ss.Message = message
	}

	util.LogWarning("Stage failed", logrus.Fields{
		"sessionID": ss.SessionID,
		"stageType": ss.StageType,
		"userID":    ss.UserID,
	})

	return nil
}

// GetStatus returns the current status of the stage
func (ss *StageState) GetStatus() StageStatus {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Status
}

// GetProgress returns the current progress
func (ss *StageState) GetProgress() int {
	ss.mutex.RLock()
	defer ss.mutex.RUnlock()
	return ss.Progress
}

// GetSessionID creates a session ID from user ID and conversation ID
func GetSessionID(userID string, conversationID int) SessionID {
	return SessionID(fmt.Sprintf("%s-%d", userID, conversationID))
}
