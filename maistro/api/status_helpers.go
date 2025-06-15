package api

import "maistro/models"

// BroadcastProcessingStage sends a processing stage status update to a specific user
func BroadcastProcessingStage(userID string, stage models.StatusUpdateStage, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, stage, "", progress, false)
}

// BroadcastError sends an error status update to a specific user
func BroadcastError(userID string, message string) {
	SendStatusUpdate(userID, models.StatusUpdateTypeError, models.StatusUpdateStageError, message, 0, false)
}

// BroadcastCompletion sends a completion status update to a specific user
func BroadcastCompletion(userID string, message string) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageCompleted, message, 100, true)
}

// BroadcastMemoryStage sends a memory retrieval status update
func BroadcastMemoryStage(userID string, message string, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageRetrievingMemories, message, progress, false)
}

// BroadcastSearchStage sends a web search status update
func BroadcastSearchStage(userID string, message string, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageSearchingWeb, message, progress, false)
}

// BroadcastSummarizingStage sends a summarization status update
func BroadcastSummarizingStage(userID string, message string, progress int) {
	SendStatusUpdate(userID, models.StatusUpdateTypeInfo, models.StatusUpdateStageSummarizing, message, progress, false)
}
