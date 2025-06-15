package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maistro/models"
	"maistro/socket"
	"maistro/util"
	"net/http"
	"time"
)

// StreamOllamaChatRequest sends a request to the Ollama API and handles streaming the response
func StreamOllamaChatRequest(ctx context.Context, modelProfile *models.ModelProfile, messages []models.ChatMessage, userID *string) (string, error) {
	// Send status update if user ID is available
	if userID != nil {
		socket.BroadcastProcessingStage(*userID, models.StatusUpdateStageInitializing, 10)
	}

	requestBody := models.ChatReq{
		Model:    modelProfile.ModelName,
		Messages: messages,
		Stream:   true,
		Options:  modelProfile.Parameters.ToMap(),
	}

	reqBody, err := json.Marshal(requestBody)
	if err != nil {
		if userID != nil {
			socket.BroadcastError(*userID, "Failed to prepare request")
		}
		return "", util.HandleError(fmt.Errorf("error marshaling request: %w", err))
	}

	util.LogInfo("Sending request to Ollama")
	if userID != nil {
		socket.BroadcastProcessingStage(*userID, models.StatusUpdateStageGenerating, 30)
	}

	handler, _, err := GetProxyHandler[*models.OllamaChatResp](ctx, reqBody, "/api/chat", http.MethodPost, true, time.Minute*10, nil)
	if err != nil {
		if userID != nil {
			socket.BroadcastError(*userID, "Failed to connect to Ollama")
		}
		return "", util.HandleError(fmt.Errorf("error streaming request: %w", err))
	}

	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	// Update status to processing
	if userID != nil {
		socket.BroadcastProcessingStage(*userID, models.StatusUpdateStageProcessing, 60)
	}

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			util.HandleError(err)
			if userID != nil {
				socket.BroadcastError(*userID, "Incomplete response from Ollama")
			}
		} else {
			if userID != nil {
				socket.BroadcastError(*userID, "Failed to process response")
			}
			return "", util.HandleError(fmt.Errorf("error handling response: %w", err))
		}
	}

	if err := wr.Flush(); err != nil {
		if userID != nil {
			socket.BroadcastError(*userID, "Failed to finalize response")
		}
		return "", util.HandleError(fmt.Errorf("error flushing response: %w", err))
	}

	// Check if the response is empty
	if res == "" {
		if userID != nil {
			socket.BroadcastError(*userID, "Empty response from Ollama")
		}
		return "", util.HandleError(fmt.Errorf("empty response from Ollama"))
	}

	// Notify of completion
	if userID != nil {
		socket.BroadcastCompletion(*userID, "Response generated successfully")
	}

	return res, nil
}
