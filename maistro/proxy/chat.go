package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maistro/models"
	"maistro/util"
	"net/http"
	"time"
)

// StreamOllamaChatRequest sends a request to the Ollama API and handles streaming the response
func StreamOllamaChatRequest(ctx context.Context, modelProfile *models.ModelProfile, messages []models.ChatMessage) (string, error) {
	requestBody := models.ChatReq{
		Model:    modelProfile.ModelName,
		Messages: messages,
		Stream:   true,
		Options:  modelProfile.Parameters.ToMap(),
	}

	reqBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", util.HandleError(fmt.Errorf("error marshaling request: %w", err))
	}

	util.LogInfo("Sending request to Ollama")

	handler, _, err := GetProxyHandler[*models.OllamaChatResp](ctx, reqBody, "/api/chat", http.MethodPost, true, time.Minute*10)
	if err != nil {
		return "", util.HandleError(fmt.Errorf("error streaming request: %w", err))
	}

	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			util.HandleError(err)
		} else {
			return "", util.HandleError(fmt.Errorf("error handling response: %w", err))
		}
	}

	if err := wr.Flush(); err != nil {
		return "", util.HandleError(fmt.Errorf("error flushing response: %w", err))
	}

	// Check if the response is empty
	if res == "" {
		return "", util.HandleError(fmt.Errorf("empty response from Ollama"))
	}

	return res, nil
}
