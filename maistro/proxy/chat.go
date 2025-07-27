package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maistro/models"
	"maistro/session"
	"maistro/util"
	"net/http"
	"time"
)

// Deprecated: Use svc.InferenceService instead
func StreamOllamaChatRequest(ctx context.Context, modelProfile *models.ModelProfile, messages []models.Message, userID string, conversationID int) (string, error) {
	requestBody := models.ChatReq{
		Model:     modelProfile.ModelName,
		Messages:  messages,
		Stream:    true,
		Options:   &modelProfile.Parameters,
		KeepAlive: util.IntPtr(0),
	}

	reqBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", util.HandleError(fmt.Errorf("error marshaling request: %w", err))
	}

	util.LogInfo("Sending request to Ollama")

	ss := session.GlobalStageManager.GetSessionState(userID, conversationID)
	handler, _, err := GetProxyHandler[*models.OllamaChatResp](ctx, ss, reqBody, "/api/chat", http.MethodPost, true, time.Minute*10, nil)
	if err != nil {
		return "", util.HandleError(fmt.Errorf("error streaming request: %w", err))
	}

	stage := ss.GetStage(models.SocketStageTypeProcessing)

	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			stage.Fail("Incomplete response from llm", err)
		} else {
			return "", stage.Fail("error handling response", err)
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
