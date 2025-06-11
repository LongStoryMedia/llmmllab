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

// StreamOllamaRequest sends a request to the Ollama API and handles streaming the response
func StreamOllamaGenerateRequest(ctx context.Context, model string, requestBody models.GenerateReq) (string, error) {
	reqBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %w", err)
	}

	util.LogInfo("Sending request to Ollama")

	handler, _, err := GetProxyHandler[*models.OllamaGenerateResponse](ctx, reqBody, "/api/generate", http.MethodPost, true, time.Minute)
	if err != nil {
		return "", fmt.Errorf("error streaming request: %w", err)
	}

	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			util.HandleError(err)
		} else {
			return "", fmt.Errorf("error handling response: %w", err)
		}
	}

	if err := wr.Flush(); err != nil {
		return "", fmt.Errorf("error flushing response: %w", err)
	}

	// Check if the response is empty
	if res == "" {
		return "", fmt.Errorf("empty response from Ollama")
	}

	return res, nil
}
