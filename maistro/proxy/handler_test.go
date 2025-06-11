package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"maistro/config"
	"maistro/models"
	"os"
	"testing"
	"time"
)

func Test_ChatRequest(t *testing.T) {
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)
	// Create a sample chat request
	req := models.ChatReq{
		Model: config.DefaultPrimaryProfile.ModelName,
		Messages: []models.ChatMessage{
			{Role: "user", Content: "Why is the sky blue?"},
		},
	}

	// Marshal the request to JSON
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal chat request: %v", err)
	}

	handler, status, err := GetProxyHandler[*models.OllamaChatResp](context.Background(), data, "/api/chat", "POST", true, time.Minute)
	if err != nil {
		if IsIncompleteError(err) {
			t.Fatalf("Chat request failed with incomplete error: %v", err)
		}
		t.Fatalf("Failed to get proxy handler: %v", err)
	}
	if status != 200 {
		t.Fatalf("Expected status 200, got %d", status)
	}

	// Create a buffer to capture the response
	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if res == "" {
			t.Fatalf("Chat request failed with empty response: %v", err)
		} else {
			t.Fatalf("connection closed, but response completed: %v", err)
		}
	}

	t.Log("Chat response:", res)
}

func Test_EmbeddingRequest(t *testing.T) {
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)
	// Create a sample embedding request
	req := models.EmbeddingReq{
		Model: config.DefaultEmbeddingProfile.ModelName,
		Input: []string{"Can I get and example of splitting a model accross multiple gpus in code?"},
	}

	// Marshal the request to JSON
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal embedding request: %v", err)
	}

	handler, status, err := GetProxyHandler[*models.OllamaEmbeddingResponse](context.Background(), data, "/api/embed", "POST", false, time.Second*15)
	if err != nil {
		t.Fatalf("Failed to get proxy handler: %v", err)
	}
	if status != 200 {
		t.Fatalf("Expected status 200, got %d", status)
	}

	// Create a buffer to capture the response
	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			t.Fatalf("Chat request failed with incomplete error: %v", err)
		}
		t.Fatalf("Embedding request failed: %v", err)
	}

	t.Log("Embedding response:", res)

	var embeddingResponse models.OllamaEmbeddingResponse
	if err := json.Unmarshal([]byte(res), &embeddingResponse); err != nil {
		t.Fatalf("Failed to decode embedding response: %v", err)
	}
	if len(embeddingResponse.Embeddings) == 0 {
		t.Fatalf("Received empty embedding response")
	}
}

func Test_GenerateRequest(t *testing.T) {
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)
	// Create a sample generate request
	req := models.GenerateReq{
		Model:  config.DefaultPrimaryProfile.ModelName,
		Prompt: "Tell me a joke about llamas.",
	}

	// Marshal the request to JSON
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal generate request: %v", err)
	}

	handler, status, err := GetProxyHandler[*models.OllamaGenerateResponse](context.Background(), data, "/api/generate", "POST", true, time.Second*15)
	if err != nil {
		t.Fatalf("Failed to get proxy handler: %v", err)
	}
	if status != 200 {
		t.Fatalf("Expected status 200, got %d", status)
	}

	// Create a buffer to capture the response
	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			t.Fatalf("Chat request failed with incomplete error: %v", err)
		}
		t.Fatalf("Generate request failed: %v", err)
	}

	if res == "" {
		t.Fatalf("Generate request returned empty response")
	}

	t.Log("Generate response:", res)
}

const filename = "testdata/ollama_test_output.md"
const TIMEOUT = 5 * time.Minute

func Test_ChatRequestWithLongContext(t *testing.T) {
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)
	os.Setenv("TEST_OUTPUT_FILE", filename)
	// Clear the output file before running the test
	if err := os.WriteFile(filename, []byte(""), 0644); err != nil {
		t.Fatalf("Failed to clear output file: %v", err)
	}
	defer os.Setenv("TEST_OUTPUT_FILE", "")

	msgs, err := os.ReadFile("testdata/messages.json")
	if err != nil {
		t.Fatalf("Failed to read messages file: %v", err)
	}
	var messages []models.ChatMessage
	if err := json.Unmarshal(msgs, &messages); err != nil {
		t.Fatalf("Failed to unmarshal messages: %v", err)
	}

	req := models.ChatReq{
		Model:    config.DefaultPrimaryProfile.ModelName,
		Messages: messages,
		Stream:   true,
		Options:  config.DefaultPrimaryProfile.Parameters.ToMap(),
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal chat request: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), TIMEOUT)
	defer cancel()
	handler, status, err := GetProxyHandler[*models.OllamaChatResp](ctx, data, "/api/chat", "POST", true, TIMEOUT)
	if err != nil {
		t.Fatalf("Failed to get proxy handler: %v", err)
	}
	if status != 200 {
		t.Fatalf("Expected status 200, got %d", status)
	}
	// Create a buffer to capture the response
	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	res, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			t.Fatalf("Chat request failed with incomplete error: %v", err)
		}
		t.Fatalf("Chat request failed: %v", err)
	}

	if res == "" {
		t.Fatalf("Chat request returned empty response")
	}

	t.Log("Chat response:", res)
}
