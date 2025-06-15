package proxy

import (
	"context"
	"maistro/config"
	"maistro/models"
	"testing"
)

func Test_StreamOllamaChatRequest(t *testing.T) {
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)

	messages := []models.ChatMessage{
		{Role: "user", Content: "Why is the sky blue?"},
	}
	content, err := StreamOllamaChatRequest(context.Background(), &config.DefaultPrimaryProfile, messages, nil)
	if err != nil {
		t.Fatalf("Failed to get chat response: %v", err)
	}
	if len(content) == 0 {
		t.Fatalf("Received empty chat for text: %s", content)
	}
	t.Logf("chat response for '%s': %v", messages[0].Content, content)
}
