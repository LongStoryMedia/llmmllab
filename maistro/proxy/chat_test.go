package proxy

import (
	"context"
	"maistro/config"
	"maistro/models"
	"maistro/util"
	"testing"
)

func Test_StreamOllamaChatRequest(t *testing.T) {
	confFile := "testdata/.config.yaml"
	config.GetConfig(&confFile)

	messages := []models.Message{
		{Role: "user", Content: []models.MessageContent{{Type: models.MessageContentTypeText, Text: util.StrPtr("Why is the sky blue?")}}},
	}
	content, err := StreamOllamaChatRequest(context.Background(), &config.DefaultPrimaryProfile, messages, "test-user", 1)
	if err != nil {
		t.Fatalf("Failed to get chat response: %v", err)
	}
	if len(content) == 0 {
		t.Fatalf("Received empty chat for text: %s", content)
	}
	t.Logf("chat response for '%v': %v", messages[0].Content, content)
}
