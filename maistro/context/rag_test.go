package context

import (
	"context"
	"maistro/config"
	"maistro/proxy"
	"maistro/test"
	"testing"
)

func Test_RetrieveAndInjectMemories(t *testing.T) {
	test.Init()
	// Initialize the context and user ID
	ctx := context.Background()
	embedding, err := proxy.GetOllamaEmbedding(ctx, "Can I get and example of splitting a model accross multiple gpus in code?", config.DefaultEmbeddingProfile.ModelName)
	if err != nil {
		t.Fatalf("Failed to get embedding: %v", err)
	}

	t.Log("Embedding:", embedding)

	if err := MockConversationContext.RetrieveAndInjectMemories(ctx, embedding, nil, nil); err != nil {
		t.Fatalf("Failed to retrieve and inject memories: %v", err)
	}

	// Check if memories were injected
	if len(MockConversationContext.RetrievedMemories) == 0 {
		t.Fatal("Expected memories to be injected, but none were found")
	}
	for _, memory := range MockConversationContext.RetrievedMemories {
		if len(memory.Fragments) == 0 {
			t.Error("Injected memory has empty content")
		}
		for _, fragment := range memory.Fragments {
			if fragment.Content == "" {
				t.Error("Injected memory fragment has empty content")
			}
		}
	}
}
