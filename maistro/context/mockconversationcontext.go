package context

import (
	"maistro/models"
	"maistro/util"
	"time"
)

var MockConversationContext = conversationContext{
	userID:         "test_user",
	conversationID: 1,
	title:          "Test Conversation",
	masterSummary:  nil,
	summaries: []models.Summary{
		{
			CreatedAt:      time.Date(2023, 10, 01, 12, 00, 00, 0, time.UTC),
			ConversationID: 1,
			Content:        `by bundling the inference for both LLMs and image generation into one container (or sidecar) that requests two GPUs, you ensure that model splitting can occur across both devices. This approach meets your requirement while keeping the gRPC interface separate from the inference logic, just as before—but now with full multi-GPU utilization available in a single process.`,
			SourceIds:      []int{1, 2, 3},
		},
	},
	retrievedMemories: []models.Memory{},
	messages: []models.Message{
		{
			ID:   util.IntPtr(1),
			Role: "user",
			Content: []models.MessageContent{{
				Type: models.MessageContentTypeText,
				Text: util.StrPtr("What are some strategies for using multiple GPUs?"),
			}},
		},
		{
			ID:   util.IntPtr(2),
			Role: "assistant",
			Content: []models.MessageContent{{
				Type: models.MessageContentTypeText,
				Text: util.StrPtr("You can split the model across GPUs or use a sidecar container to handle inference."),
			}},
		},
		{
			ID:   util.IntPtr(3),
			Role: "user",
			Content: []models.MessageContent{{
				Type: models.MessageContentTypeText,
				Text: util.StrPtr("Can a single model leverage VRAM across multiple GPUs?"),
			}},
		},
		{
			ID:   util.IntPtr(4),
			Role: "assistant",
			Content: []models.MessageContent{{
				Type: models.MessageContentTypeText,
				Text: util.StrPtr("Yes, by bundling inference into one container that requests two GPUs, you can utilize VRAM across both devices."),
			}},
		},
		{
			ID:   util.IntPtr(5),
			Role: "user",
			Content: []models.MessageContent{{
				Type: models.MessageContentTypeText,
				Text: util.StrPtr("How can I ensure that my LLM and image generation models can run in parallel on multiple GPUs?"),
			}},
		},
	},
	searchResults: []models.SearchResult{},
}
