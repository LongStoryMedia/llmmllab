package context

import (
	"context"
	"maistro/models"
	"maistro/storage"
	"maistro/test"
	"maistro/util"
	"slices"
	"testing"
)

func setup() error {
	ctx := context.Background()
	usrCfg, err := GetUserConfig(MockConversationContext.UserID)
	if err != nil {
		return util.HandleError(err)
	}
	_, err = storage.ConversationStoreInstance.CreateConversation(ctx, MockConversationContext.UserID, MockConversationContext.Title)
	if err != nil {
		return util.HandleError(err)
	}
	for _, msg := range MockConversationContext.Messages {
		_, err := storage.MessageStoreInstance.AddMessage(ctx, MockConversationContext.ConversationID, msg.Role, msg.Content, usrCfg)
		if err != nil {
			return util.HandleError(err)
		}
	}

	asstMsg := models.Message{
		Role:    "assistant",
		Content: "By using a single container that handles both LLM and image generation inference, you can ensure they run in parallel on multiple GPUs.",
		ID:      6,
	}

	MockConversationContext.Messages = append(MockConversationContext.Messages, asstMsg)

	_, err = storage.MessageStoreInstance.AddMessage(ctx, MockConversationContext.ConversationID, asstMsg.Role, asstMsg.Content, usrCfg)
	if err != nil {
		return util.HandleError(err)
	}
	_, err = storage.SummaryStoreInstance.CreateSummary(ctx, MockConversationContext.ConversationID, MockConversationContext.Summaries[0].Content, 1, MockConversationContext.Summaries[0].SourceIds)
	if err != nil {
		return util.HandleError(err)
	}

	return nil
}

func Test_SummarizeMessages(t *testing.T) {
	// Initialize test context
	test.LLmmLLab_Test(t, setup, func(t *testing.T) {
		sum, err := MockConversationContext.SummarizeMessages(context.Background())
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		for _, id := range []int{4, 5, 6} {
			if !slices.Contains(sum.SourceIds, id) {
				t.Errorf("expected source ID %d to be in summary, got %v", id, sum.SourceIds)
			}
		}

	})
}
