package test

import (
	"context"
	pxcx "maistro/context"
	"maistro/models"
)

type mockCC struct {
	realCC pxcx.ConversationContext
	calls  []string
}

func getMockCC(realCC pxcx.ConversationContext) pxcx.ConversationContext {
	return &mockCC{realCC: realCC}
}

func (c *mockCC) GetConversationID() int {
	c.calls = append(c.calls, "GetConversationID")
	return c.realCC.GetConversationID()
}

func (c *mockCC) GetUserID() string {
	c.calls = append(c.calls, "GetUserID")
	return c.realCC.GetUserID()
}

func (c *mockCC) GetTitle() string {
	c.calls = append(c.calls, "GetTitle")
	return c.realCC.GetTitle()
}

func (c *mockCC) GetMessages() []models.Message {
	c.calls = append(c.calls, "GetMessages")
	return c.realCC.GetMessages()
}

func (c *mockCC) GetSummaries() []models.Summary {
	c.calls = append(c.calls, "GetSummaries")
	return c.realCC.GetSummaries()
}

func (c *mockCC) GetImages() []models.ImageMetadata {
	c.calls = append(c.calls, "GetImages")
	return c.realCC.GetImages()
}

func (c *mockCC) GetMasterSummary() *models.Summary {
	c.calls = append(c.calls, "GetMasterSummary")
	return c.realCC.GetMasterSummary()
}

func (c *mockCC) GetRetrievedMemories() []models.Memory {
	c.calls = append(c.calls, "GetRetrievedMemories")
	return c.realCC.GetRetrievedMemories()
}

func (c *mockCC) GetSearchResults() []models.SearchResult {
	c.calls = append(c.calls, "GetSearchResults")
	return c.realCC.GetSearchResults()
}

func (c *mockCC) GetIntent() *pxcx.Intent {
	c.calls = append(c.calls, "GetIntent")
	return c.realCC.GetIntent()
}

func (c *mockCC) ClearNotes() {
	c.calls = append(c.calls, "ClearNotes")
	c.realCC.ClearNotes()
}

func (c *mockCC) AddUserMessage(ctx context.Context, content string) ([][]float32, int, error) {
	c.calls = append(c.calls, "AddUserMessage")
	return c.realCC.AddUserMessage(ctx, content)
}

func (c *mockCC) AddAssistantMessage(ctx context.Context, content string) ([][]float32, error) {
	c.calls = append(c.calls, "AddAssistantMessage")
	return c.realCC.AddAssistantMessage(ctx, content)
}

func (c *mockCC) SummarizeMessages(ctx context.Context) (*models.Summary, error) {
	c.calls = append(c.calls, "SummarizeMessages")
	return c.realCC.SummarizeMessages(ctx)
}

func (c *mockCC) PrepareOllamaRequest(ctx context.Context, req models.ChatRequest) ([]byte, *models.ChatReq, error) {
	c.calls = append(c.calls, "PrepareOllamaRequest")
	return c.realCC.PrepareOllamaRequest(ctx, req)
}
