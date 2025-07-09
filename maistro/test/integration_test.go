package test

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"os"
	"time"

	"maistro/config"
	pxcx "maistro/context"
	"maistro/models"
	"maistro/proxy"
	svc "maistro/services"
	"maistro/storage"
	"maistro/util"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
)

var responseQualifierJsonSchema map[string]any = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"kind": map[string]any{
			"type": "string",
			"enum": []string{"affirmative", "negative", "neutral", "unknown"},
		},
	},
}

type testSetup struct {
	uid               string
	messages          []models.ChatMessage
	newMessage        models.ChatMessage
	ctx               context.Context
	userConfig        models.UserConfig
	expectedQualifier string
}

var testSetups []testSetup = []testSetup{
	{
		uid: "test_user",
		messages: []models.ChatMessage{
			{Role: "user", Content: "What are some strategies for using multiple GPUs?"},
			{Role: "assistant", Content: "You can split the model across GPUs or use a sidecar container to handle inference."},
			{Role: "user", Content: "Can a single model leverage VRAM across multiple GPUs?"},
			{Role: "assistant", Content: "Yes, by bundling inference into one container that requests two GPUs, you can utilize VRAM across both devices."},
			{Role: "user", Content: "How can I ensure that my LLM and image generation models can run in parallel on multiple GPUs?"},
		},
		newMessage: models.ChatMessage{
			Role:    "user",
			Content: "can a single model leverage VRAM across multiple GPUs?",
		},
		ctx:               context.Background(),
		userConfig:        MockUserConfig,
		expectedQualifier: "affirmative",
	},
	{
		uid: "test_user",
		messages: []models.ChatMessage{
			{Role: "user", Content: "Help me remember some things"},
			{Role: "assistant", Content: "Okay, what would you like to remember?"},
			{Role: "user", Content: "I paid the kids their allowance for June."},
			{Role: "assistant", Content: "Got it, I will remember that you paid the kids their allowance for June."},
		},
		newMessage: models.ChatMessage{
			Role:    "user",
			Content: "did I pay the kids their allowance for June?",
		},
		ctx:               context.Background(),
		userConfig:        MockUserConfig,
		expectedQualifier: "affirmative",
	},
}

func Test_Integration(t *testing.T) {
	confFile := "testdata/.config.yaml"
	ctx := context.Background()

	pgc, err := postgres.Run(ctx,
		"timescale/timescaledb-ha:pg17",
		postgres.WithInitScripts("testdata/init_test_db.sh"),
		postgres.BasicWaitStrategies(),
	)
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := testcontainers.TerminateContainer(pgc); err != nil {
			log.Printf("failed to terminate container: %s", err)
		}
	}()

	config.GetConfig(&confFile) // Load the configuration from the specified file

	psqlconn, err := pgc.ConnectionString(ctx)
	if err != nil {
		t.Fatal("Failed to get connection string:", err)
	}

	InitMockStore()

	if err := storage.InitDB(psqlconn); err != nil {
		t.Fatal("Failed to initialize database:", err)
	}
	util.LogInfo("Connected to PostgreSQL database", logrus.Fields{
		"connection": psqlconn,
	})

	if err := storage.UserConfigStoreInstance.UpdateUserConfig(ctx, "test_user", &MockUserConfig); err != nil {
		t.Fatal("Failed to update user config:", err)
	}

	t.Run("Integration Tests", func(tt *testing.T) {
		for _, s := range testSetups {
			// Invalidate cached user config to ensure we get fresh data
			pxcx.InvalidateCachedUserConfig(s.uid)

			// Explicitly set user config in the mock store
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			// Make sure the test user config is set in our mock store
			err := storage.UserConfigStoreInstance.UpdateUserConfig(ctx, s.uid, &s.userConfig)
			if err != nil {
				tt.Fatalf("Failed to set user config: %v", err)
			}

			// Explicitly set in the context cache as well
			err = pxcx.SetUserConfig(&s.userConfig)
			if err != nil {
				tt.Fatalf("Failed to cache user config: %v", err)
			}

			// Log for debugging
			tt.Logf("Setting user config for %s: Memory enabled=%v, Memory limit=%d",
				s.uid, s.userConfig.Memory.Enabled, s.userConfig.Memory.Limit)

			cco, err := pxcx.GetOrCreateConversation(s.ctx, s.uid, nil)
			if err != nil {
				t.Fatalf("Failed to get cached conversation: %v", err)
			}

			cc := getMockCC(cco)

			for _, m := range s.messages {
				if m.Role == "user" {
					if _, _, err := cc.AddUserMessage(s.ctx, m.Content); err != nil {
						t.Fatal("Failed to add user message:", err)
					}
				} else {
					if _, err := cc.AddAssistantMessage(s.ctx, m.Content); err != nil {
						t.Fatal("Failed to add assistant message:", err)
					}
				}
			}

			cfg, err := pxcx.GetUserConfig(s.uid)
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			if cfg.Memory == nil {
				tt.Fatal("Expected memory config to be set, got nil")
			}

			if !cfg.Memory.Enabled {
				tt.Fatalf("Expected memory to be enabled, got false (Memory: %+v)", cfg.Memory)
			}

			tt.Logf("User config retrieved successfully: Memory enabled: %v, Memory limit: %d",
				cfg.Memory.Enabled, cfg.Memory.Limit)

			_, req, err := cc.PrepareOllamaRequest(s.ctx, models.ChatRequest{
				Content:        s.newMessage.Content,
				ConversationID: cc.GetConversationID(),
			})
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			p, err := storage.ModelProfileStoreInstance.GetModelProfile(s.ctx, cfg.ModelProfiles.PrimaryProfileID)
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			file, err := os.OpenFile("/dev/null", os.O_WRONLY, 0)
			if err != nil {
				tt.Fatalf("Failed to open /dev/null: %v", err)
			}
			defer file.Close()
			w := bufio.NewWriter(file)
			defer w.Flush()

			res, err := svc.GetInferenceService().RelayUserMessage(s.ctx, p, req.Messages, cfg.UserID, cc.GetConversationID(), w)
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			if _, err := cc.AddAssistantMessage(s.ctx, res); err != nil {
				tt.Fatalf("Error adding assistant message: %v", err)
			}

			fmtp, err := storage.ModelProfileStoreInstance.GetModelProfile(s.ctx, cfg.ModelProfiles.FormattingProfileID)
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			qr, err := proxy.StreamOllamaGenerateRequest(s.ctx, models.GenerateReq{
				Model:   fmtp.ModelName,
				Prompt:  "Qualify the following assistant response:\n" + res + "\nIf the response was essentially a 'yes', it was 'affirmative', if 'no' it was 'negative'. Otherwise, it was 'neutral'.",
				Format:  responseQualifierJsonSchema,
				Options: fmtp.Parameters.ToMap(),
				Think:   util.BoolPtr(false),
			})
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			var qrRes struct {
				Kind string `json:"kind"`
			}
			if err := json.Unmarshal([]byte(util.RemoveThinkTags(qr)), &qrRes); err != nil {
				tt.Fatalf("Failed to unmarshal QR response: %v", err)
			}

			if qrRes.Kind != s.expectedQualifier {
				tt.Errorf("Expected qualifier '%s', got '%s'", s.expectedQualifier, qrRes.Kind)
			}

			embeddings, err := svc.GetInferenceService().GetEmbedding(s.ctx, s.newMessage.Content, &config.DefaultEmbeddingProfile, s.uid, cc.GetConversationID())
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			results, err := storage.MemoryStoreInstance.SearchSimilarity(s.ctx, embeddings, cfg.Memory.SimilarityThreshold, cfg.Memory.Limit, nil, nil, nil, nil)
			if err != nil {
				tt.Fatalf("Expected no error, got %v", err)
			}

			if results == nil {
				tt.Fatal("Expected results, got nil")
			}

			if len(results) == 0 {
				tt.Fatal("Expected results, got empty slice")
			}

			for _, msg := range results {
				if len(msg.Fragments) != 2 && msg.Source == "message" {
					tt.Errorf("Expected 2 fragments, got %d for message ID %d", len(msg.Fragments), msg.SourceID)
				}
			}

			messages, err := storage.MessageStoreInstance.GetConversationHistory(s.ctx, cc.GetConversationID())
			if err != nil {
				tt.Fatalf("Failed to get messages: %v", err)
			}
			if len(messages) == 0 {
				tt.Fatal("Expected messages, got none")
			}
			for _, msg := range messages {
				if msg.ConversationID != cc.GetConversationID() {
					tt.Errorf("Expected conversation ID %d, got %d", cc.GetConversationID(), msg.ConversationID)
				}
			}
			tt.Logf("Retrieved %d messages for conversation ID %d", len(messages), cc.GetConversationID())
			if len(messages) < 5 {
				tt.Errorf("Expected at least 5 messages, got %d", len(messages))
			}
		}
	})
}
