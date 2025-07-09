package context

// import (
// 	"encoding/json"
// 	"maistro/config"
// 	"maistro/models"
// 	"maistro/test"
// 	"reflect"
// 	"testing"
// )

// func Test_ToJSON(t *testing.T) {
// 	test.Init()
// 	msgs := []models.ChatMessage{}
// 	for _, m := range MockConversationContext.messages {
// 		msgs = append(msgs, models.ChatMessage{
// 			Role:    m.Role,
// 			Content: m.Content,
// 		})
// 	}
// 	req := models.ChatReq{
// 		Messages: msgs,
// 		Model:    config.DefaultPrimaryProfile.ModelName,
// 		Stream:   true,
// 		Options:  config.DefaultPrimaryProfile.Parameters.ToMap(),
// 	}

// 	// Convert to JSON
// 	jsonData, _, err := MockConversationContext.ChainMessages(&req)
// 	if err != nil {
// 		t.Fatalf("Failed to convert to JSON: %v", err)
// 	}

// 	rereq := models.ChatReq{}
// 	if err := json.Unmarshal(jsonData, &rereq); err != nil {
// 		t.Fatalf("Failed to unmarshal JSON: %v", err)
// 	}

// 	if len(rereq.Messages) != len(req.Messages) {
// 		t.Fatalf("Expected %d messages, got %d", len(req.Messages), len(rereq.Messages))
// 	}
// 	for i, msg := range rereq.Messages {
// 		if msg.Role != req.Messages[i].Role || msg.Content != req.Messages[i].Content {
// 			t.Fatalf("Message %d mismatch: expected role '%s' and content '%s', got role '%s' and content '%s'",
// 				i, req.Messages[i].Role, req.Messages[i].Content, msg.Role, msg.Content)
// 		}
// 	}
// 	// if rereq.Model == nil || *rereq.Model != config.DefaultPrimaryProfile.ModelName {
// 	// 	t.Fatalf("Expected model name '%s', got '%s'", config.DefaultPrimaryProfile.ModelName, *rereq.Model)
// 	// }
// 	if !rereq.Stream {
// 		t.Fatalf("Expected stream to be true, got false")
// 	}
// 	if len(rereq.Options) != len(config.DefaultPrimaryProfile.Parameters.ToMap()) {
// 		t.Fatalf("Expected %d options, got %d", len(config.DefaultPrimaryProfile.Parameters.ToMap()), len(rereq.Options))
// 	}

// 	// num_ctx
// 	// repeat_last_n
// 	// repeat_penalty
// 	// temperature
// 	// seed
// 	// stop
// 	// num_predict
// 	// top_k
// 	// top_p
// 	// min_p
// 	// Compare each key-value pair instead of using DeepEqual
// 	paramMap := config.DefaultPrimaryProfile.Parameters.ToMap()
// 	if paramMap["num_ctx"].(int) != int(rereq.Options["num_ctx"].(float64)) {
// 		t.Fatalf("Expected num_ctx %v, got %v", paramMap["num_ctx"], rereq.Options["num_ctx"])
// 	}
// 	if paramMap["repeat_last_n"].(int) != int(rereq.Options["repeat_last_n"].(float64)) {
// 		t.Fatalf("Expected repeat_last_n %v, got %v", paramMap["repeat_last_n"], rereq.Options["repeat_last_n"])
// 	}
// 	if paramMap["repeat_penalty"].(float64) != rereq.Options["repeat_penalty"].(float64) {
// 		t.Fatalf("Expected repeat_penalty %v, got %v", paramMap["repeat_penalty"], rereq.Options["repeat_penalty"])
// 	}
// 	if paramMap["seed"].(int) != int(rereq.Options["seed"].(float64)) {
// 		t.Fatalf("Expected seed %v, got %v", paramMap["seed"], rereq.Options["seed"])
// 	}
// 	if paramMap["num_predict"].(int) != int(rereq.Options["num_predict"].(float64)) {
// 		t.Fatalf("Expected num_predict %v, got %v", paramMap["num_predict"], rereq.Options["num_predict"])
// 	}
// 	if paramMap["top_k"].(int) != int(rereq.Options["top_k"].(float64)) {
// 		t.Fatalf("Expected top_k %v, got %v", paramMap["top_k"], rereq.Options["top_k"])
// 	}
// 	if paramMap["min_p"].(float64) != rereq.Options["min_p"].(float64) {
// 		t.Fatalf("Expected min_p %v, got %v", paramMap["min_p"], rereq.Options["min_p"])
// 	}
// 	if paramMap["temperature"].(float64) != rereq.Options["temperature"].(float64) {
// 		t.Fatalf("Expected temperature %v, got %v", paramMap["temperature"], rereq.Options["temperature"])
// 	}
// 	if paramMap["min_p"].(float64) != rereq.Options["min_p"].(float64) {
// 		t.Fatalf("Expected min_p %v, got %v", paramMap["min_p"], rereq.Options["min_p"])
// 	}

// 	// Convert []any to []string
// 	stopAny := rereq.Options["stop"].([]any)
// 	stopStr := make([]string, len(stopAny))
// 	for i, v := range stopAny {
// 		stopStr[i] = v.(string)
// 	}

// 	if !reflect.DeepEqual(paramMap["stop"].([]string), stopStr) {
// 		t.Fatalf("Expected stop %v, got %v", paramMap["stop"], rereq.Options["stop"])
// 	}

// 	if len(req.Messages) != 6 {
// 		t.Fatalf("Expected 6 messages, got %d", len(req.Messages))
// 	}
// 	if req.Messages[0].Role != "system" {
// 		t.Fatalf("Expected first message role to be 'system', got '%s'", req.Messages[0].Role)
// 	}
// }
