package api

import (
	"bytes"
	"encoding/json"
	"maistro/models"
	"maistro/test"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
)

var problemRequestString = `How can I alter a table in postgres such that I create a new serial column and make it the primary key while also removing the existing primary key, which is a compound key made of 3 other existing columns.
to explain better, I need this schema:
CREATE TABLE IF NOT EXISTS memories(
  user_id text NOT NULL,
  source_id integer NOT NULL,
  source text NOT NULL,
  embedding vector(768) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  PRIMARY KEY (source, source_id, created_at)
);
updated to this:
CREATE TABLE IF NOT EXISTS memories(
  id serial,
  user_id text NOT NULL,
  source_id integer NOT NULL,
  source text NOT NULL,
  embedding vector(768) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
and I want the id column to be added (and incremented) on all existing rows.`

var conversationId = 1

var problemRequest = models.ChatRequest{
	Content:        problemRequestString,
	ConversationID: conversationId,
}

const TIMEOUT = 5 * time.Minute

var filename = "testdata/ollama_test_output.md"

func Test_OllamaHandler(t *testing.T) {
	test.Init()
	app := test.NewMockApp()
	os.Setenv("TEST_OUTPUT_FILE", filename)
	// Clear the output file before running the test
	if err := os.WriteFile(filename, []byte(""), 0644); err != nil {
		t.Fatalf("Failed to clear output file: %v", err)
	}

	app.Post("/api/chat", ChatHandler)

	reqBody, err := json.Marshal(problemRequest)
	if err != nil {
		os.Setenv("TEST_OUTPUT_FILE", "")
		t.Fatalf("Failed to marshal request: %v", err)
	}

	req := httptest.NewRequest("POST", "/api/chat", bytes.NewBuffer(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req, int(TIMEOUT.Milliseconds()))
	if err != nil {
		os.Setenv("TEST_OUTPUT_FILE", "")
		t.Fatalf("Failed to test request: %v", err)
	}
	if resp.StatusCode != fiber.StatusOK {
		os.Setenv("TEST_OUTPUT_FILE", "")
		t.Fatalf("Expected status OK, got %d", resp.StatusCode)
	}
	os.Setenv("TEST_OUTPUT_FILE", "")
}
