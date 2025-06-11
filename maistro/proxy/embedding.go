package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maistro/models"
	"maistro/util"
	"net/http"
	"time"
)

const MAX_EMBEDDING_LENGTH = 2500 // Maximum length for text embeddings, adjust as needed

// splitTextIntoChunks splits a string into slices of maxChunkSize length
func splitTextIntoChunks(text string, maxChunkSize int) []string {
	var chunks []string
	runes := []rune(text)
	for i := 0; i < len(runes); i += maxChunkSize {
		end := i + maxChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks
}

// GetOllamaEmbedding retrieves a vector embedding for the provided text from Ollama
func GetOllamaEmbedding(ctx context.Context, textToEmbed string, modelName string) ([][]float32, error) {
	// Sanitize the input text before embedding
	cleanText := util.SanitizeText(textToEmbed)

	var inputChunks []string
	if len([]rune(cleanText)) > MAX_EMBEDDING_LENGTH {
		inputChunks = splitTextIntoChunks(cleanText, MAX_EMBEDDING_LENGTH)
	} else {
		inputChunks = []string{cleanText}
	}

	requestPayload := models.EmbeddingReq{
		Model:    modelName,
		Input:    inputChunks,
		Truncate: util.BoolPtr(true), // Truncate long inputs to fit model constraints
	}
	payloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to marshal embedding request: %w", err))
	}

	handler, _, err := GetProxyHandler[*models.OllamaEmbeddingResponse](ctx, payloadBytes, "/api/embed", http.MethodPost, false, time.Second*15)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to get proxy handler for embedding request: %w", err))
	}

	util.LogDebug("Successfully retrieved proxy handler for embedding request")

	w := &bytes.Buffer{}
	wr := bufio.NewWriter(w)

	respStr, err := handler(wr)
	if err != nil {
		if IsIncompleteError(err) {
			util.HandleError(err)
		} else {
			return nil, util.HandleError(fmt.Errorf("error handling embedding response: %w", err))
		}
	}

	var embeddingResponse models.OllamaEmbeddingResponse
	if err := json.Unmarshal([]byte(respStr), &embeddingResponse); err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to decode embedding response: %w", err))
	}

	if len(embeddingResponse.Embeddings) == 0 {
		util.LogWarning("Received empty embedding response from Ollama")
		return [][]float32{}, nil // Return empty slice instead of error
	}

	return embeddingResponse.Embeddings, nil
}
