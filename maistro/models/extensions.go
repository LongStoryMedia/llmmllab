package models

import (
	"encoding/json"
	"fmt"
)

func (p *ModelParameters) ToMap() map[string]any {
	return map[string]any{
		"num_ctx":        p.NumCtx,
		"repeat_last_n":  p.RepeatLastN,
		"repeat_penalty": p.RepeatPenalty,
		"temperature":    p.Temperature,
		"seed":           p.Seed,
		"stop":           p.Stop,
		"num_predict":    p.NumPredict,
		"top_k":          p.TopK,
		"top_p":          p.TopP,
		"min_p":          p.MinP,
	}
}

func (r *OllamaChatResp) UnmarshalJSON(data []byte) error {
	type Alias OllamaChatResp
	aux := (*Alias)(r)
	return json.Unmarshal(data, aux)
}

func (r *OllamaChatResp) GetChunkContent() string {
	if r.Message.Content != "" {
		return r.Message.Content
	}
	return ""
}

func (r *OllamaChatResp) IsDone() bool {
	return r.Done
}

func (r *OllamaGenerateResponse) UnmarshalJSON(data []byte) error {
	type Alias OllamaGenerateResponse
	aux := (*Alias)(r)
	return json.Unmarshal(data, aux)
}
func (r *OllamaGenerateResponse) GetChunkContent() string {
	if r.Response != "" {
		return r.Response
	}
	return ""
}
func (r *OllamaGenerateResponse) IsDone() bool {
	return r.Done
}

func (r *OllamaEmbeddingResponse) UnmarshalJSON(data []byte) error {
	type Alias OllamaEmbeddingResponse
	aux := (*Alias)(r)
	return json.Unmarshal(data, aux)
}
func (r *OllamaEmbeddingResponse) GetChunkContent() string {
	if len(r.Embeddings) > 0 && len(r.Embeddings[0]) > 0 {
		// Convert the first embedding to a string representation
		return fmt.Sprintf("%v", r.Embeddings[0])
	}
	return ""
}
func (r *OllamaEmbeddingResponse) IsDone() bool {
	return len(r.Embeddings) > 0
}
