package models

// NewResponse returns a pointer to an OllamaResponse based on the type argument
func NewResponse[T OllamaResponse]() T {
	var response T
	switch any(response).(type) {
	case *OllamaChatResp:
		return any(&OllamaChatResp{}).(T) // Return a new OllamaChatResp
	case *OllamaGenerateResponse:
		return any(&OllamaGenerateResponse{}).(T) // Return a new OllamaGenerateResponse
	case *OllamaEmbeddingResponse:
		return any(&OllamaEmbeddingResponse{}).(T) // Return a new OllamaEmbeddingResponse
	default:
		panic("unsupported response type")
	}
}

type OllamaResponse interface {
	IsDone() bool
	GetChunkContent() string
	UnmarshalJSON(data []byte) error
}

// OllamaChatResp represents a response from the Ollama API
type OllamaChatResp struct {
	Model              string      `json:"model"`
	CreatedAt          string      `json:"created_at"`
	Message            ChatMessage `json:"message"`
	Done               bool        `json:"done"`
	DoneReason         string      `json:"done_reason"`
	TotalDuration      float64     `json:"total_duration"`
	LoadDuration       float64     `json:"load_duration"`
	PromptEvalCount    int         `json:"prompt_eval_count"`
	PromptEvalDuration float64     `json:"prompt_eval_duration"`
	EvalCount          int         `json:"eval_count"`
	EvalDuration       float64     `json:"eval_duration"`
}

// OllamaGenerateResponse represents the response from Ollama's generate API
type OllamaGenerateResponse struct {
	Model              string  `json:"model"`
	CreatedAt          string  `json:"created_at"`
	Response           string  `json:"response"`
	Done               bool    `json:"done"`
	DoneReason         string  `json:"done_reason"`
	Context            []int   `json:"context,omitempty"` // context IDs for the response
	PromptEvalCount    int     `json:"prompt_eval_count"`
	PromptEvalDuration float64 `json:"prompt_eval_duration"`
	EvalCount          int     `json:"eval_count"`
	EvalDuration       float64 `json:"eval_duration"`
	TotalDuration      float64 `json:"total_duration"`
	LoadDuration       float64 `json:"load_duration"`
}

// OllamaEmbeddingResponse represents the response from Ollama's embeddings API
type OllamaEmbeddingResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int         `json:"total_duration"`
	LoadDuration    int         `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}
