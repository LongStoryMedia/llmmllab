package svc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maistro/config"
	"maistro/models"
	"maistro/mq"
	"maistro/util"
	"net/http"
	"reflect"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// contextKey is a custom type for context keys to avoid collisions
type ContextKey string
type ContextHeaders map[string]string

const (
	ReqHeadersKey ContextKey    = "reqHeaders"
	StreamTimeout time.Duration = 10 * time.Minute // Timeout for streaming responses
)

type InferenceService interface {
	RelayUserMessage(ctx context.Context, modelProfile *models.ModelProfile, messages []models.ChatMessage, userID string, conversationID int, writer *bufio.Writer) (string, error)
	GenerateImage(ctx context.Context, userID string, conversationID int, originalRequest models.ImageGenerateRequest)
	EditImage(ctx context.Context, userID string, conversationID int, originalRequest models.ImageGenerateRequest)
	GetEmbedding(ctx context.Context, textToEmbed string, mp *models.ModelProfile, userID string, conversationID int) ([][]float32, error)
}

type InferenceSvc struct {
	client   *http.Client
	s        *InferenceScheduler
	mqClient *mq.RabbitMQClient // RabbitMQ client for async inference requests
}

var (
	infSvc InferenceService = &InferenceSvc{
		client: &http.Client{
			Timeout: StreamTimeout, // Long timeout for streaming requests
			Transport: &http.Transport{
				IdleConnTimeout:       0,
				MaxIdleConns:          100,
				MaxIdleConnsPerHost:   100,
				DisableKeepAlives:     false,
				ResponseHeaderTimeout: 0,
				ExpectContinueTimeout: 0,
				TLSHandshakeTimeout:   0,
			},
		},
		s:        NewInferenceScheduler(),
		mqClient: nil, // Will be initialized in GetInferenceService
	}

	reqHeaders = ContextHeaders{
		"User-Agent":   "maistro/1.0",
		"Content-Type": "application/json",
		"Accept":       "application/json",
	}

	streamingReqHeaders = ContextHeaders{
		"Accept-Encoding": "identity",
		"X-Stream":        "true",
	}

	httpReqHeaders = ContextHeaders{
		"Accept-Encoding": "gzip, deflate", // Enable compression for non-streaming
		"X-Stream":        "false",
	}

	resHeaders = ContextHeaders{
		"Content-Type":  "application/json",
		"Cache-Control": "no-cache",
		"Connection":    "keep-alive",
	}

	streamingResHeaders = ContextHeaders{
		"Transfer-Encoding": "chunked",
		"X-Accel-Buffering": "no",
	}
)

// GetInferenceService creates/returns the InferenceService instance
func GetInferenceService() InferenceService {
	// Initialize the RabbitMQ client if it's not already initialized
	svc := infSvc.(*InferenceSvc)
	if svc.mqClient == nil {
		conf := config.GetConfig(nil)
		if conf.Rabbitmq.Enabled {
			svc.mqClient = mq.NewRabbitMQClient()
			if err := svc.mqClient.Initialize(); err != nil {
				util.HandleError(err, logrus.Fields{
					"message": "Failed to initialize RabbitMQ client",
				})
				// Continue without RabbitMQ - will fall back to direct API calls
			} else {
				logrus.Info("RabbitMQ client initialized successfully")
			}
		}
	}
	return infSvc
}

func (s *InferenceSvc) setRequestHeaders(req *http.Request, headers ...ContextHeaders) {
	if headers == nil {
		return
	}
	for _, h := range headers {
		if h == nil {
			continue
		}
		for key, value := range h {
			if value != "" {
				req.Header.Set(key, value)
			}
		}
	}
}

func (s *InferenceSvc) setResponseHeaders(resp *http.Response, headers ...ContextHeaders) {
	if headers == nil {
		return
	}
	for _, h := range headers {
		if h == nil {
			continue
		}
		for key, value := range h {
			if value != "" {
				resp.Header.Set(key, value)
			}
		}
	}
}

func (s *InferenceSvc) RelayUserMessage(ctx context.Context, modelProfile *models.ModelProfile, messages []models.ChatMessage, userID string, conversationID int, w *bufio.Writer) (string, error) {
	rc := NewResponseChan()
	ir := &InferenceRequest{
		Priority:       1,
		RequiredMemory: util.Gb2b(6), // 100 MB, adjust as needed
		EnqueueTime:    time.Now(),
		DispatchArgs:   []any{ctx, modelProfile, messages, userID, conversationID, w},
		Dispatch:       s.packageForDispatch(s.relayForUser),
		ResponseChan:   rc,
	}

	s.s.Enqueue(ir)
	res := <-rc
	if res.Err != nil {
		return "", res.Err
	}

	if e, ok := res.Result.(string); ok {
		return e, nil
	}

	return "", util.HandleError(fmt.Errorf("unexpected result type: %T, expected string", res.Result))
}

// GetEmbedding retrieves a vector embedding for the provided text using the specified model profile
func (s *InferenceSvc) GetEmbedding(ctx context.Context, textToEmbed string, mp *models.ModelProfile, userID string, conversationID int) ([][]float32, error) {
	rc := NewResponseChan()
	ir := &InferenceRequest{
		Priority:       1,
		RequiredMemory: util.Mb2b(100), // 100 MB, adjust as needed
		EnqueueTime:    time.Now(),
		DispatchArgs:   []any{ctx, textToEmbed, mp, userID, conversationID},
		Dispatch:       s.packageForDispatch(s.getEmbedding),
		ResponseChan:   rc,
	}

	s.s.Enqueue(ir)
	res := <-rc
	if res.Err != nil {
		return nil, res.Err
	}

	if e, ok := res.Result.([][]float32); ok {
		return e, nil
	}

	return nil, util.HandleError(fmt.Errorf("unexpected result type: %T, expected [][]float32", res.Result))
}

//

func (s *InferenceSvc) GenerateImage(ctx context.Context, userID string, conversationID int, originalRequest models.ImageGenerateRequest) {
	// Use RabbitMQ if available
	if s.mqClient != nil && s.mqClient.Initialized() {
		util.LogInfo("Using RabbitMQ for image generation request")

		// Create a response handler that processes the image generation result
		responseHandler := func(requestID mq.CorrelationID, result models.InferenceQueueMessage, err error) {
			if err != nil {
				util.HandleError(err)
				return
			}

			util.LogInfo("Received image generation result from RabbitMQ", logrus.Fields{
				"requestId":  requestID,
				"resultTask": result.Task,
				"resultType": result.Type,
			})

			pldAny := result.Payload
			if pldAny == nil {
				util.HandleError(fmt.Errorf("received nil payload for request ID %s", requestID))
				return
			}

			pldByts, err := json.Marshal(pldAny)
			if err != nil {
				util.HandleError(fmt.Errorf("failed to marshal payload: %w", err))
				return
			}

			pld := &models.ImageGenerateResponse{}
			if err := json.Unmarshal(pldByts, pld); err != nil {
				util.HandleError(fmt.Errorf("failed to unmarshal payload: %w", err))
				return
			}
			// Send the image generation response back to the client
			img, err := GetImgSvc().SaveImage(pld, conversationID, userID)
			if err != nil {
				util.HandleError(err)
				return
			}

			// Send the image metadata back to the client
			GetSocketService().SendCompletion(models.SocketStageTypeGeneratingImage, conversationID, userID, "Image generation completed", img)
			util.LogInfo("Image generation completed successfully", logrus.Fields{
				"conversationId": conversationID,
				"userId":         userID,
				"imageId":        img.ID,
			})
		}

		// Submit the request through RabbitMQ
		requestID, err := mq.SubmitImageGenerationRequest(s.mqClient, originalRequest, userID, conversationID, 10, responseHandler)
		if err != nil {
			util.HandleError(err)
			// Fall back to direct dispatch
		} else {
			util.LogInfo("Image generation request submitted to RabbitMQ", logrus.Fields{
				"requestId": requestID,
			})
			return // Successfully submitted to RabbitMQ, no need to continue
		}
	}
}

// EditImage processes an image editing request, either through RabbitMQ or directly
func (s *InferenceSvc) EditImage(ctx context.Context, userID string, conversationID int, originalRequest models.ImageGenerateRequest) {
	// Use RabbitMQ if available
	if s.mqClient != nil && s.mqClient.Initialized() {
		util.LogInfo("Using RabbitMQ for image editing request")
		// Create a response handler that processes the image editing result
		responseHandler := func(requestID mq.CorrelationID, result models.InferenceQueueMessage, err error) {
			if err != nil {
				util.HandleError(err)
				return
			}

			util.LogInfo("Received image editing result from RabbitMQ", logrus.Fields{
				"requestId":  requestID,
				"resultTask": result.Task,
				"resultType": result.Type,
			})

			pldAny := result.Payload
			if pldAny == nil {
				util.HandleError(fmt.Errorf("received nil payload for request ID %s", requestID))
				return
			}

			pldByts, err := json.Marshal(pldAny)
			if err != nil {
				util.HandleError(fmt.Errorf("failed to marshal payload: %w", err))
				return
			}

			pld := &models.ImageGenerateResponse{}
			if err := json.Unmarshal(pldByts, pld); err != nil {
				util.HandleError(fmt.Errorf("failed to unmarshal payload: %w", err))
				return
			}

			// Send the image editing response back to the client
			img, err := GetImgSvc().SaveImage(pld, conversationID, userID)
			if err != nil {
				util.HandleError(err)
				return
			}

			// Send the image metadata back to the client
			GetSocketService().SendCompletion(models.SocketStageTypeGeneratingImage, conversationID, userID, "Image editing completed", img)
			util.LogInfo("Image editing completed successfully", logrus.Fields{
				"conversationId": conversationID,
				"userId":         userID,
				"imageId":        img.ID,
			})
		}
		// Submit the request through RabbitMQ
		requestID, err := mq.SubmitImageEditRequest(s.mqClient, originalRequest, userID, conversationID, 10, responseHandler)
		if err != nil {
			util.HandleError(err)
			// Fall back to direct dispatch
		} else {
			util.LogInfo("Image editing request submitted to RabbitMQ", logrus.Fields{
				"requestId": requestID,
			})
			return // Successfully submitted to RabbitMQ, no need to continue
		}
	}
}

// packageForDispatch wraps any function as func(args ...any) (any, error)
func (s *InferenceSvc) packageForDispatch(fn any) DispatchFunc {
	return func(args ...any) InferenceResponse {
		fnVal := reflect.ValueOf(fn)
		if fnVal.Kind() != reflect.Func {
			return InferenceResponse{Result: nil, Err: fmt.Errorf("provided value is not a function")}
		}
		if len(args) != fnVal.Type().NumIn() {
			return InferenceResponse{Result: nil, Err: fmt.Errorf("argument count mismatch")}
		}
		in := make([]reflect.Value, len(args))
		for i, arg := range args {
			if arg == nil {
				in[i] = reflect.Zero(fnVal.Type().In(i))
			} else {
				in[i] = reflect.ValueOf(arg)
			}
		}
		out := fnVal.Call(in)
		var result any
		var err error
		if len(out) > 0 {
			result = out[0].Interface()
		}
		if len(out) > 1 && !out[1].IsNil() {
			err = out[1].Interface().(error)
		}
		return InferenceResponse{Result: result, Err: err}
	}
}

func (s *InferenceSvc) relayForUser(ctx context.Context, modelProfile *models.ModelProfile, messages []models.ChatMessage, userID string, conversationID int, w *bufio.Writer) (string, error) {
	requestBody := models.ChatReq{
		Model:    modelProfile.ModelName,
		Messages: messages,
		Stream:   true,
		Options:  modelProfile.Parameters.ToMap(),
	}

	sock := GetSocketService()

	reqBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", sock.SendError(models.SocketStageTypeInitializing, conversationID, userID, "Failed to prepare request")
	}

	util.LogInfo("Sending request to Ollama")

	// ss := session.GlobalStageManager.GetSessionState(userID, conversationID)

	conf := config.GetConfig(nil)
	url := conf.InferenceServices.Ollama.BaseURL + "/api/chat"

	// Create a response content builder
	var responseContent strings.Builder

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return "", sock.SendError(models.SocketStageTypeInitializing, conversationID, userID, err.Error())
	}

	s.setRequestHeaders(req, reqHeaders, streamingReqHeaders)

	// Make the request to Ollama
	resp, err := s.client.Do(req)
	if err != nil {
		return "", sock.SendError(models.SocketStageTypeInitializing, conversationID, userID, err.Error())
	}
	defer resp.Body.Close() // Ensure the body is closed to avoid resource leaks

	// Set response headers
	s.setResponseHeaders(resp, resHeaders, streamingResHeaders)

	// Handle error responses from Ollama
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return "", sock.SendError(models.SocketStageTypeInitializing, conversationID, userID, fmt.Sprintf("ollama error response: Status Code: %d, Body: %s", resp.StatusCode, string(body)))
	}

	proxyToClient := w != nil // If w is nil, we don't want to proxy to the client
	completed := false
	util.LogInfo("Streaming response from Ollama")

	scanner := bufio.NewScanner(resp.Body)

	// Set up a done channel to handle timeouts or context cancellation
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			util.LogInfo("Context done, stopping streaming")
			close(done)
		case <-time.After(StreamTimeout): // Fallback timeout
			util.LogInfo("Timeout reached, stopping streaming")
			close(done)
		}
	}()

loopScan:
	for scanner.Scan() {
		select {
		case <-done:
			util.LogInfo("Stopping streaming due to context done or timeout")
			break loopScan // Exit if context is done or timeout occurs
		default:
			// Continue processing
		}

		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Extract content from the JSON chunk
		respObj := &models.OllamaChatResp{}
		if err := json.Unmarshal(line, respObj); err != nil {
			util.HandleError(fmt.Errorf("error unmarshaling response: %w", err))
			continue // Skip this line on error but continue processing
		}

		if respObj.Message.Content != "" {
			responseContent.WriteString(respObj.Message.Content)
		}

		// Check if this is done before attempting to write to the client
		if respObj.Done {
			if !proxyToClient {
				util.LogInfo("Streaming completed, but not proxying to client")
				completed = true
				break
			}
			// For done message, always try to send to client if proxyToClient is true
		}

		if proxyToClient {
			// Write the line to the client
			if _, writeErr := w.Write(line); writeErr != nil {
				util.HandleError(writeErr)
				proxyToClient = false // Stop proxying on error
			} else {
				// Only attempt to flush if the write was successful
				if flushErr := w.Flush(); flushErr != nil {
					util.HandleError(flushErr)
					proxyToClient = false // Stop proxying on error
				}
			}

			// Handle completion after successful write of final chunk
			if respObj.Done {
				util.LogInfo("Streaming completed successfully")
				completed = true
				break
			}
		}
	}
	if err := scanner.Err(); err != nil {
		util.HandleError(err)
	}
	if proxyToClient {
		if flushErr := w.Flush(); flushErr != nil {
			util.HandleError(flushErr)
		}
	}

	if completed {
		if !proxyToClient {
			util.LogWarning("Streaming completed but not proxying to client, returning accumulated content")
		}
		return responseContent.String(), nil
	} else {
		return responseContent.String(), sock.SendError(models.SocketStageTypeInitializing, conversationID, userID, "streaming incomplete, client disconnected or context canceled")
	}
}

// getEmbedding retrieves a vector embedding for the provided text
func (s *InferenceSvc) getEmbedding(ctx context.Context, textToEmbed string, mp *models.ModelProfile, userID string, conversationID int) ([][]float32, error) {
	// Sanitize the input text before embedding
	cleanText := util.SanitizeText(textToEmbed)
	const MAX_EMBEDDING_LENGTH = 2500 // Maximum length for text embeddings, adjust as needed

	var inputChunks []string
	if len([]rune(cleanText)) > MAX_EMBEDDING_LENGTH {
		inputChunks = splitTextIntoChunks(cleanText, MAX_EMBEDDING_LENGTH)
	} else {
		inputChunks = []string{cleanText}
	}

	requestPayload := models.EmbeddingReq{
		Model:    mp.ModelName,
		Input:    inputChunks,
		Truncate: util.BoolPtr(true), // Truncate long inputs to fit model constraints
		Options:  mp.Parameters.ToMap(),
	}
	payloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to marshal embedding request: %w", err))
	}

	sock := GetSocketService()

	conf := config.GetConfig(nil)
	url := conf.InferenceServices.Ollama.BaseURL + "/api/embed"

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, sock.SendError(models.SocketStageTypeProcessing, conversationID, userID, err.Error())
	}

	s.setRequestHeaders(req, reqHeaders, httpReqHeaders)

	// Make the request to Ollama
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, sock.SendError(models.SocketStageTypeProcessing, conversationID, userID, err.Error())
	}
	defer resp.Body.Close() // Ensure the body is closed to avoid resource leaks
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, sock.SendError(models.SocketStageTypeProcessing, conversationID, userID, fmt.Sprintf("failed to read response body: %v", err))
	}

	// Set response headers
	s.setResponseHeaders(resp, resHeaders)

	// Handle error responses from Ollama
	if resp.StatusCode >= 400 {
		return nil, sock.SendError(models.SocketStageTypeInitializing, conversationID, userID, fmt.Sprintf("ollama error response: Status Code: %d, Body: %s", resp.StatusCode, string(body)))
	}

	var embeddingResponse models.OllamaEmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResponse); err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to decode embedding response: %w", err))
	}

	if len(embeddingResponse.Embeddings) == 0 {
		util.LogWarning("Received empty embedding response from Ollama")
		return [][]float32{}, nil // Return empty slice instead of error
	}

	return embeddingResponse.Embeddings, nil
}

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

// GetMQClient returns the RabbitMQ client
func (s *InferenceSvc) GetMQClient() *mq.RabbitMQClient {
	return s.mqClient
}
