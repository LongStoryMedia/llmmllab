package proxy

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"maistro/config"
	"maistro/models"
	"maistro/util"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// contextKey is a custom type for context keys to avoid collisions
type ContextKey string
type ContextHeaders map[string]string
type IncompleteResponseError struct {
	Message string
}

func (e *IncompleteResponseError) Error() string {
	return fmt.Sprintf("incomplete response: %s", e.Message)
}

// IsIncompleteError checks if the given error is an IncompleteResponseError
func IsIncompleteError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*IncompleteResponseError)
	return ok
}

func NewIncompleteResponseError(message string) *IncompleteResponseError {
	return &IncompleteResponseError{Message: message}
}

// Context keys
const (
	ReqHeadersKey ContextKey    = "reqHeaders"
	StreamTimeout time.Duration = 10 * time.Minute // Timeout for streaming responses
)

func GetHeadersFromContext(ctx context.Context) ContextHeaders {
	headers, ok := ctx.Value(ReqHeadersKey).(ContextHeaders)
	if !ok {
		return nil
	}
	return headers
}

// GetProxyHandler handles streaming responses from Ollama and collects the full assistant response
// It properly streams to the client while also returning the accumulated content
func GetProxyHandler[T models.OllamaResponse](ctx context.Context, reqBody []byte, path, method string, stream bool, timeout time.Duration) (func(w *bufio.Writer) (string, error), int, error) {
	util.LogInfo("Proxying request to Ollama", logrus.Fields{"url": path})
	conf := config.GetConfig(nil)
	url := conf.InferenceServices.Ollama.BaseURL + path

	// Create a response content builder
	var responseContent strings.Builder

	// Create the request
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fiber.StatusInternalServerError, err
	}

	// Create HTTP client
	client := createHTTPClient()

	req.Header.Set("User-Agent", "maistro/1.0")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if stream {
		req.Header.Set("Accept-Encoding", "identity") // Disable compression for streaming
		req.Header.Set("X-Stream", "true")            // Custom header to indicate streaming
	} else {
		req.Header.Set("Accept-Encoding", "gzip, deflate") // Enable compression for non-streaming
		req.Header.Set("X-Stream", "false")                // Custom header to indicate non-streaming
	}

	// Make the request to Ollama
	resp, err := client.Do(req)
	if err != nil {
		return nil, fiber.StatusBadGateway, err
	}

	resp.Header.Set("Content-Type", "application/json")
	resp.Header.Set("Cache-Control", "no-cache")
	resp.Header.Set("Connection", "keep-alive")
	if stream {
		resp.Header.Set("Transfer-Encoding", "chunked")
		resp.Header.Set("X-Accel-Buffering", "no")
	}

	// Handle error responses from Ollama
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		errMsg := fmt.Errorf("ollama error response: Status Code: %d, Body: %s", resp.StatusCode, string(body))
		util.HandleError(errMsg, logrus.Fields{"statusCode": resp.StatusCode, "body": string(body)})
		return nil, resp.StatusCode, fiber.NewError(resp.StatusCode, string(body))
	}

	// Use a buffered channel to wait for streaming to complete
	return func(w *bufio.Writer) (string, error) {
		if stream {
			return streamHandler[T](w, resp, &responseContent, timeout)
		}
		return resHandler(w, resp)
	}, resp.StatusCode, nil
}

func streamHandler[T models.OllamaResponse](w *bufio.Writer, resp *http.Response, responseContent *strings.Builder, timeout time.Duration) (string, error) {
	defer resp.Body.Close() // Ensure connection is closed when done
	proxyToClient := true
	completed := false
	util.LogInfo("Streaming response from Ollama")

	scanner := bufio.NewScanner(resp.Body)

	// Use a channel to detect context cancellation or client disconnection
	ctx, cancel := context.WithTimeout(context.Background(), timeout) // Or pass the context from the handler
	defer cancel()
	// Set up a done channel to handle timeouts or context cancellation
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			util.LogInfo("Context done, stopping streaming")
			close(done)
		case <-time.After(timeout): // Fallback timeout
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
		respObj := models.NewResponse[T]()
		respObj.UnmarshalJSON(line)
		content := respObj.GetChunkContent()
		if content != "" {
			responseContent.WriteString(content)
		}

		if !proxyToClient {
			// If proxyToClient is false, we don't want to write to the client
			// but we still want to accumulate the response content
			if respObj.IsDone() {
				util.LogInfo("Streaming completed, but not proxying to client")
				completed = true
				break
			}
			continue
		}

		if _, writeErr := w.Write(line); writeErr != nil {
			util.HandleError(writeErr)
			proxyToClient = false
		}
		if flushErr := w.Flush(); flushErr != nil {
			util.HandleError(flushErr)
			proxyToClient = false
		}

		// Check if this is the last chunk (done=true)
		if respObj.IsDone() {
			util.LogInfo("Streaming completed successfully")
			completed = true
			break
		}
	}
	if err := scanner.Err(); err != nil {
		util.HandleError(err)
	}
	if flushErr := w.Flush(); flushErr != nil {
		util.HandleError(flushErr)
	}

	if completed {
		if !proxyToClient {
			return responseContent.String(), util.HandleError(fmt.Errorf("connection closed by client"))
		}
		return responseContent.String(), nil
	} else {
		return responseContent.String(), NewIncompleteResponseError("streaming incomplete, client disconnected or context canceled")
	}
}

func resHandler(w *bufio.Writer, resp *http.Response) (string, error) {
	defer resp.Body.Close() // Ensure connection is closed when done
	util.LogInfo("Reading response from Ollama")

	res, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", util.HandleError(err)
	}
	if len(res) == 0 {
		return "", util.HandleError(fmt.Errorf("empty response from Ollama"))
	}
	// Write to response
	if _, writeErr := w.Write(res); writeErr != nil {
		return "", util.HandleError(writeErr)
	}
	// Flush frequently to avoid client timeouts
	if flushErr := w.Flush(); flushErr != nil {
		return "", util.HandleError(flushErr)

	}
	return string(res), nil
}

// createHTTPClient returns a configured HTTP client optimized for streaming responses
func createHTTPClient() *http.Client {
	return &http.Client{
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
	}
}
