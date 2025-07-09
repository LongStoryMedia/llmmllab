package proto

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maistro/models"
	"maistro/util"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// apiKeyAuth is a simple implementation of the credentials.PerRPCCredentials interface
// for API key based authentication
type apiKeyAuth struct {
	apiKey string
}

// newAPIKeyAuth creates a new API key authentication credential
func newAPIKeyAuth(apiKey string) *apiKeyAuth {
	return &apiKeyAuth{
		apiKey: apiKey,
	}
}

// GetRequestMetadata implements the credentials.PerRPCCredentials interface
func (a *apiKeyAuth) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return map[string]string{
		"x-api-key": a.apiKey,
	}, nil
}

// RequireTransportSecurity implements the credentials.PerRPCCredentials interface
func (a *apiKeyAuth) RequireTransportSecurity() bool {
	// Ideally should be true in production for secure transmission
	return false
}

// GRPCClient implements the InferenceService interface using gRPC
type GRPCClient struct {
	conn     *grpc.ClientConn
	client   InferenceServiceClient
	address  string
	secure   bool
	apiKey   string
	connLock sync.Mutex // Protects conn and client
}

// NewGRPCClient creates a new gRPC client for the inference service
func NewGRPCClient(address string, secure bool, apiKey string) (*GRPCClient, error) {
	c := &GRPCClient{
		address: address,
		secure:  secure,
		apiKey:  apiKey,
	}

	// Initialize the connection
	if err := c.initConnection(); err != nil {
		return nil, err
	}

	return c, nil
}

// initConnection initializes the gRPC connection
func (c *GRPCClient) initConnection() error {
	c.connLock.Lock()
	defer c.connLock.Unlock()

	if c.conn != nil {
		// Connection already exists
		return nil
	}

	// Set up connection options
	opts := []grpc.DialOption{
		grpc.WithBlock(), // Block until the connection is established
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second, // Send pings every 10 seconds if there is no activity
			Timeout:             5 * time.Second,  // Wait 5 seconds for ping ack before assuming the connection is dead
			PermitWithoutStream: true,             // Allow pings even without active streams
		}),
	}

	// Set up transport credentials
	if c.secure {
		// Use TLS for secure connection
		creds := credentials.NewTLS(&tls.Config{
			InsecureSkipVerify: false, // Set to true to skip certificate verification (not recommended for production)
		})
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		// Use insecure for development/testing
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	// Add API key as a per-RPC credential if provided
	if c.apiKey != "" {
		opts = append(opts, grpc.WithPerRPCCredentials(newAPIKeyAuth(c.apiKey)))
	}

	// Set connection timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Establish the connection
	var err error
	c.conn, err = grpc.DialContext(ctx, c.address, opts...)
	if err != nil {
		return fmt.Errorf("failed to connect to inference service: %w", err)
	}

	// Create the client
	c.client = NewInferenceServiceClient(c.conn)

	util.LogInfo("Connected to inference service", logrus.Fields{
		"address": c.address,
		"secure":  c.secure,
	})

	return nil
}

// Close closes the connection
func (c *GRPCClient) Close() error {
	c.connLock.Lock()
	defer c.connLock.Unlock()

	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		c.client = nil
		return err
	}

	return nil
}

// reconnect attempts to reestablish the connection if it's broken
func (c *GRPCClient) reconnect() error {
	c.Close()                 // Close the existing connection
	return c.initConnection() // Create a new connection
}

// ensureConnection ensures that the connection is established
func (c *GRPCClient) ensureConnection() error {
	c.connLock.Lock()
	defer c.connLock.Unlock()

	if c.conn == nil || c.client == nil {
		return c.initConnection()
	}

	// Test the connection with a lightweight call if possible
	// We could implement a health check here

	return nil
}

// ChatStream streams chat responses
func (c *GRPCClient) ChatStream(ctx context.Context, req models.ChatReq, writer *bufio.Writer) (*models.ChatResponse, error) {
	// Check connection and reconnect if needed
	if err := c.ensureConnection(); err != nil {
		return nil, err
	}

	if req.ConversationID == nil {
		return nil, util.HandleError(errors.New("conversation ID is required for chat requests"))
	}

	cid := int32(*req.ConversationID)

	msgs := make([]*ChatMessage, 0, len(req.Messages))
	for _, msg := range req.Messages {
		tc := make([]*ChatMessage_ToolCallsItem, 0, len(msg.ToolCalls))
		for _, toolCall := range msg.ToolCalls {
			// tc to bytes
			uf, err := json.Marshal(toolCall)
			if err != nil {
				return nil, util.HandleError(err)
			}

			tc = append(tc, &ChatMessage_ToolCallsItem{
				unknownFields: uf,
			})
		}

		msgId := -1
		if msg.ID != nil {
			msgId = *msg.ID
		}

		msgs = append(msgs, &ChatMessage{
			Role:           msg.Role,
			Content:        msg.Content,
			Images:         msg.Images,
			ToolCalls:      tc,
			Id:             int32(msgId),
			ConversationId: int32(cid),
		})
	}

	fm, err := json.Marshal(req.Format)
	if err != nil {
		return nil, util.HandleError(err)
	}

	ka := 0
	if req.KeepAlive != nil {
		ka = *req.KeepAlive
	}

	t := make([]*ChatReq_ToolsItem, 0, len(req.Tools))
	for _, tool := range req.Tools {
		// tool to bytes
		uf, err := json.Marshal(tool)
		if err != nil {
			return nil, util.HandleError(err)
		}

		t = append(t, &ChatReq_ToolsItem{
			unknownFields: uf,
		})
	}

	opts, err := json.Marshal(req.Options)
	if err != nil {
		return nil, util.HandleError(err)
	}

	thk := true
	if req.Think != nil {
		thk = *req.Think
	}

	// Convert request to gRPC format
	grpcReq := &ChatReq{
		ConversationId: cid,
		Model:          req.Model,
		Messages:       msgs,
		Stream:         req.Stream,
		Format:         &ChatReq_Format{unknownFields: fm},
		KeepAlive:      int32(ka),
		Tools:          t,
		Options:        &ChatReq_Options{unknownFields: opts},
		Think:          thk,
	}

	// Make the streaming request
	stream, err := c.client.ChatStream(ctx, grpcReq)
	if err != nil {
		return nil, fmt.Errorf("failed to start chat stream: %w", err)
	}

	// Stream the responses
	res := models.ChatResponse{
		Done: false,
		Message: &models.ChatMessage{
			Role:      "assistant",
			Content:   "",
			Images:    nil,
			ToolCalls: nil,
			Thinking:  nil,
		},
		CreatedAt:          time.Now().Format(time.RFC3339),
		Model:              req.Model,
		Context:            nil,
		DoneReason:         nil,
		TotalDuration:      nil,
		LoadDuration:       nil,
		PromptEvalCount:    nil,
		PromptEvalDuration: nil,
		EvalCount:          nil,
		EvalDuration:       nil,
	}
	proxyToClient := writer != nil // If w is nil, we don't want to proxy to the client
	completed := false
	var responseContent strings.Builder
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			// End of stream
			break
		}
		if err != nil {
			return &res, fmt.Errorf("error receiving from chat stream: %w", err)
		}

		// Accumulate the full response
		if resp.Message.Content != "" {
			responseContent.WriteString(resp.Message.Content)
		}
		// Check if this is done before attempting to write to the client
		if resp.Done {
			if !proxyToClient {
				util.LogInfo("Streaming completed, but not proxying to client")
				completed = true
				break
			}
			// For done message, always try to send to client if proxyToClient is true
		}

		if proxyToClient {
			// Write the line to the client
			if _, writeErr := writer.Write([]byte(resp.Message.Content)); writeErr != nil {
				util.HandleError(writeErr)
				proxyToClient = false // Stop proxying on error
			} else {
				// Only attempt to flush if the write was successful
				if flushErr := writer.Flush(); flushErr != nil {
					util.HandleError(flushErr)
					proxyToClient = false // Stop proxying on error
				}
			}

			// Handle completion after successful write of final chunk
			if resp.Done {
				util.LogInfo("Streaming completed successfully")
				completed = true
				res.Message.Content = responseContent.String()
				res.Done = true
				res.CreatedAt = time.Now().Format(time.RFC3339)
				res.Model = req.Model
				res.DoneReason = util.StrPtr(resp.DoneReason)
				res.TotalDuration = util.Float32Ptr(float32(resp.TotalDuration))
				res.LoadDuration = util.Float32Ptr(float32(resp.LoadDuration))
				res.PromptEvalCount = util.Float32Ptr(float32(resp.PromptEvalCount))
				res.PromptEvalDuration = util.Float32Ptr(float32(resp.PromptEvalDuration))
				res.EvalCount = util.Float32Ptr(float32(resp.EvalCount))
				res.EvalDuration = util.Float32Ptr(float32(resp.EvalDuration))
				break
			}
		}
	}
	if proxyToClient {
		if flushErr := writer.Flush(); flushErr != nil {
			util.HandleError(flushErr)
		}
	}

	if completed {
		if !proxyToClient {
			util.LogWarning("Streaming completed but not proxying to client, returning accumulated content")
		}
		return &res, nil
	}

	return &res, nil
}

// GenerateStream streams generated text
func (c *GRPCClient) GenerateStream(ctx context.Context, req models.GenerateReq, writer *bufio.Writer) (*models.GenerateResponse, error) {
	// Check connection and reconnect if needed
	if err := c.ensureConnection(); err != nil {
		return nil, err
	}

	opts, err := json.Marshal(req.Options)
	if err != nil {
		return nil, util.HandleError(err)
	}
	fm, err := json.Marshal(req.Format)
	if err != nil {
		return nil, util.HandleError(err)
	}

	ka := 0
	if req.KeepAlive != nil {
		ka = *req.KeepAlive
	}

	suf := ""
	if req.Suffix != nil {
		suf = *req.Suffix
	}

	str := true
	if req.Stream != nil {
		str = *req.Stream
	}

	sys := ""
	if req.System != nil {
		sys = *req.System
	}

	tpl := ""
	if req.Template != nil {
		tpl = *req.Template
	}

	raw := false
	if req.Raw != nil {
		raw = *req.Raw
	}

	thk := true
	if req.Think != nil {
		thk = *req.Think
	}

	// Convert request to gRPC format
	grpcReq := &GenerateReq{
		Model:     req.Model,
		Prompt:    req.Prompt,
		Suffix:    suf,
		Images:    req.Images,
		KeepAlive: int32(ka),
		Format:    &GenerateReq_Format{unknownFields: fm},
		Options:   &GenerateReq_Options{unknownFields: opts},
		Stream:    str,
		System:    sys,
		Template:  tpl,
		Raw:       raw,
		Think:     thk,
		// Context: "",
	}

	// Make the streaming request
	stream, err := c.client.GenerateStream(ctx, grpcReq)
	if err != nil {
		return nil, fmt.Errorf("failed to start generate stream: %w", err)
	}

	// Stream the responses
	res := models.GenerateResponse{
		Model:              req.Model,
		CreatedAt:          nil,
		Response:           "",
		Done:               false,
		DoneReason:         nil,
		Context:            nil,
		PromptEvalCount:    nil,
		PromptEvalDuration: nil,
		EvalCount:          nil,
		EvalDuration:       nil,
		TotalDuration:      nil,
		LoadDuration:       nil,
	}
	proxyToClient := writer != nil // If w is nil, we don't want to proxy to the client
	completed := false
	var responseContent strings.Builder
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			// End of stream
			break
		}
		if err != nil {
			return &res, fmt.Errorf("error receiving from chat stream: %w", err)
		}

		// Accumulate the full response
		if resp.Response != "" {
			responseContent.WriteString(resp.Response)
		}
		// Check if this is done before attempting to write to the client
		if resp.Done {
			if !proxyToClient {
				util.LogInfo("Streaming completed, but not proxying to client")
				completed = true
				break
			}
			// For done message, always try to send to client if proxyToClient is true
		}

		if proxyToClient {
			// Write the line to the client
			if _, writeErr := writer.Write([]byte(resp.Response)); writeErr != nil {
				util.HandleError(writeErr)
				proxyToClient = false // Stop proxying on error
			} else {
				// Only attempt to flush if the write was successful
				if flushErr := writer.Flush(); flushErr != nil {
					util.HandleError(flushErr)
					proxyToClient = false // Stop proxying on error
				}
			}

			// Handle completion after successful write of final chunk
			if resp.Done {
				util.LogInfo("Streaming completed successfully")
				completed = true
				res.Response = responseContent.String()
				res.Done = true
				res.DoneReason = util.StrPtr(resp.DoneReason)
				res.CreatedAt = util.StrPtr(time.Now().Format(time.RFC3339))
				res.Model = req.Model
				res.DoneReason = util.StrPtr(resp.DoneReason)
				res.TotalDuration = util.Float32Ptr(float32(resp.TotalDuration))
				res.LoadDuration = util.Float32Ptr(float32(resp.LoadDuration))
				res.PromptEvalCount = util.IntPtr(int(resp.PromptEvalCount))
				res.PromptEvalDuration = util.Float32Ptr(float32(resp.PromptEvalDuration))
				res.EvalCount = util.IntPtr(int(resp.EvalCount))
				res.EvalDuration = util.Float32Ptr(float32(resp.EvalDuration))
				break
			}
		}
	}
	if proxyToClient {
		if flushErr := writer.Flush(); flushErr != nil {
			util.HandleError(flushErr)
		}
	}

	if completed {
		if !proxyToClient {
			util.LogWarning("Streaming completed but not proxying to client, returning accumulated content")
		}
		return &res, nil
	}

	return &res, nil

}

// GetEmbedding gets embeddings for text
func (c *GRPCClient) GetEmbedding(ctx context.Context, req models.EmbeddingReq) (*models.EmbeddingResponse, error) {
	// Check connection and reconnect if needed
	if err := c.ensureConnection(); err != nil {
		return nil, err
	}

	opts, err := json.Marshal(req.Options)
	if err != nil {
		return nil, util.HandleError(err)
	}
	ka := 0
	if req.KeepAlive != nil {
		ka = *req.KeepAlive
	}
	tr := false
	if req.Truncate != nil {
		tr = *req.Truncate
	}

	// Create the request
	grpcReq := &EmbeddingReq{
		Model:     req.Model,
		Input:     req.Input,
		Truncate:  tr,
		KeepAlive: int32(ka),
		Options:   &EmbeddingReq_Options{unknownFields: opts},
	}

	// Make the request
	resp, err := c.client.GetEmbedding(ctx, grpcReq)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}

	emb := make([][]float32, len(resp.Embeddings))
	for i, embList := range resp.Embeddings {
		// Convert the flat array to a slice of float32
		emb[i] = make([]float32, len(embList.Values))
		for j, val := range embList.Values {
			emb[i][j] = float32(val)
		}
	}

	res := &models.EmbeddingResponse{
		Model:           resp.Model,
		Embeddings:      emb,
		TotalDuration:   int(resp.TotalDuration),
		LoadDuration:    int(resp.LoadDuration),
		PromptEvalCount: int(resp.PromptEvalCount),
	}

	return res, nil
}

// GenerateImage requests image generation
func (c *GRPCClient) GenerateImage(ctx context.Context, req models.ImageGenerateRequest) (*models.ImageGenerateResponse, error) {
	// Check connection and reconnect if needed
	if err := c.ensureConnection(); err != nil {
		return nil, err
	}

	np := ""
	if req.NegativePrompt != nil {
		np = *req.NegativePrompt
	}

	w := 0
	if req.Width != nil {
		w = *req.Width
	}
	h := 0
	if req.Height != nil {
		h = *req.Height
	}
	is := 0
	if req.InferenceSteps != nil {
		is = *req.InferenceSteps
	}
	gs := 0.0
	if req.GuidanceScale != nil {
		gs = float64(*req.GuidanceScale)
	}
	lmm := false
	if req.LowMemoryMode != nil {
		lmm = *req.LowMemoryMode
	}
	loras := []string{}
	if req.Loras != nil {
		loras = req.Loras
	}
	iid := -1
	if req.ImageID != nil {
		iid = *req.ImageID
	}
	url := ""
	if req.URL != nil {
		url = *req.URL
	}
	fn := ""
	if req.Filename != nil {
		fn = *req.Filename
	}

	// Convert request to gRPC format
	grpcReq := &ImageGenerateRequest{
		Prompt:         req.Prompt,
		NegativePrompt: np,
		Model:          req.Model,
		Width:          int32(w),
		Height:         int32(h),
		InferenceSteps: int32(is),
		GuidanceScale:  float32(gs),
		LowMemoryMode:  lmm,
		Loras:          loras,
		ImageId:        int32(iid),
		Url:            url,
		Filename:       fn,
	}

	// Make the request
	resp, err := c.client.GenerateImage(ctx, grpcReq)
	if err != nil {
		return nil, fmt.Errorf("failed to request image generation: %w", err)
	}

	res := models.ImageGenerateResponse{
		Image:    resp.Image,
		Download: resp.Download,
	}

	return &res, nil
}
