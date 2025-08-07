package proto

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maistro/config"
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
func GetGRPCClient() (*GRPCClient, error) {
	if gRPCClient != nil {
		// If the client already exists, return it
		return gRPCClient, nil
	}
	conf := config.GetConfig(nil)
	gRPCClient = &GRPCClient{
		address: fmt.Sprintf("%s:%d", conf.InferenceServices.Host, conf.InferenceServices.Port),
		secure:  false,
		apiKey:  "",
	}

	// Initialize the connection
	if err := gRPCClient.initConnection(); err != nil {
		return nil, err
	}

	util.LogInfo("gRPC client initialized", logrus.Fields{
		"address": gRPCClient.address,
		"secure":  gRPCClient.secure,
	})

	return gRPCClient, nil
}

// GRPCClient is the global singleton instance of the gRPC client
var gRPCClient *GRPCClient

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

	// Add more detailed logging for connection attempts
	util.LogInfo("Attempting to connect to inference service", logrus.Fields{
		"address": c.address,
		"secure":  c.secure,
		"timeout": "30s",
	})

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

	// Set connection timeout (increase to 30 seconds for debugging)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Establish the connection
	var err error
	c.conn, err = grpc.DialContext(ctx, c.address, opts...)
	if err != nil {
		util.HandleError(err, logrus.Fields{
			"address": c.address,
			"secure":  c.secure,
		})
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

func convertParamsToGRPC(opts *models.ModelParameters) *ModelParameters {
	mp := ModelParameters{}
	if opts == nil {
		return &mp // Return empty ModelParameters if opts is nil
	}
	if opts.NumCtx != nil {
		mp.NumCtx = int32(*opts.NumCtx)
	}
	if opts.MinP != nil {
		mp.MinP = float64(*opts.MinP)
	}
	if opts.NumPredict != nil {
		mp.NumPredict = int32(*opts.NumPredict)
	}
	if opts.Temperature != nil {
		mp.Temperature = float64(*opts.Temperature)
	}
	if opts.TopP != nil {
		mp.TopP = float64(*opts.TopP)
	}
	if opts.TopK != nil {
		mp.TopK = int32(*opts.TopK)
	}
	if opts.RepeatLastN != nil {
		mp.RepeatLastN = int32(*opts.RepeatLastN)
	}
	if opts.RepeatPenalty != nil {
		mp.RepeatPenalty = float64(*opts.RepeatPenalty)
	}
	if opts.Stop != nil {
		mp.Stop = make([]string, len(opts.Stop))
		copy(mp.Stop, opts.Stop)
	}
	return &mp
}

func messageRoleModelToProto(r models.MessageRole) MessageRole {
	return MessageRole(MessageRole_value[strings.ToUpper(string(r))])
}

// ChatStream streams chat responses
func (c *GRPCClient) ChatStream(ctx context.Context, req *models.ChatReq, writer *bufio.Writer) (*models.ChatResponse, error) {
	// Check connection and reconnect if needed
	if err := c.ensureConnection(); err != nil {
		return nil, err
	}

	if req.ConversationID == nil {
		return nil, util.HandleError(errors.New("conversation ID is required for chat requests"))
	}

	cid := int32(*req.ConversationID)

	// Instead, properly construct the protobuf message:
	var format *ChatReq_Format
	if req.Format != nil {
		format = &ChatReq_Format{}
		// Only set specific fields you need from req.Format
	} else {
		format = nil // Use nil if no format is provided
	}

	// When converting messages, don't use JSON for embedded structs
	msgs := make([]*Message, 0, len(req.Messages))
	for _, msg := range req.Messages {
		// For tool calls, use properly typed protobuf messages
		tc := make([]*Message_ToolCallsItem, 0, len(msg.ToolCalls))
		for range msg.ToolCalls {
			// Create proper protobuf tool call objects:
			pbToolCall := &Message_ToolCallsItem{}
			tc = append(tc, pbToolCall)
		}

		msgId := -1
		if msg.ID != nil {
			msgId = *msg.ID
		}

		// Convert models.MessageContent to []*MessageContent
		content := make([]*MessageContent, len(msg.Content))
		for i, c := range msg.Content {
			contentType := MessageContentType_TEXT
			switch c.Type {
			case models.MessageContentTypeImage:
				contentType = MessageContentType_IMAGE
			case models.MessageContentTypeToolCall:
				contentType = MessageContentType_TOOL_CALL
			}
			cont := MessageContent{
				Type: contentType,
			}
			if c.Text != nil {
				cont.Text = *c.Text
			}
			if c.URL != nil {
				cont.Url = *c.URL
			}
			content[i] = &cont
		}

		msgs = append(msgs, &Message{
			Role:           messageRoleModelToProto(msg.Role),
			Content:        content,
			ToolCalls:      tc, // Use properly typed tool calls
			Id:             int32(msgId),
			ConversationId: int32(cid),
		})
	}

	util.LogDebug("Preparing chat request", logrus.Fields{
		"msgs": msgs,
	})

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

	thk := true
	if req.Think != nil {
		thk = *req.Think
	}

	// Convert request to gRPC format - use properly typed fields
	grpcReq := &ChatReq{
		ConversationId: cid,
		Model:          req.Model,
		Messages:       msgs,
		Stream:         req.Stream,
		Format:         format, // Use properly typed format instead of unknownFields
		KeepAlive:      int32(ka),
		Tools:          t,
		Options:        convertParamsToGRPC(req.Options),
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
		Message: &models.Message{
			Role:      "assistant",
			Content:   nil,
			ToolCalls: nil,
			Thinking:  nil,
		},
		CreatedAt:          time.Now(),
		Model:              req.Model,
		Context:            nil,
		FinishReason:       nil,
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
			return nil, fmt.Errorf("error receiving from chat stream: %w", err)
		}

		// Accumulate the full response
		for _, content := range resp.Message.Content {
			if content.Text != "" {
				fmt.Print(content.Text)
				responseContent.WriteString(content.Text)
			}
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
			r, err := json.Marshal(resp)
			if err != nil {
				util.HandleError(err)
				proxyToClient = false
			}
			// Write the line to the client
			if _, writeErr := writer.Write(r); writeErr != nil {
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
		Options:   convertParamsToGRPC(req.Options),
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
