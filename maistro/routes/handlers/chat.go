package handlers

import (
	"bufio"
	"context"
	"maistro/auth"
	pxcx "maistro/context"
	"maistro/middleware"
	"maistro/models"
	"maistro/proto"
	"maistro/proxy"
	svc "maistro/services"
	"maistro/storage"
	"maistro/util"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// ChatHandler forwards the request to Ollama and streams the response back to the client
func ChatHandler(c *fiber.Ctx) error {
	// Start timing the handler execution
	startTime := time.Now()

	// Record timing metrics using a deferred function
	defer func() {
		totalTime := time.Since(startTime)
		util.LogInfo("ChatHandler execution completed", logrus.Fields{
			"totalExecutionTime_ms": totalTime.Milliseconds(),
		})
	}()

	// Parse the incoming request
	var req models.ChatReq
	parseStartTime := time.Now()
	if err := c.BodyParser(&req); err != nil {
		parseTime := time.Since(parseStartTime)
		util.LogWarning("Failed to parse request", logrus.Fields{
			"error":        err,
			"parseTime_ms": parseTime.Milliseconds(),
		})
		return handleApiError(err, fiber.StatusBadRequest, "Invalid request body")
	}
	parseTime := time.Since(parseStartTime)

	// Log successful parsing
	util.LogInfo("Request parsed successfully", logrus.Fields{
		"parseTime_ms": parseTime.Milliseconds(),
	})

	if req.Messages == nil || len(req.Messages) == 0 {
		util.LogWarning("Empty messages in chat request", logrus.Fields{
			"request": req,
		})
		return fiber.NewError(fiber.StatusBadRequest, "Messages cannot be empty")
	}

	ss := getSessionState(c)
	cc := getConversationContext(c)
	cfg := getUserConfig(c)

	initState := ss.GetStage(models.SocketStageTypeInitializing)

	for _, msg := range req.Messages {
		if msg.Role == "" {
			util.LogWarning("Message role is empty", logrus.Fields{
				"message": msg,
			})
			return fiber.NewError(fiber.StatusBadRequest, "Message role cannot be empty")
		}
		if len(msg.Content) == 0 {
			util.LogWarning("Message content is empty", logrus.Fields{
				"message": msg,
			})
			return fiber.NewError(fiber.StatusBadRequest, "Message content cannot be empty")
		}
		for _, content := range msg.Content {
			if content.Type == "" {
				util.LogWarning("Message content type is empty", logrus.Fields{
					"message": msg,
				})
				return fiber.NewError(fiber.StatusBadRequest, "Message content type cannot be empty")
			}

			if content.Type == models.MessageContentTypeImageGeneration {
				if content.Text == nil {
					util.LogWarning("Image generation content text is empty", logrus.Fields{
						"message": msg,
					})
					return fiber.NewError(fiber.StatusBadRequest, "Image generation content text cannot be empty")
				}

				imageGenStartTime := time.Now()
				util.LogInfo("Starting image generation flow", nil)

				wg := &sync.WaitGroup{}
				wg.Add(1)
				go func(prompt string, uid string, cid int, cfg *models.UserConfig) {
					defer wg.Done()
					var p string
					if cfg.ImageGeneration.AutoPromptRefinement {
						fp, err := storage.ModelProfileStoreInstance.GetModelProfile(context.Background(), cfg.ModelProfiles.ImageGenerationPromptProfileID)
						if err != nil {
							util.HandleError(err)
							return
						}

						ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
						defer cancel()
						p, err = pxcx.FmtQuery(ctx, fp, prompt, pxcx.ImgGeneratePrompt)
						if err != nil {
							util.HandleError(err)
							return
						}
					}

					// Prepare the image generation request
					imgReq := models.ImageGenerateRequest{
						Prompt:         p,
						NegativePrompt: util.StrPtr("anime, cartoon, sketch, drawing, lowres, bad anatomy, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username"),
						Model:          "stabilityai-stable-diffusion-3.5-medium",
					}

					svc.GetInferenceService().GenerateImage(context.Background(), uid, cid, imgReq)
				}(*content.Text, cfg.UserID, cc.GetConversationID(), cfg)
				wg.Wait()
				imageGenTotalTime := time.Since(imageGenStartTime)
				svc.GetSocketService().SendCompletion(models.SocketStageTypeGeneratingImage, cc.GetConversationID(), cfg.UserID, "Image generation request sent", nil)

				// Log image generation timing
				util.LogInfo("Image generation flow completed", logrus.Fields{
					"imageGenTime_ms": imageGenTotalTime.Milliseconds(),
				})
			}
		}
	}

	initState.Complete("Chat request initialized successfully")

	// Time the request preparation
	prepReqStartTime := time.Now()
	_, err := cc.PrepareOllamaRequest(c.UserContext(), &req)
	prepReqTime := time.Since(prepReqStartTime)

	if err != nil {
		util.LogWarning("Failed to prepare Ollama request", logrus.Fields{
			"error":          err,
			"prepReqTime_ms": prepReqTime.Milliseconds(),
		})
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to prepare Ollama request")
	}

	util.LogInfo("Ollama request prepared successfully", logrus.Fields{
		"prepReqTime_ms": prepReqTime.Milliseconds(),
	})

	ss.Checkpoint()

	// Time model profile retrieval
	profileStartTime := time.Now()
	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(c.UserContext(), cfg.ModelProfiles.PrimaryProfileID)
	profileRetrieveTime := time.Since(profileStartTime)

	if err != nil {
		util.LogWarning("Failed to retrieve model profile", logrus.Fields{
			"error":                  err,
			"profileRetrieveTime_ms": profileRetrieveTime.Milliseconds(),
		})
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve model profile")
	}

	util.LogInfo("Model profile retrieved successfully", logrus.Fields{
		"profileId":              cfg.ModelProfiles.PrimaryProfileID,
		"profileRetrieveTime_ms": profileRetrieveTime.Milliseconds(),
	})

	ctx := c.UserContext()

	gc, err := proto.GetGRPCClient()
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to get gRPC client")
	}

	req.Options = &profile.Parameters
	req.Think = profile.Think

	c.Response().SetBodyStreamWriter(func(w *bufio.Writer) {
		// Start the processing stage
		streamStartTime := time.Now()

		sCtx, cancel := context.WithTimeout(ctx, time.Minute*60)
		defer cancel()

		res, err := gc.ChatStream(sCtx, &req, w)
		if err != nil {
			if proxy.IsIncompleteError(err) {
				ss.GetStage(models.SocketStageTypeProcessing).Fail("Incomplete response", err)
			} else {
				handleApiError(err, fiber.StatusInternalServerError, "Error during handler execution")
				return
			}
		}

		util.LogInfo("Chat response streaming completed", logrus.Fields{
			"response": res,
		})

		// if res, err = svc.GetInferenceService().RelayUserMessage(ctx, profile, req.Messages, cfg.UserID, cc.GetConversationID(), w); err != nil {
		// 	if proxy.IsIncompleteError(err) {
		// 		ss.GetStage(models.SocketStageTypeProcessing).Fail("Incomplete response", err)
		// 	} else {
		// 		handleApiError(err, fiber.StatusInternalServerError, "Error during handler execution")
		// 		return
		// 	}
		// }

		// Only store the assistant response if there was no context error
		if res != nil {
			// Apply refinement steps to the response in a separate goroutine
			// to avoid keeping the client connection open longer than necessary
			// go func(response, userID string, convID int) {
			// 	pxcx.RefineResponse(response, userMessage, userID, convID)
			// }(res, cc.UserID, cc.ConversationID)

			go func(r *models.ChatResponse, cctx pxcx.ConversationContext) {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute*60)
				defer cancel()
				_, err := cctx.AddAssistantMessage(ctx, r.Message)
				if err != nil {
					util.HandleError(err)
				} else {
					util.LogInfo("Successfully stored assistant message in conversation", logrus.Fields{"conversationId": cctx.GetConversationID()})
				}
			}(res, cc)
		} else {
			util.LogWarning("Empty assistant response, not storing")
		}

		cc.ClearNotes()

		// Notify of completion
		// ss.GetStage(models.SocketStageTypeProcessing).Complete("Chat response generated successfully")
		streamDuration := time.Since(streamStartTime)
		util.LogInfo("Chat response streaming completed", logrus.Fields{
			"streamDuration_ms": streamDuration.Milliseconds(),
		})
	})

	// Log the total execution time for the handler
	totalExecutionTime := time.Since(startTime)
	util.LogInfo("ChatHandler execution completed", logrus.Fields{
		"totalExecutionTime_ms": totalExecutionTime.Milliseconds(),
	})

	return nil
}

// GetUserConversations returns all conversations for the authenticated user
func GetUserConversations(c *fiber.Ctx) error {
	cfg := getUserConfig(c)

	conversations, err := storage.ConversationStoreInstance.GetUserConversations(c.UserContext(), cfg.UserID)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve conversations")
	}

	return c.JSON(conversations)
}

// GetConversation returns a specific conversation
func GetConversation(c *fiber.Ctx) error {
	cid, err := c.ParamsInt(string(middleware.CIDPKey), 0)
	if err != nil || cid <= 0 {
		util.LogWarning("Invalid conversation ID", logrus.Fields{
			"error": err,
			"param": c.Params(string(middleware.CIDPKey)),
			"path":  c.Path(),
		})
		return fiber.NewError(fiber.StatusBadRequest, "invalid conversation ID")
	}

	conversation, err := storage.ConversationStoreInstance.GetConversation(c.UserContext(), cid)
	if err != nil {
		return fiber.NewError(fiber.StatusNotFound, "Conversation not found")
	}

	if !auth.CanAccess(c, conversation.UserID) {
		return fiber.NewError(fiber.StatusForbidden, "Access denied")
	}

	return c.JSON(conversation)
}

// GetConversationMessages returns all messages in a conversation
func GetConversationMessages(c *fiber.Ctx) error {
	cid, err := c.ParamsInt(string(middleware.CIDPKey), 0)
	if err != nil || cid <= 0 {
		util.LogWarning("Invalid conversation ID", logrus.Fields{
			"error": err,
			"param": c.Params(string(middleware.CIDPKey)),
			"path":  c.Path(),
		})
		return fiber.NewError(fiber.StatusBadRequest, "invalid conversation ID")
	}

	// Verify ownership
	conversation, err := storage.ConversationStoreInstance.GetConversation(c.UserContext(), cid)
	if err != nil {
		return fiber.NewError(fiber.StatusNotFound, "Conversation not found")
	}

	if !auth.CanAccess(c, conversation.UserID) {
		return fiber.NewError(fiber.StatusForbidden, "Access denied")
	}

	messages, err := storage.MessageStoreInstance.GetConversationHistory(c.UserContext(), cid)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to retrieve messages")
	}

	return c.JSON(messages)
}

// DeleteConversation deletes a conversation and all its messages
func DeleteConversation(c *fiber.Ctx) error {
	cid, err := c.ParamsInt(string(middleware.CIDPKey), 0)
	if err != nil || cid <= 0 {
		util.LogWarning("Invalid conversation ID", logrus.Fields{
			"error": err,
			"param": c.Params(string(middleware.CIDPKey)),
			"path":  c.Path(),
		})
		return fiber.NewError(fiber.StatusBadRequest, "invalid conversation ID")
	}

	// Verify ownership
	conversation, err := storage.ConversationStoreInstance.GetConversation(c.UserContext(), cid)
	if err != nil {
		return handleApiError(err, fiber.StatusNotFound, "Conversation not found")
	}
	if !auth.CanAccess(c, conversation.UserID) {
		return fiber.NewError(fiber.StatusForbidden, "Access denied")
	}

	// Add deletion function to storage package
	err = storage.ConversationStoreInstance.DeleteConversation(c.UserContext(), cid)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to delete conversation")
	}

	return c.SendStatus(fiber.StatusOK)
}

// CreateConversation creates a new conversation
func CreateConversation(c *fiber.Ctx) error {
	uid := c.UserContext().Value(auth.UserIDKey).(string)

	var req struct {
		Model string `json:"model"`
		Title string `json:"title"`
	}
	if err := c.BodyParser(&req); err != nil {
		return handleApiError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	// Create the conversation in storage
	title := req.Title
	if title == "" {
		title = "New Conversation"
	}
	conversationID, err := storage.ConversationStoreInstance.CreateConversation(c.UserContext(), uid, title)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to create conversation")
	}

	// Optionally, update the model if needed (not shown here)

	// Fetch the full conversation object
	conversation, err := storage.ConversationStoreInstance.GetConversation(c.UserContext(), conversationID)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to fetch conversation after creation")
	}

	return c.JSON(conversation)
}

func Pause(c *fiber.Ctx) error {
	ss := getSessionState(c)
	ss.Pause()
	return c.SendStatus(fiber.StatusAccepted)
}

func Resume(c *fiber.Ctx) error {
	ss := getSessionState(c)
	ss.Resume()
	return c.SendStatus(fiber.StatusAccepted)
}

func Cancel(c *fiber.Ctx) error {
	ss := getSessionState(c)
	ss.Cancel()
	return c.SendStatus(fiber.StatusAccepted)
}
