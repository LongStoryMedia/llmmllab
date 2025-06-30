package handlers

import (
	"bufio"
	"context"
	"maistro/auth"
	pxcx "maistro/context"
	"maistro/middleware"
	"maistro/models"
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
	var chatReq models.ChatRequest
	parseStartTime := time.Now()
	if err := c.BodyParser(&chatReq); err != nil {
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

	ss := getSessionState(c)
	cc := getConversationContext(c)
	cfg := getUserConfig(c)

	initState := ss.GetStage(models.SocketStageTypeInitializing)

	if chatReq.Metadata != nil {
		// Handle image generation
		if chatReq.Metadata.GenerateImage {
			imageGenStartTime := time.Now()
			util.LogInfo("Starting image generation flow", nil)

			if cc.Notes == nil {
				cc.Notes = make([]string, 0)
			}
			wg := &sync.WaitGroup{}
			wg.Add(1)
			go func(uid string, cid int, cfg *models.UserConfig) {
				defer wg.Done()
				p := chatReq.Content

				if cfg.ImageGeneration.AutoPromptRefinement {
					fp, err := storage.ModelProfileStoreInstance.GetModelProfile(context.Background(), cfg.ModelProfiles.ImageGenerationPromptProfileID)
					if err != nil {
						util.HandleError(err)
						return
					}

					ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
					defer cancel()
					p, err = pxcx.FmtQuery(ctx, fp, chatReq.Content, pxcx.ImgGeneratePrompt)
					if err != nil {
						util.HandleError(err)
						return
					}
				}

				// Prepare the image generation request
				imgReq := models.ImageGenerateRequest{
					Prompt:         p,
					NegativePrompt: util.StrPtr("anime, cartoon, sketch, drawing, lowres, bad anatomy, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username"),
				}

				svc.GetInferenceService().GenerateImage(context.Background(), uid, cid, imgReq)
			}(cfg.UserID, cc.ConversationID, cfg)
			wg.Wait()
			imageGenTotalTime := time.Since(imageGenStartTime)
			svc.GetSocketService().SendCompletion(models.SocketStageTypeGeneratingImage, cc.ConversationID, cfg.UserID, "Image generation request sent", nil)

			// Log image generation timing
			util.LogInfo("Image generation flow completed", logrus.Fields{
				"imageGenTime_ms": imageGenTotalTime.Milliseconds(),
			})

			return nil
		}

		// TODO: remove this after debugging
		// svc.GetSocketService().PauseAndBroadcast(cc.ConversationID, cfg.UserID)
		// svc.GetSocketService().CancelAndBroadcast(cc.ConversationID, cfg.UserID)
	}

	initState.Complete("Chat request initialized successfully")

	// Time the request preparation
	prepReqStartTime := time.Now()
	_, req, err := cc.PrepareOllamaRequest(c.UserContext(), chatReq)
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

	var res string
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

	c.Response().SetBodyStreamWriter(func(w *bufio.Writer) {
		// Start the processing stage
		streamStartTime := time.Now()
		if res, err = svc.GetInferenceService().RelayUserMessage(ctx, profile, req.Messages, cfg.UserID, cc.ConversationID, w); err != nil {
			if proxy.IsIncompleteError(err) {
				ss.GetStage(models.SocketStageTypeProcessing).Fail("Incomplete response", err)
			} else {
				handleApiError(err, fiber.StatusInternalServerError, "Error during handler execution")
				return
			}
		}

		// Only store the assistant response if there was no context error
		if res != "" {
			// Apply refinement steps to the response in a separate goroutine
			// to avoid keeping the client connection open longer than necessary
			// go func(response, userID string, convID int) {
			// 	pxcx.RefineResponse(response, userMessage, userID, convID)
			// }(res, cc.UserID, cc.ConversationID)

			go func(r string, cctx *pxcx.ConversationContext) {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute*60)
				defer cancel()
				_, err := cctx.AddAssistantMessage(ctx, res)
				if err != nil {
					util.HandleError(err)
				} else {
					util.LogInfo("Successfully stored assistant message in conversation", logrus.Fields{"conversationId": cctx.ConversationID})
				}
			}(res, cc)
		} else if res == "" {
			util.LogWarning("Empty assistant response, not storing")
		}

		cc.Notes = make([]string, 0)

		// Notify of completion
		ss.GetStage(models.SocketStageTypeProcessing).Complete("Chat response generated successfully")
		streamDuration := time.Since(streamStartTime)
		util.LogInfo("Chat response streaming completed", logrus.Fields{
			"streamDuration_ms": streamDuration.Milliseconds(),
			"responseLength":    len(res),
		})
	})

	// Log the total execution time for the handler
	totalExecutionTime := time.Since(startTime)
	util.LogInfo("ChatHandler execution completed", logrus.Fields{
		"totalExecutionTime_ms": totalExecutionTime.Milliseconds(),
		"responseLength":        len(res),
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
	cc, err := pxcx.GetOrCreateConversation(c.UserContext(), uid, nil)
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to create conversation")
	}

	var req struct {
		Model string `json:"model"`
		Title string `json:"title"`
	}
	if err := c.BodyParser(&req); err != nil {
		return handleApiError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	return c.JSON(fiber.Map{string(pxcx.ConversationContextKey): cc.ConversationID})
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
