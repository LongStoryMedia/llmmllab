// Deprecated: Use the routes package instead
package api

// import (
// 	"bufio"
// 	"context"
// 	"maistro/auth"
// 	pxcx "maistro/context"
// 	"maistro/models"
// 	"maistro/proxy"
// 	"maistro/session"
// 	"maistro/storage"
// 	"maistro/util"
// 	"net/http"
// 	"time"

// 	"github.com/gofiber/fiber/v2"
// 	"github.com/sirupsen/logrus"
// )

// // Deprecated: Use the routes package instead
// func RegisterChatRoutes(app *fiber.App) {
// 	// app.Post("/api/chat/:conversationId/pause", PauseHandler)   // Pause chat processing
// 	// app.Post("/api/chat/:conversationId/resume", ResumeHandler) // Resume chat processing
// 	// app.Post("/api/chat/:conversationId/cancel", CancelHandler) // Cancel chat processing
// 	app.Post("/api/chat", ChatHandler) // List available chat models
// }

// // ChatHandler forwards the request to Ollama and streams the response back to the client
// func ChatHandler(c *fiber.Ctx) error {
// 	// Parse the incoming request
// 	var chatReq models.ChatRequest
// 	if err := c.BodyParser(&chatReq); err != nil {
// 		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
// 	}

// 	uid := c.UserContext().Value(auth.UserIDKey).(string)
// 	// Get or create conversation context
// 	cc, err := pxcx.GetCachedConversation(uid, chatReq.ConversationID)
// 	if err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to process conversation")
// 	}

// 	ss := session.GlobalStageManager.GetSessionState(cc.UserID, cc.ConversationID)
// 	initState := ss.GetStage(models.SocketStageTypeInitializing)

// 	if chatReq.Metadata != nil {
// 		// Handle image generation
// 		if chatReq.Metadata.GenerateImage {
// 			if cc.Notes == nil {
// 				cc.Notes = make([]string, 0)
// 			}
// 			cc.Notes = append(cc.Notes, "An image is being generated based on the request content using the selected Image Generation Model Profile.")
// 		}
// 		go func(uid string, cid int) {
// 			ctx, cancel := context.WithTimeout(context.Background(), time.Minute*60)
// 			defer cancel()

// 			uc, err := pxcx.GetUserConfig(uid)
// 			if err != nil {
// 				util.HandleError(err)
// 				return
// 			}

// 			fp, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, uc.ModelProfiles.FormattingProfileID)
// 			if err != nil {
// 				util.HandleError(err)
// 				return
// 			}

// 			p, err := pxcx.FmtQuery(ctx, fp, chatReq.Content, pxcx.ImgGeneratePrompt)
// 			if err != nil {
// 				util.HandleError(err)
// 				return
// 			}

// 			// Prepare the image generation request
// 			imgReq := models.ImageGenerateRequest{
// 				Prompt:         p,
// 				ConversationID: cid,
// 				Model:          "nvidia/Cosmos-Predict2-14B-Text2Image",
// 				Loras:          []string{"black-forest-labs/FLUX.1-dev"},
// 			}
// 			if err := GenImg(ctx, uid, imgReq); err != nil {
// 				util.HandleError(err)
// 				return
// 			}
// 		}(uid, cc.ConversationID)
// 	}

// 	initState.Complete("Chat request initialized successfully")

// 	ollamaReqBody, err := cc.PrepareOllamaRequest(c.UserContext(), chatReq)
// 	if err != nil {
// 		return handleError(err, fiber.StatusInternalServerError, "Failed to prepare Ollama request")
// 	}

// 	handler, statusCode, err := proxy.GetProxyHandler[*models.OllamaChatResp](c.UserContext(), ss, ollamaReqBody, c.Path(), http.MethodPost, true, time.Minute*10, nil)
// 	if err != nil {
// 		return handleError(err, fiber.StatusBadGateway, "Error during streaming")
// 	}
// 	c.Status(statusCode)
// 	var res string
// 	ss.Checkpoint()

// 	c.Response().SetBodyStreamWriter(func(w *bufio.Writer) {
// 		res, err = handler(w)
// 		if err != nil {
// 			if proxy.IsIncompleteError(err) {
// 				util.HandleError(err)
// 			} else {
// 				handleError(err, fiber.StatusInternalServerError, "Error during handler execution")
// 				return
// 			}
// 		}

// 		// Only store the assistant response if there was no context error
// 		if res != "" {
// 			// Apply refinement steps to the response in a separate goroutine
// 			// to avoid keeping the client connection open longer than necessary
// 			// go func(response, userID string, convID int) {
// 			// 	pxcx.RefineResponse(response, userMessage, userID, convID)
// 			// }(res, cc.UserID, cc.ConversationID)

// 			go func(r string, cctx *pxcx.ConversationContext) {
// 				ctx, cancel := context.WithTimeout(context.Background(), time.Minute*60)
// 				defer cancel()
// 				_, err := cc.AddAssistantMessage(ctx, res)
// 				if err != nil {
// 					util.HandleError(err)
// 				} else {
// 					util.LogInfo("Successfully stored assistant message in conversation", logrus.Fields{"conversationId": cc.ConversationID})
// 				}
// 			}(res, cc)
// 		} else if res == "" {
// 			util.LogWarning("Empty assistant response, not storing")
// 		}

// 		cc.Notes = make([]string, 0)

// 		// Notify of completion
// 		ss.GetStage(models.SocketStageTypeProcessing).Complete("Chat response generated successfully")
// 	})

// 	return nil
// }
