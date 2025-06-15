package api

import (
	"bufio"
	"context"
	"maistro/auth"
	pxcx "maistro/context"
	"maistro/models"
	"maistro/proxy"
	"maistro/util"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// ChatHandler forwards the request to Ollama and streams the response back to the client
func ChatHandler(c *fiber.Ctx) error {
	// Parse the incoming request
	var chatReq models.ChatRequest
	if err := c.BodyParser(&chatReq); err != nil {
		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	uid := c.UserContext().Value(auth.UserIDKey).(string)
	// Get or create conversation context
	cc, err := pxcx.GetCachedConversation(uid, chatReq.ConversationID)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to process conversation")
	}

	util.LogDebug("Chat Metadata", logrus.Fields{
		"conversationId": cc.ConversationID,
		"userId":         cc.UserID,
		"metadata":       chatReq.Metadata,
	})

	if chatReq.Metadata != nil {
		// Handle image generation
		if chatReq.Metadata.GenerateImage {
			if cc.Notes == nil {
				cc.Notes = make([]string, 0)
			}
			cc.Notes = append(cc.Notes, "An image is being generated based on the request content using the selected Image Generation Model Profile.")
		}

		// Handle continuation request
		if chatReq.Metadata.IsContinuation {
			util.LogInfo("Handling continuation request with additional context", logrus.Fields{
				"conversationId": cc.ConversationID,
				"userId":         cc.UserID,
			})
			// Add a system note to indicate this is a continuation
			if cc.Notes == nil {
				cc.Notes = make([]string, 0)
			}
			cc.Notes = append(cc.Notes, "This is a continuation of a previous request with additional context.")
		}
	}

	ollamaReqBody, err := cc.PrepareOllamaRequest(c.UserContext(), chatReq.Content)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to prepare Ollama request")
	}

	handler, statusCode, err := proxy.GetProxyHandler[*models.OllamaChatResp](c.UserContext(), ollamaReqBody, c.Path(), http.MethodPost, true, time.Minute*10, nil)
	if err != nil {
		return handleError(err, fiber.StatusBadGateway, "Error during streaming")
	}
	c.Status(statusCode)
	var res string

	c.Response().SetBodyStreamWriter(func(w *bufio.Writer) {
		res, err = handler(w)
		if err != nil {
			if proxy.IsIncompleteError(err) {
				util.HandleError(err)
			} else {
				handleError(err, fiber.StatusInternalServerError, "Error during handler execution")
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
				_, err := cc.AddAssistantMessage(ctx, res)
				if err != nil {
					util.HandleError(err)
				} else {
					util.LogInfo("Successfully stored assistant message in conversation", logrus.Fields{"conversationId": cc.ConversationID})
				}
			}(res, cc)
		} else if res == "" {
			util.LogWarning("Empty assistant response, not storing")
		}

		cc.Notes = make([]string, 0)

		// Log the completion of the chat streaming
		util.LogInfo("Chat streaming complete", logrus.Fields{
			"conversationId": cc.ConversationID,
			"userId":         cc.UserID,
		})
	})

	return nil
}
