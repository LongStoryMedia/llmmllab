package middleware

import (
	"context"
	"maistro/auth"
	pxcx "maistro/context"
	"maistro/util"
	"net/http"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

func Convo(c *fiber.Ctx) error {
	cid, err := c.ParamsInt(string(CIDPKey), 0)
	if err != nil || cid <= 0 {
		util.LogWarning("Invalid conversation ID", logrus.Fields{
			"error": err,
			"param": c.Params(string(CIDPKey)),
			"path":  c.Path(),
		})
		return fiber.NewError(fiber.StatusBadRequest, "invalid conversation ID")
	}

	cc, err := pxcx.GetCachedConversation(c.UserContext().Value(auth.UserIDKey).(string), cid)
	if err != nil {
		util.HandleError(err, logrus.Fields{"error": "Failed to get conversation context", "conversation_id": cid})
		return c.Status(http.StatusInternalServerError).JSON(fiber.Map{
			"error": "internal server error",
		})
	}

	// Set up context with auth information
	ctx := context.WithValue(c.UserContext(), pxcx.ConversationContextKey, cc)
	c.SetUserContext(ctx)

	return c.Next()
}
