package middleware

import (
	"maistro/auth"
	"maistro/session"
	"maistro/util"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

func Session(c *fiber.Ctx) error {
	cid, err := c.ParamsInt(string(CIDPKey), 0)
	if err != nil || cid <= 0 {
		util.LogWarning("Invalid conversation ID", logrus.Fields{
			"error": err,
			"param": c.Params(string(CIDPKey)),
			"path":  c.Path(),
		})
		return fiber.NewError(fiber.StatusBadRequest, "invalid conversation ID")
	}

	uid := c.UserContext().Value(auth.UserIDKey).(string)

	util.LogInfo("Session middleware params", logrus.Fields{
		"conversationId": cid,
		"path":           c.Path(),
	})

	if cid <= 0 {
		util.LogWarning("conversation ID is required")
		return fiber.NewError(fiber.StatusBadRequest, "conversation ID is required")
	}

	ss := session.GlobalStageManager.GetSessionState(uid, cid)
	c.SetUserContext(context.WithValue(c.UserContext(), session.SessionCtxKey, ss))
	return c.Next()
}
