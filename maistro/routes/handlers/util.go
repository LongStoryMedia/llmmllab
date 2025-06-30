package handlers

import (
	"context"
	"maistro/auth"
	"maistro/config"
	pxcx "maistro/context"
	"maistro/middleware"
	"maistro/models"
	"maistro/session"
	"maistro/util"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// handleApiError is a helper function that logs the error and returns a fiber error
// to standardize error handling across API endpoints
func handleApiError(err error, status int, message string) error {
	util.HandleErrorAtCallLevel(err, 2)
	return fiber.NewError(status, message)
}

// getUserConfig retrieves the user configuration from the user context
func getUserConfig(c *fiber.Ctx) *models.UserConfig {
	userCtx := c.UserContext()
	if userCtx == nil {
		util.LogWarning("User context is nil. Cannot retrieve user configuration.", logrus.Fields{"path": c.Path()})
		return nil
	}

	cfgCtx := userCtx.Value(config.CfgKey)
	if cfgCtx == nil {
		util.LogWarning("User configuration not found in user context. Creating a new one.", logrus.Fields{"path": c.Path()})
		uid := userCtx.Value(auth.UserIDKey).(string)
		cfg, _ := pxcx.GetUserConfig(uid)
		c.SetUserContext(context.WithValue(userCtx, config.CfgKey, cfg))
		return cfg
	}

	if cfg, ok := cfgCtx.(*models.UserConfig); ok {
		return cfg
	}

	util.LogWarning("User configuration is not of type *models.UserConfig. Returning nil.", logrus.Fields{"path": c.Path()})
	return nil
}

// getConversationContext retrieves the conversation context from the user context
func getConversationContext(c *fiber.Ctx) *pxcx.ConversationContext {
	userCtx := c.UserContext()
	if userCtx == nil {
		util.LogWarning("User context is nil. Cannot retrieve conversation context.", logrus.Fields{"path": c.Path()})
		return nil
	}

	ccCtx := userCtx.Value(pxcx.ConversationContextKey)
	if ccCtx == nil {
		util.LogWarning("Conversation context not found in user context. Creating a new one.", logrus.Fields{"path": c.Path()})
		cc, _ := pxcx.GetCachedConversation(userCtx.Value(auth.UserIDKey).(string), userCtx.Value(middleware.CIDPKey).(int))
		c.SetUserContext(context.WithValue(userCtx, pxcx.ConversationContextKey, cc))
		ccCtx = userCtx.Value(pxcx.ConversationContextKey)
	}

	if cc, ok := ccCtx.(*pxcx.ConversationContext); ok {
		return cc
	}

	util.LogWarning("Conversation context is not of type *pxcx.ConversationContext. Returning nil.", logrus.Fields{"path": c.Path()})
	return nil
}

// getSessionState retrieves the session state from the user context, creating it if necessary
func getSessionState(c *fiber.Ctx) *session.SessionState {
	userCtx := c.UserContext()
	if userCtx == nil {
		util.LogWarning("User context is nil. Cannot retrieve session state.", logrus.Fields{"path": c.Path()})
		return nil
	}

	ssCtx := userCtx.Value(session.SessionCtxKey)
	if ssCtx == nil {
		util.LogWarning("Session context not found in user context. Creating a new one.", logrus.Fields{"path": c.Path()})
		setSessionState(c)
		ssCtx = userCtx.Value(session.SessionCtxKey)
	}

	if ss, ok := ssCtx.(*session.SessionState); ok {
		return ss
	}

	util.LogWarning("Session context is not of type *session.SessionState. Returning nil.", logrus.Fields{"path": c.Path()})
	return nil
}

// setSessionState initializes the session state in the user context if it doesn't exist
func setSessionState(c *fiber.Ctx) {
	userCtx := c.UserContext()
	util.LogWarning("Session context not found in user context. Creating a new one.", logrus.Fields{"path": c.Path()})
	uid := userCtx.Value(auth.UserIDKey).(string)
	cid := userCtx.Value(middleware.CIDPKey).(int)
	ss := session.GlobalStageManager.GetSessionState(uid, cid)
	c.SetUserContext(context.WithValue(userCtx, session.SessionCtxKey, ss))
}
