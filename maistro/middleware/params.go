package middleware

type paramKey string

const (
	CIDPKey   paramKey = "conversationId"
	UIDPKey   paramKey = "userId"
	MPPKey    paramKey = "modelProfileId"
	IIDKey    paramKey = "imageId"
	ParamsKey paramKey = "params"
)

type Params struct {
	ConversationID int    `json:"conversationId,omitempty"`
	UserID         string `json:"userId,omitempty"`
	ModelProfileID string `json:"modelProfileId,omitempty"`
	ImageID        string `json:"imageId,omitempty"`
}

// func SetParams(c *fiber.Ctx) error {
// 	params := Params{}

// 	// Log all route parameters for debugging
// 	util.LogInfo("All route parameters", logrus.Fields{
// 		"path":      c.Path(),
// 		"method":    c.Method(),
// 		"params":    c.AllParams(),
// 		"CIDPKey":   "conversationId",
// 		"raw_value": c.Params("conversationId"),
// 	})

// 	// Get conversation ID from path param
// 	cidParam := c.Params("conversationId")
// 	if cidParam != "" {
// 		util.LogInfo("Found conversationId param", logrus.Fields{
// 			"param": cidParam,
// 			"path":  c.Path(),
// 		})

// 		if cid, err := c.ParamsInt("conversationId"); err == nil && cid > 0 {
// 			params.ConversationID = cid
// 			util.LogInfo("Parsed conversationId", logrus.Fields{
// 				"conversationId": cid,
// 				"path":           c.Path(),
// 			})
// 		} else if err != nil {
// 			util.LogWarning("Failed to parse conversationId", logrus.Fields{
// 				"error": err.Error(),
// 				"param": cidParam,
// 				"path":  c.Path(),
// 			})
// 		}
// 	}

// 	// Get user ID from path param
// 	uid := c.Params(string(UIDPKey))
// 	if uid != "" {
// 		params.UserID = uid
// 	}

// 	// Get model profile ID from path param
// 	mpp := c.Params(string(MPPKey))
// 	if mpp != "" {
// 		params.ModelProfileID = mpp
// 	}

// 	// Get image ID from path param
// 	iid := c.Params(string(IIDKey))
// 	if iid != "" {
// 		params.ImageID = iid
// 	}

// 	// Store params in context and locals for later use
// 	ctx := c.UserContext()
// 	ctx = context.WithValue(ctx, ParamsKey, &params)
// 	c.Locals(string(ParamsKey), &params)
// 	c.SetUserContext(ctx)

// 	util.LogInfo("Params set", logrus.Fields{
// 		"params": params,
// 		"path":   c.Path(),
// 		"method": c.Method(),
// 	})

// 	return c.Next()
// }
