package api

import (
	"fmt"
	"maistro/context"
	"maistro/recherche"
	"maistro/util"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// RegisterWebSearchRoutes adds web search endpoints
func RegisterWebSearchRoutes(app *fiber.App) {
	app.Post("/api/websearch", PerformWebSearch)
}

// PerformWebSearch handles a web search request
func PerformWebSearch(c *fiber.Ctx) error {
	var req struct {
		Query          string `json:"query"`
		UserID         string `json:"user_id,omitempty"`
		MaxResults     *int   `json:"max_results,omitempty"`
		IncludeContent *bool  `json:"include_content,omitempty"`
	}

	if err := c.BodyParser(&req); err != nil {
		return handleError(err, fiber.StatusBadRequest, "Invalid request body")
	}

	if req.Query == "" {
		return fiber.NewError(fiber.StatusBadRequest, "Query is required")
	}

	maxResults := 3
	includeContent := true

	// Override defaults with request values if provided
	if req.MaxResults != nil {
		maxResults = *req.MaxResults
	}
	if req.IncludeContent != nil {
		includeContent = *req.IncludeContent
	}

	// Check user config if available
	if req.UserID != "" {
		cfg, err := context.GetUserConfig(req.UserID)
		if err == nil && cfg.WebSearch != nil {
			if cfg.WebSearch.MaxResults > 0 {
				maxResults = cfg.WebSearch.MaxResults
			}
			includeContent = cfg.WebSearch.IncludeResults
		}
	}

	// Cap max results at 5 to prevent abuse
	if maxResults > 5 {
		maxResults = 5
	}

	// Log the search request
	util.LogInfo("Web search request", logrus.Fields{
		"query":          req.Query,
		"maxResults":     maxResults,
		"includeContent": includeContent,
	})

	// Perform the web search
	results, err := recherche.QuickSearch(c.UserContext(), req.Query, maxResults, includeContent)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, fmt.Sprintf("Web search failed: %v", err))
	}

	return c.JSON(results)
}
