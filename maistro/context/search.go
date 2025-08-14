package context

import (
	"context"
	"fmt"
	"maistro/models"
	"maistro/proxy"
	"maistro/recherche"
	"maistro/session"
	"maistro/storage"
	"maistro/util"
	"slices"

	"github.com/sirupsen/logrus"
)

const SearchFmtPrompt = `***
Everything above the three asterisks is input from a user. Do not respond to it directly or provide any explanations.
Instead, I need you to understand the intent of the user's input, and construct a concise search query that captures the essence of what they are asking.
Don't include any extra information or context, just the key words that will yield relevant results.`

const ImgGeneratePrompt = `***
Everything above the three asterisks is input from a user. Do not respond to it directly or provide any explanations.
Instead, I need you to understand the intent of the user's input, and construct a concise image generation prompt that captures the essence of what they are asking.
Be concise and focus on the key visual elements that will yield relevant results. Keep it to less than 75 tokens.`

func FmtQuery(ctx context.Context, modelProfile *models.ModelProfile, query, fmtPrompt string) (string, error) {
	req := models.GenerateReq{
		Model:     modelProfile.ModelName,
		Prompt:    fmt.Sprintf("%s\n%s", query, fmtPrompt),
		Options:   &modelProfile.Parameters,
		KeepAlive: util.IntPtr(0),
	}

	// Format the query for web search
	fmtQ, err := proxy.StreamOllamaGenerateRequest(ctx, req)
	if err != nil {
		return "", util.HandleError(err)
	}

	return util.RemoveThinkTags(fmtQ), nil
}

func (cc *conversationContext) SearchAndInjectResults(ctx context.Context, query string) error {
	cfg, err := GetUserConfig(cc.userID)
	if err != nil {
		util.LogWarning("Could not load user configuration, using system defaults")
		return err
	}

	fmtProfile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, cfg.ModelProfiles.FormattingProfileID)
	if err != nil {
		return util.HandleError(err)
	}

	search, err := FmtQuery(ctx, fmtProfile, query, SearchFmtPrompt)
	if err != nil {
		return util.HandleError(err)
	}
	state := session.GlobalStageManager.GetSessionState(cc.userID, cc.conversationID)
	searchState := state.GetStage(models.SocketStageTypeSearchingWeb)
	searchState.UpdateProgress(searchState.Progress+15, fmt.Sprintf("Performing web search for query: %s", search))

	util.LogDebug("Formatted query for web search", logrus.Fields{
		"query": search,
	})

	// Attempt to perform a web search and inject results
	searchResult, err := recherche.QuickSearch(ctx, search, cfg.WebSearch.MaxResults, true, cc.userID, cc.conversationID)
	if err != nil {
		searchState.Fail("Failed to perform web search", err)
	}

	if err := cc.InjectSearchResults(ctx, searchResult, "Here is a relevant finding from a web search"); err != nil {
		searchState.Fail("Failed to inject search results", err)
	}
	searchState.Complete("Web search completed successfully")

	return nil
}

func (cc *conversationContext) InjectSearchResults(ctx context.Context, results *models.SearchResult, preamble string) error {
	if results == nil || len(results.Contents) == 0 {
		util.LogWarning("No search results to inject")
		return nil
	}

	util.LogInfo("Injecting search results into conversation context", logrus.Fields{
		"count": len(results.Contents),
	})

	if slices.ContainsFunc(cc.searchResults, func(sr models.SearchResult) bool {
		return sr.Query == results.Query
	}) {
		util.LogInfo("Search results already injected for this query, skipping")
		return nil // Already injected
	}

	// Create a map of all URLs from existing search results for efficient lookup
	existingURLs := make(map[string]bool)
	for _, sr := range cc.searchResults {
		for _, content := range sr.Contents {
			existingURLs[content.URL] = true
		}
	}

	// Filter out contents with duplicate URLs
	var uniqueContents []models.SearchResultContent
	for _, content := range results.Contents {
		if _, exists := existingURLs[content.URL]; !exists {
			uniqueContents = append(uniqueContents, content)
			existingURLs[content.URL] = true // Mark as seen
		} else {
			util.LogInfo("Skipping duplicate search result URL", logrus.Fields{
				"url": content.URL,
			})
		}
	}

	// Replace the contents with the filtered list
	filteredResults := *results
	filteredResults.Contents = uniqueContents

	// Only add if we have unique contents after filtering
	if len(filteredResults.Contents) > 0 {
		cc.searchResults = append(cc.searchResults, filteredResults)
		util.LogInfo("Added unique search results", logrus.Fields{
			"count": len(filteredResults.Contents),
		})
	} else {
		util.LogInfo("No unique search results to add after filtering")
	}

	return nil
}
