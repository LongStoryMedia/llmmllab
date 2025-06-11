// Package recherche provides shared functionality for web search and content extraction
package recherche

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"

	"maistro/models"
	"maistro/util"

	customsearch "google.golang.org/api/customsearch/v1"
	"google.golang.org/api/option"
)

var googleSeachApiKey string = "AIzaSyB5RNu4WR24OapJug9rbzAgsPru7gbvFTk"
var cx string = "0445a4af16a624cfe"

// PerformWebSearch performs a web search for the given query and returns URLs
func PerformWebSearch(ctx context.Context, query string, numResults int) ([]string, error) {
	svc, err := customsearch.NewService(ctx, option.WithAPIKey(googleSeachApiKey))
	if err != nil {
		util.HandleError(err)
	}

	resp, err := svc.Cse.List().Cx(cx).Q(query).Do()
	if err != nil {
		util.HandleError(err)
	}

	// Collect URLs from the response
	var urls []string

	for _, result := range resp.Items {
		urls = append(urls, result.Link)

		// Stop if we have enough results
		if len(urls) >= numResults {
			break
		}
	}

	// If we didn't get any results, fall back to Wikipedia and Britannica
	if len(urls) == 0 {
		util.LogWarning("No search results found, falling back to default sources")
		urls = append(urls,
			"https://en.wikipedia.org/wiki/"+strings.ReplaceAll(query, " ", "_"),
			"https://www.britannica.com/search?query="+strings.ReplaceAll(query, " ", "+"))
	}

	// Limit to requested number of results
	if len(urls) > numResults {
		urls = urls[:numResults]
	}

	util.LogInfo(fmt.Sprintf("Found %d search results for query: %s", len(urls), query))
	return urls, nil
}

// ExtractTextFromURL extracts text content from a URL
func ExtractTextFromURL(ctx context.Context, urlString string) (string, error) {
	util.LogInfo(fmt.Sprintf("Extracting text from URL: %s", urlString))

	client := http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, "GET", urlString, nil)
	if err != nil {
		return "", util.HandleError(fmt.Errorf("failed to create request: %w", err))
	}

	// Add a user agent to simulate a browser
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

	resp, err := client.Do(req)
	if err != nil {
		return "", util.HandleError(fmt.Errorf("failed to fetch URL: %w", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", util.HandleError(fmt.Errorf("failed with status %d", resp.StatusCode))
	}

	// Use goquery to parse the HTML
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		util.HandleError(fmt.Errorf("failed to parse HTML: %w", err))
		return "", fmt.Errorf("failed to parse HTML: %w", err)
	}

	// Extract text from relevant HTML elements
	var textBuilder strings.Builder
	doc.Find("pre, p, h1, h2, h3, h4, h5, article, section").Each(func(i int, s *goquery.Selection) {
		// Skip empty elements or elements with very little content
		text := strings.TrimSpace(s.Text())
		if len(text) > 15 { // Only include elements with meaningful content
			textBuilder.WriteString(text)
			textBuilder.WriteString("\n\n")
		}
	})

	// If no content was found with the main selectors, try a more generic approach
	if textBuilder.Len() < 100 {
		// Get all text from the body
		bodyText := strings.TrimSpace(doc.Find("body").Text())
		if len(bodyText) > 0 {
			// Clean up the text (remove excessive whitespace)
			bodyText = strings.Join(strings.Fields(bodyText), " ")
			textBuilder.WriteString(bodyText)
		}
	}

	result := textBuilder.String()
	if len(result) == 0 {
		return "", fmt.Errorf("no text content extracted")
	}

	// Truncate if too long
	maxLen := 5000
	if len(result) > maxLen {
		result = result[:maxLen] + "...(text truncated)..."
	}

	return result, nil
}

func ExtractUrlContentFromQuery(ctx context.Context, query string) (*models.SearchResult, error) {
	// Check if the query contains a URL
	if strings.Contains(query, "http://") || strings.Contains(query, "https://") {
		// Extract URLs from the query
		words := strings.Fields(query)
		var urls []string
		for _, word := range words {
			if strings.HasPrefix(word, "http://") || strings.HasPrefix(word, "https://") {
				// Basic URL validation by parsing
				if _, err := url.ParseRequestURI(word); err == nil {
					urls = append(urls, word)
				}
			}
		}

		if len(urls) > 0 {
			util.LogInfo(fmt.Sprintf("Extracted URLs from query: %v", urls))
			// Remove duplicates
			urls = slices.Compact(urls)
			results := make([]models.SearchResultContent, 0)

			for _, u := range urls {
				// Ensure each URL is valid
				if _, err := url.ParseRequestURI(u); err != nil {
					util.LogWarning(fmt.Sprintf("Invalid URL found in query: %s, error: %v", u, err))
					continue
				}
				// Normalize the URL by removing trailing slashes
				address := strings.TrimSuffix(u, "/")
				content, err := ExtractTextFromURL(ctx, address)
				if err != nil {
					util.HandleError(fmt.Errorf("failed to extract content from URL: %w", err))
					continue
				}
				results = append(results, models.SearchResultContent{
					URL:     address,
					Content: content,
				})
			}

			return &models.SearchResult{
				IsFromURLInUserQuery: true,
				Query:                "",
				Contents:             results,
			}, nil
		}
		util.LogInfo("No valid URLs found in query")
	}
	return nil, nil
}

// QuickSearch performs a complete search operation: search, extract content, and format results
func QuickSearch(ctx context.Context, query string, maxResults int, includeContents bool) (*models.SearchResult, error) {
	result := &models.SearchResult{
		Query: query,
	}

	// If no URL found, perform web search
	urls, err := PerformWebSearch(ctx, query, maxResults)
	if err != nil {
		result.Error = util.StrPtr(err.Error())
		return result, err
	}
	// Remove duplicates
	urls = slices.Compact(urls)

	// Extract content if requested
	if includeContents && len(urls) > 0 {
		result.Contents = make([]models.SearchResultContent, 0)
		for _, u := range urls {
			// Ensure each URL is valid
			if _, err := url.ParseRequestURI(u); err != nil {
				util.LogWarning(fmt.Sprintf("Invalid URL found in query: %s, error: %v", u, err))
				continue
			}
			// Normalize the URL by removing trailing slashes
			address := strings.TrimSuffix(u, "/")
			content, err := ExtractTextFromURL(ctx, address)
			if err != nil {
				// Just log the error and continue
				util.LogWarning(fmt.Sprintf("Error extracting text from URL %s: %v", address, err))
				continue
			}
			if content != "" {
				result.Contents = append(result.Contents, models.SearchResultContent{
					URL:     address,
					Content: content,
				})
			}
		}
	}

	return result, nil
}
