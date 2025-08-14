package research

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"maistro/config"
	pxcx "maistro/context"
	"maistro/models"
	"maistro/recherche"
	"maistro/storage"
	"maistro/util"

	"github.com/sirupsen/logrus"
)

// format is the JSON schema for the research plan
var format map[string]any = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"id": map[string]any{
			"type": "string",
		},
		"question": map[string]any{
			"type": "string",
		},
		"synthesized_answer": map[string]any{
			"type": "string",
		},
		"error_message": map[string]any{
			"type": "string",
		},
	},
}

// PerformDeepResearch is the main orchestrator for research tasks
func PerformDeepResearch(ctx context.Context, taskID int, userID, originalQuery string, conversationID *int) {
	util.LogInfo("Starting research task", logrus.Fields{
		"taskID": taskID,
		"query":  originalQuery,
	})
	updateTaskStatus(ctx, taskID, models.ResearchTaskStatusPLANNING, nil)

	// 1. Decompose & Plan
	util.LogInfo("Planning phase - Decomposing query", logrus.Fields{"taskID": taskID})
	plan, err := planResearchTask(ctx, userID, taskID, originalQuery)
	if err != nil {
		util.LogWarning("Error in planning phase", logrus.Fields{"taskID": taskID, "error": err})
		errorMsg := fmt.Sprintf("Planning failed: %v", err)
		updateTaskStatus(ctx, taskID, models.ResearchTaskStatusFAILED, &errorMsg)
		return
	}

	util.LogInfo("Plan created with sub-questions", logrus.Fields{"taskID": taskID, "subQuestions": len(plan.SubQuestions)})
	updateTaskStatus(ctx, taskID, models.ResearchTaskStatusGATHERING, nil)

	// Channel for collecting synthesized sub-answers
	subResultsChan := make(chan models.ResearchQuestionResult, len(plan.SubQuestions))
	var wg sync.WaitGroup // To wait for all sub-question processing goroutines

	// 2. Information Gathering & Initial Synthesis per Sub-Question
	for _, sq := range plan.SubQuestions {
		wg.Add(1)
		go func(subQ models.ResearchQuestion) {
			defer wg.Done()
			processSubQuestion(ctx, userID, taskID, subQ, subResultsChan)
		}(sq)
	}

	// Wait for all sub-questions to be processed
	wg.Wait()
	close(subResultsChan)

	// Collect all sub-results
	var finalSubAnswers []models.ResearchQuestionResult
	for res := range subResultsChan {
		finalSubAnswers = append(finalSubAnswers, res)

		// Store individual sub-answers in the database
		if res.ErrorMessage != nil {
			_, _, _ = storage.ResearchTaskStoreInstance.UpdateSubtaskStatus(ctx, taskID, res.ID, string(models.ResearchTaskStatusFAILED), res.ErrorMessage)
		} else {
			_, _ = storage.ResearchTaskStoreInstance.StoreSynthesizedAnswer(ctx, taskID, res.ID, *res.SynthesizedAnswer)
			_, _, _ = storage.ResearchTaskStoreInstance.UpdateSubtaskStatus(ctx, taskID, res.ID, string(models.ResearchTaskStatusCOMPLETED), nil)
		}
	}

	// 3. Consolidate & Final Report
	updateTaskStatus(ctx, taskID, models.ResearchTaskStatusSYNTHESIZING, nil)
	finalReport, err := consolidateResearchResults(ctx, userID, taskID, originalQuery, plan, finalSubAnswers)
	if err != nil {
		util.LogWarning("Error generating final report", logrus.Fields{"taskID": taskID, "error": err})
		errorMsg := fmt.Sprintf("Final report generation failed: %v", err)
		updateTaskStatus(ctx, taskID, models.ResearchTaskStatusFAILED, &errorMsg)
		return
	}

	// Store final result in database
	_, err = storage.ResearchTaskStoreInstance.StoreFinalResult(ctx, taskID, finalReport)
	if err != nil {
		util.LogWarning("Error storing final result", logrus.Fields{"taskID": taskID, "error": err})
	}

	updateTaskStatus(ctx, taskID, models.ResearchTaskStatusCOMPLETED, nil)
	util.LogInfo("Deep research completed successfully", logrus.Fields{"taskID": taskID})

	// Optional: If conversationID is provided, add the final report as an assistant message
	if conversationID != nil {
		err := addResearchResultToConversation(ctx, userID, finalReport, conversationID)
		if err != nil {
			util.LogWarning("Failed to add result to conversation", logrus.Fields{"taskID": taskID, "error": err})
		}
	}
}

// planResearchTask uses the LLM to decompose a query into sub-questions
func planResearchTask(ctx context.Context, userID string, taskID int, query string) (*models.ResearchPlan, error) {
	// Call the model for the planning step
	plan, err := CallLLMForResearchPlan(ctx, userID, query)
	if err != nil {
		util.LogWarning("Error parsing plan JSON", logrus.Fields{"taskID": taskID, "error": err})
		util.LogWarning("Raw plan JSON", logrus.Fields{"taskID": taskID, "rawPlan": plan.RawPlan})
		return nil, fmt.Errorf("plan parsing failed: %w", err)
	}

	// Create subtasks in database
	for _, sq := range plan.SubQuestions {
		_, err := storage.ResearchTaskStoreInstance.SaveSubtask(ctx, &models.ResearchSubtask{
			TaskID:     taskID,
			QuestionID: sq.ID,
			Status:     models.ResearchSubtaskStatusPENDING,
		})
		if err != nil {
			util.LogWarning("Error saving subtask", logrus.Fields{"taskID": taskID, "error": err})
			continue
		}
	}

	_, err = storage.ResearchTaskStoreInstance.StoreResearchPlan(ctx, taskID, plan)
	if err != nil {
		util.LogWarning("Error storing subtasks", logrus.Fields{"taskID": taskID, "error": err})
	}

	return plan, nil
}

// processSubQuestion handles the information gathering and synthesis for a single sub-question
func processSubQuestion(ctx context.Context, userID string, taskID int, subQ models.ResearchQuestion, resultChan chan<- models.ResearchQuestionResult) {
	util.LogInfo("Starting sub-question", logrus.Fields{
		"taskID":   taskID,
		"subQID":   subQ.ID,
		"question": subQ.Question,
	})
	updateSubtaskStatus(ctx, taskID, subQ.ID, models.ResearchTaskStatusGATHERING, nil)

	var allExtractedTexts []string
	var allSources []string

	// Gather information from web sources using the keywords
	for _, keyword := range subQ.Keywords {
		// Use shared recherche package for web search
		searchResults, err := recherche.PerformWebSearch(ctx, keyword, 3)
		if err != nil {
			util.LogWarning("Error searching for keyword", logrus.Fields{"taskID": taskID, "subQID": subQ.ID, "keyword": keyword, "error": err})
			continue
		}

		for _, resultURL := range searchResults {
			// Extract content from URLs using shared function
			textContent, err := recherche.ExtractTextFromURL(ctx, resultURL)
			if err != nil {
				util.LogWarning("Error extracting from URL", logrus.Fields{"taskID": taskID, "subQID": subQ.ID, "url": resultURL, "error": err})
				continue
			}

			// Add the content and source if successful
			if len(textContent) > 0 {
				allExtractedTexts = append(allExtractedTexts, textContent)
				allSources = append(allSources, resultURL)
				util.LogInfo("Added content from URL", logrus.Fields{
					"taskID": taskID,
					"subQID": subQ.ID,
					"url":    resultURL,
					"length": len(textContent),
				})

				// Limit the number of sources per question
				if len(allExtractedTexts) >= 5 {
					break
				}
			}
		}

		// If we have enough sources, stop searching with other keywords
		if len(allExtractedTexts) >= 5 {
			break
		}
	}

	// Store gathered information
	_, err := storage.ResearchTaskStoreInstance.StoreGatheredInfo(ctx, taskID, subQ.ID, allExtractedTexts, allSources)
	if err != nil {
		util.LogWarning("Error storing gathered info", logrus.Fields{"taskID": taskID, "subQID": subQ.ID, "error": err})
	}

	// Check if we gathered any useful information
	if len(allExtractedTexts) == 0 {
		util.LogWarning("No text gathered", logrus.Fields{
			"taskID": taskID,
			"subQID": subQ.ID,
		})
		errorMsg := "No information found for this sub-question"
		updateSubtaskStatus(ctx, taskID, subQ.ID, models.ResearchTaskStatusFAILED, &errorMsg)
		resultChan <- models.ResearchQuestionResult{
			ID:                subQ.ID,
			Question:          subQ.Question,
			SynthesizedAnswer: util.StrPtr("No information found for this sub-question."),
			ErrorMessage:      util.StrPtr("No information found for this sub-question."),
		}
		return
	}

	// Synthesize gathered information
	updateSubtaskStatus(ctx, taskID, subQ.ID, models.ResearchTaskStatusPROCESSING, nil)

	// Combine texts with source citations
	var combinedText strings.Builder
	for i, text := range allExtractedTexts {
		combinedText.WriteString(fmt.Sprintf("Source %d (%s):\n%s\n\n", i+1, allSources[i], text))

		// Break if the text is getting too large for the context window
		if combinedText.Len() > 12000 {
			combinedText.WriteString("...(truncated due to length)...")
			break
		}
	}

	util.LogInfo("Synthesizing information from sources", logrus.Fields{
		"taskID":  taskID,
		"subQID":  subQ.ID,
		"sources": len(allExtractedTexts),
	})
	result, err := CallLLMForSubResult(ctx, userID, combinedText.String(), nil)
	if err != nil {
		errorMsg := fmt.Sprintf("Error synthesizing: %v", err)
		util.LogWarning("Error synthesizing", logrus.Fields{
			"taskID": taskID,
			"subQID": subQ.ID,
			"error":  errorMsg,
		})
		updateSubtaskStatus(ctx, taskID, subQ.ID, models.ResearchTaskStatusFAILED, &errorMsg)
		result.ErrorMessage = &errorMsg
		resultChan <- *result
		return
	}

	util.LogInfo("Synthesis complete", logrus.Fields{
		"taskID": taskID,
		"subQID": subQ.ID,
		"length": len(*result.SynthesizedAnswer),
	})
	resultChan <- models.ResearchQuestionResult{
		ID:                subQ.ID,
		Question:          subQ.Question,
		SynthesizedAnswer: result.SynthesizedAnswer,
	}
}

// consolidateResearchResults creates a final coherent report from all sub-question results
func consolidateResearchResults(ctx context.Context, userID string, taskID int, originalQuery string, plan *models.ResearchPlan, subResults []models.ResearchQuestionResult) (*models.ResearchQuestionResult, error) {
	util.LogInfo("Consolidating results from sub-questions", logrus.Fields{
		"taskID":       taskID,
		"subQuestions": len(subResults),
	})

	var consolidationInput strings.Builder
	consolidationInput.WriteString(fmt.Sprintf("Original User Research Request: %s\n\n", originalQuery))
	consolidationInput.WriteString("Main Intent: " + plan.MainIntent + "\n\n")
	consolidationInput.WriteString("Synthesized Findings for Sub-Questions:\n\n")

	for _, result := range subResults {
		if result.ErrorMessage != nil {
			consolidationInput.WriteString(fmt.Sprintf("### Question %d: %s\n\nError: %s\n\n",
				result.ID, result.Question, *result.ErrorMessage))
		} else {
			consolidationInput.WriteString(fmt.Sprintf("### Question %d: %s\n\n%s\n\n",
				result.ID, result.Question, *result.SynthesizedAnswer))
		}
	}

	finalReport, err := CallLLMForResult(ctx, userID, consolidationInput.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("final report generation failed: %w", err)
	}

	return finalReport, nil
}

// CallLLMForResult calls the LLM with the given prompt for research steps
func CallLLMForResult(ctx context.Context, userID, consolidatedInput string, userMessages []models.Message) (*models.ResearchQuestionResult, error) {
	var ollamaMessages []models.Message

	cfg, err := pxcx.GetUserConfig(userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user config: %w", err)
	}

	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, cfg.ModelProfiles.ResearchConsolidationProfileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get model profile: %w", err)
	}
	formattedText := fmt.Sprintf("%s \n\n%s", profile.SystemPrompt, consolidatedInput)
	ollamaMessages = append(ollamaMessages, models.Message{
		Role:    "system",
		Content: []models.MessageContent{{Type: models.MessageContentTypeText, Text: util.StrPtr(formattedText)}},
	})

	// Add any additional user messages
	ollamaMessages = append(ollamaMessages, userMessages...)

	// Return the assistant's message content
	return doResearch[models.ResearchQuestionResult](ctx, profile, ollamaMessages)
}

// CallLLMForResult calls the LLM with the given prompt for research steps
func CallLLMForSubResult(ctx context.Context, userID, consolidatedInput string, userMessages []models.Message) (*models.ResearchQuestionResult, error) {
	var ollamaMessages []models.Message

	cfg, err := pxcx.GetUserConfig(userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user config: %w", err)
	}

	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, cfg.ModelProfiles.ResearchAnalysisProfileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get model profile: %w", err)
	}
	formattedText := fmt.Sprintf("%s \n\n%s", profile.SystemPrompt, consolidatedInput)
	ollamaMessages = append(ollamaMessages, models.Message{
		Role:    "system",
		Content: []models.MessageContent{{Type: models.MessageContentTypeText, Text: util.StrPtr(formattedText)}},
	})

	// Add any additional user messages
	ollamaMessages = append(ollamaMessages, userMessages...)

	// Return the assistant's message content
	return doResearch[models.ResearchQuestionResult](ctx, profile, ollamaMessages)
}

// CallLLMForResearchPlan calls the LLM with the given prompt for research steps
func CallLLMForResearchPlan(ctx context.Context, userID, query string) (*models.ResearchPlan, error) {
	var ollamaMessages []models.Message

	cfg, err := pxcx.GetUserConfig(userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user config: %w", err)
	}

	systemPrompt := fmt.Sprintf("%s User Query: %s", cfg.ModelProfiles.ResearchPlanProfileID, query)
	ollamaMessages = append(ollamaMessages, models.Message{
		Role:    "system",
		Content: []models.MessageContent{{Type: models.MessageContentTypeText, Text: util.StrPtr(systemPrompt)}},
	})

	profile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, cfg.ModelProfiles.ResearchPlanProfileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get model profile: %w", err)
	}

	return doResearch[models.ResearchPlan](ctx, profile, ollamaMessages)
}

func doResearch[T any](ctx context.Context, profile *models.ModelProfile, ollamaMessages []models.Message) (*T, error) {
	// Create a non-streaming request to get the full response at once
	ollamaReq := models.ChatReq{
		Model:    profile.ModelName,
		Messages: ollamaMessages,
		Format:   format,
		Stream:   false, // We want the complete response at once
	}

	// Convert to JSON
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Get Ollama URL from config
	conf := config.GetConfig(nil)
	url := conf.InferenceServices.Ollama.BaseURL + "/api/chat"

	// Send request to Ollama
	resp, err := http.Post(url, "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Ollama: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse response
	var ollamaResp T
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	// Return the assistant's message content
	return &ollamaResp, nil
}

// Helper functions

// updateTaskStatus updates the status of a research task in the database
func updateTaskStatus(ctx context.Context, taskID int, status models.ResearchTaskStatus, errorMsg *string) {
	_, err := storage.ResearchTaskStoreInstance.UpdateTaskStatus(ctx, taskID, string(status), errorMsg)
	if err != nil {
		util.LogWarning("Error updating task status", logrus.Fields{"taskID": taskID, "error": err})
	}
}

// updateSubtaskStatus updates the status of a research subtask in the database
func updateSubtaskStatus(ctx context.Context, taskID int, questionID int, status models.ResearchTaskStatus, errorMsg *string) {
	_, _, err := storage.ResearchTaskStoreInstance.UpdateSubtaskStatus(ctx, taskID, questionID, string(status), errorMsg)
	if err != nil {
		util.LogWarning("Error updating subtask status", logrus.Fields{"taskID": taskID, "subQID": questionID, "error": err})
	}
}

// extractJSON tries to extract a JSON object from a string that might contain additional text
func extractJSON(input string) string {
	// Find the start and end of a JSON object
	start := strings.Index(input, "{")
	end := strings.LastIndex(input, "}")

	if start >= 0 && end > start {
		return input[start : end+1]
	}

	return input // Return original if no JSON found
}

// addResearchResultToConversation adds the research result to a conversation as an assistant message
func addResearchResultToConversation(ctx context.Context, userID string, finalResult *models.ResearchQuestionResult, conversationID *int) error {
	if conversationID == nil {
		return fmt.Errorf("no conversation ID provided")
	}

	// Get the conversation context
	convCtx, err := pxcx.GetOrCreateConversation(ctx, userID, conversationID)
	if err != nil {
		return fmt.Errorf("failed to get conversation context: %w", err)
	}

	// Add the research result as an assistant message
	message := "Research Results:\n\n" + *finalResult.SynthesizedAnswer

	asstMsg := models.Message{
		Role:    models.MessageRoleAssistant,
		Content: []models.MessageContent{{Type: models.MessageContentTypeText, Text: util.StrPtr(message)}},
	}

	if _, err := convCtx.AddAssistantMessage(ctx, &asstMsg); err != nil {
		return fmt.Errorf("failed to add assistant message: %w", err)
	}

	return nil
}
