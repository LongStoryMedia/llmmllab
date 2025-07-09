package context

import (
	"context"
	"fmt"
	"maistro/models"
	"maistro/storage"

	"maistro/util"

	"github.com/sirupsen/logrus"
)

// TODO: Work in dynamic model profile functionality for summaries

// SummarizeMessages creates a summary of the oldest messages
func (cc *conversationContext) SummarizeMessages(ctx context.Context) (*models.Summary, error) {
	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		return &models.Summary{}, util.HandleError(fmt.Errorf("failed to get user config: %w", err))
	}

	if !usrCfg.Summarization.Enabled {
		return &models.Summary{}, nil // Summarization is disabled
	}

	// Find messages that haven't been summarized yet
	unsummarizedMessages, messageIDs := cc.getUnsummarizedMessages(ctx)

	// Only summarize when we have enough unsummarized messages
	if len(unsummarizedMessages) < usrCfg.Summarization.MessagesBeforeSummary {
		return &models.Summary{}, nil // Not enough unsummarized messages to summarize
	}

	// Summarize the oldest N messages
	messagesToSummarize := unsummarizedMessages[:usrCfg.Summarization.MessagesBeforeSummary]

	// Extract the oldest N/2 unsummarized messages to summarize
	var messagesToSummarizeContent []models.Message
	var messageIDsToSummarize []int

	for i := range messagesToSummarize {
		messagesToSummarizeContent = append(messagesToSummarizeContent, unsummarizedMessages[i])
		messageIDsToSummarize = append(messageIDsToSummarize, messageIDs[i])
	}

	summaryProfile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.SummarizationProfileID)
	if err != nil {
		return &models.Summary{}, util.HandleError(fmt.Errorf("failed to get summarization profile: %w", err))
	}

	// Generate the summary using the selected prompt
	summaryContent, err := cc.generateSummarization(messagesToSummarizeContent, summaryProfile)
	if err != nil {
		return &models.Summary{}, util.HandleError(fmt.Errorf("failed to generate summary: %w", err))
	}

	// Create a new summary record in the database
	summaryID, err := storage.SummaryStoreInstance.CreateSummary(ctx, cc.conversationID, summaryContent, 1, messageIDsToSummarize)
	if err != nil {
		return &models.Summary{}, util.HandleError(fmt.Errorf("failed to store summary: %w", err))
	}

	// Add the summary to our context
	summary := models.Summary{
		Content:        summaryContent,
		Level:          1,
		ID:             summaryID,
		SourceIds:      messageIDsToSummarize,
		ConversationID: cc.conversationID,
	}
	cc.summaries = append(cc.summaries, summary)

	if _, err := cc.createSummaryMemory(ctx, summary, usrCfg); err != nil {
		util.LogWarning("Failed to create summary memory", logrus.Fields{
			"error": err,
		})
	}

	// Remove the summarized messages from our in-memory context
	cc.removeMessagesById(messageIDsToSummarize)

	// Check if we need to consolidate summaries at level 1
	if cc.shouldConsolidateLevel(1) {
		if err := cc.consolidateLevel(ctx, 1); err != nil {
			util.LogWarning("Failed to consolidate level 1 summaries", logrus.Fields{
				"error": err,
			})
		}
	}

	return &summary, nil
}

// getUnsummarizedMessages returns messages that haven't been summarized yet
func (cc *conversationContext) getUnsummarizedMessages(ctx context.Context) ([]models.Message, []int) {
	// Create a set to track which message IDs have already been summarized
	summarizedMessageIDs := make(map[int]bool)

	// Get all summaries to determine which messages have already been summarized
	summaries, err := storage.SummaryStoreInstance.GetSummariesForConversation(ctx, cc.conversationID)
	if err == nil {
		// Collect all message IDs that have already been included in summaries
		for _, summary := range summaries {
			for _, msgID := range summary.SourceIds {
				summarizedMessageIDs[msgID] = true
			}
		}
	}

	// Filter messages that haven't been summarized yet
	var unsummarizedMessages []models.Message
	var unsummarizedIDs []int
	for _, msg := range cc.messages {
		if !summarizedMessageIDs[msg.ID] && msg.ID > 0 {
			// Only include messages that have a valid ID and haven't been summarized
			unsummarizedMessages = append(unsummarizedMessages, msg)
			unsummarizedIDs = append(unsummarizedIDs, msg.ID)
		}
	}

	return unsummarizedMessages, unsummarizedIDs
}

// removeMessagesById removes messages with specific IDs from the conversation
func (cc *conversationContext) removeMessagesById(ids []int) {
	var remainingMessages []models.Message
	for _, msg := range cc.messages {
		shouldKeep := true
		for _, id := range ids {
			if msg.ID == id {
				shouldKeep = false
				break
			}
		}
		if shouldKeep {
			remainingMessages = append(remainingMessages, msg)
		}
	}
	cc.messages = remainingMessages
}

// shouldConsolidateLevel checks if we have exactly X summaries at a specific level
func (cc *conversationContext) shouldConsolidateLevel(level int) bool {
	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		util.LogWarning("Failed to get user config", logrus.Fields{
			"error": err,
		})
		return false
	}
	levelCount := countSummariesAtLevel(cc.summaries, level)

	// We only consolidate when we have exactly X summaries at this level
	return levelCount == usrCfg.Summarization.SummariesBeforeConsolidation
}

// consolidateLevel creates a summary of summaries at a specific level
func (cc *conversationContext) consolidateLevel(ctx context.Context, level int) error {
	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		return fmt.Errorf("failed to get user config: %w", err)
	}

	// Get summaries for this level
	var summariesToConsolidate []models.Summary
	var summaryIDs []int

	// Filter summaries by level
	for _, summary := range cc.summaries {
		if summary.Level == level {
			summariesToConsolidate = append(summariesToConsolidate, summary)
			summaryIDs = append(summaryIDs, summary.ID)
		}
	}

	// We only consolidate when we have EXACTLY X summaries
	if len(summariesToConsolidate) != usrCfg.Summarization.SummariesBeforeConsolidation {
		return nil // Not exactly the right number of summaries to consolidate
	}

	util.LogInfo("Consolidating summaries", map[string]interface{}{
		"count": len(summariesToConsolidate),
		"level": level,
	})

	// Convert summaries to messages for the summary generator
	var messagesToSummarize []models.Message
	for _, summary := range summariesToConsolidate {
		messagesToSummarize = append(messagesToSummarize, models.Message{
			Role:    "system",
			Content: summary.Content,
			ID:      summary.ID,
		})
	}

	var profile *models.ModelProfile

	// Generate the next level summary
	nextLevel := level + 1

	// Determine the appropriate prompt type for consolidation
	promptType := "standard"

	// For higher-level summaries (level 2+), focus more on key points and structure
	if level >= 2 {
		profile, err = storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.KeyPointsProfileID)
		if err != nil {
			return fmt.Errorf("failed to get master summary profile: %w", err)
		}
	} else {
		profile, err = storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.SummarizationProfileID)
		if err != nil {
			return fmt.Errorf("failed to get summary profile: %w", err)
		}
	}

	summaryContent, err := cc.generateSummarization(messagesToSummarize, profile)
	if err != nil {
		return fmt.Errorf("failed to generate level %d summary: %w", nextLevel, err)
	}

	// Store the new summary in the database
	summaryID, err := storage.SummaryStoreInstance.CreateSummary(ctx, cc.conversationID, summaryContent, nextLevel, summaryIDs)
	if err != nil {
		return fmt.Errorf("failed to store level %d summary: %w", nextLevel, err)
	}

	util.LogInfo("Created new summary", map[string]interface{}{
		"nextLevel":  nextLevel,
		"summaryId":  summaryID,
		"count":      len(summariesToConsolidate),
		"level":      level,
		"promptType": promptType,
	})

	// Add the new summary to our context
	newSummary := models.Summary{
		Content: summaryContent,
		Level:   nextLevel,
		ID:      summaryID,
	}
	cc.summaries = append(cc.summaries, newSummary)

	// Remove consolidated summaries from our in-memory context
	cc.removeSummariesByIdAndLevel(summaryIDs, level)

	// Check if we need to create/update a master summary
	maxSummaryLevel := usrCfg.Summarization.MaxSummaryLevels
	if nextLevel == maxSummaryLevel {
		levelMaxCount := countSummariesAtLevel(cc.summaries, maxSummaryLevel)

		// If we have exactly X summaries at max level, create/update master summary
		if levelMaxCount == usrCfg.Summarization.SummariesBeforeConsolidation {
			util.LogInfo("Creating/updating master summary", logrus.Fields{
				"count": levelMaxCount,
				"level": maxSummaryLevel,
			})

			if cc.masterSummary == nil {
				if err := cc.createMasterSummary(ctx); err != nil {
					util.LogWarning("Failed to create master summary", logrus.Fields{
						"error": err,
					})
				}
			} else {
				// Update the existing master summary
				if err := cc.updateMasterSummary(ctx); err != nil {
					util.LogWarning("Failed to update master summary", logrus.Fields{
						"error": err,
					})
				}
			}
		} else {
			util.LogInfo("Not enough summaries to create master summary yet", logrus.Fields{
				"currentCount": levelMaxCount,
				"targetCount":  usrCfg.Summarization.SummariesBeforeConsolidation,
				"level":        maxSummaryLevel,
			})
		}
	}

	return nil
}

// removeSummariesByIdAndLevel removes summaries with specific IDs and level from the conversation
func (cc *conversationContext) removeSummariesByIdAndLevel(ids []int, level int) {
	var updatedSummaries []models.Summary
	for _, s := range cc.summaries {
		shouldKeep := true
		for _, id := range ids {
			if s.ID == id && s.Level == level {
				shouldKeep = false
				break
			}
		}
		if shouldKeep {
			updatedSummaries = append(updatedSummaries, s)
		}
	}
	cc.summaries = updatedSummaries
}

// createMasterSummary generates a weighted summary of all summaries
func (cc *conversationContext) createMasterSummary(ctx context.Context) error {
	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		return fmt.Errorf("failed to get user config: %w", err)
	}

	masterSummaryProfile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.MasterSummaryProfileID)
	if err != nil {
		return fmt.Errorf("failed to get master summary profile: %w", err)
	}
	// Create messages for the master summary
	// Add summaries from each level with appropriate weighting
	messagesToSummarize := cc.prepareMasterSummaryMessages(usrCfg, *masterSummaryProfile)

	// Generate the master summary
	masterSummaryContent, err := cc.generateSummarization(messagesToSummarize, masterSummaryProfile)
	if err != nil {
		return fmt.Errorf("failed to generate master summary: %w", err)
	}

	// Get IDs of all summaries
	var summaryIDs []int
	for _, summary := range cc.summaries {
		summaryIDs = append(summaryIDs, summary.ID)
	}

	// Store the master summary with a special level (0 for master)
	masterLevel := 0 // Special level for master summary
	masterSummaryID, err := storage.SummaryStoreInstance.CreateSummary(ctx, cc.conversationID, masterSummaryContent, masterLevel, summaryIDs)
	if err != nil {
		return fmt.Errorf("failed to store master summary: %w", err)
	}

	util.LogInfo("Created master summary", logrus.Fields{
		"summaryId": masterSummaryID,
		"count":     len(summaryIDs),
	})

	// Store the master summary in our context
	cc.masterSummary = &models.Summary{
		Content: masterSummaryContent,
		Level:   masterLevel,
		ID:      masterSummaryID,
	}

	return nil
}

// updateMasterSummary updates the existing master summary with new information
func (cc *conversationContext) updateMasterSummary(ctx context.Context) error {
	if cc.masterSummary == nil {
		return cc.createMasterSummary(ctx)
	}

	// Create messages for updating the master summary
	var messagesToSummarize []models.Message

	// Start with the existing master summary as context
	messagesToSummarize = append(messagesToSummarize, models.Message{
		Role:    "system",
		Content: fmt.Sprintf("Current master summary: %s", cc.masterSummary.Content),
		ID:      cc.masterSummary.ID,
	})

	// Find new summaries that weren't part of the original master summary
	newSummaryMessages := cc.getNewSummariesForMasterUpdate(ctx)

	// Skip if no new summaries to integrate
	if len(newSummaryMessages) == 0 {
		util.LogInfo("No new summaries to integrate into master summary", logrus.Fields{})
		return nil
	}

	// Add new summaries to the messages list
	messagesToSummarize = append(messagesToSummarize, newSummaryMessages...)

	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		return fmt.Errorf("failed to get user config: %w", err)
	}

	masterSummaryProfile, err := storage.ModelProfileStoreInstance.GetModelProfile(ctx, usrCfg.ModelProfiles.MasterSummaryProfileID)
	if err != nil {
		return fmt.Errorf("failed to get master summary profile: %w", err)
	}

	masterSummaryContent, err := cc.generateSummarization(messagesToSummarize, masterSummaryProfile)
	if err != nil {
		return fmt.Errorf("failed to update master summary: %w", err)
	}

	// Get IDs of all summaries
	var summaryIDs []int
	for _, summary := range cc.summaries {
		summaryIDs = append(summaryIDs, summary.ID)
	}

	// Add the old master summary ID
	summaryIDs = append(summaryIDs, cc.masterSummary.ID)

	// Store the updated master summary
	masterLevel := 0 // Special level for master summary
	masterSummaryID, err := storage.SummaryStoreInstance.CreateSummary(ctx, cc.conversationID, masterSummaryContent, masterLevel, summaryIDs)
	if err != nil {
		return fmt.Errorf("failed to store updated master summary: %w", err)
	}

	util.LogInfo("Updated master summary", logrus.Fields{
		"summaryId": masterSummaryID,
		"profile":   masterSummaryProfile.Name,
	})

	// Update the master summary in our context
	cc.masterSummary = &models.Summary{
		Content: masterSummaryContent,
		Level:   masterLevel,
		ID:      masterSummaryID,
	}

	return nil
}

// prepareMasterSummaryMessages creates weighted messages for the master summary
func (cc *conversationContext) prepareMasterSummaryMessages(cfg *models.UserConfig, masterProfile models.ModelProfile) []models.Message {
	var messagesToSummarize []models.Message

	// Start with a system message describing the importance of weighting
	messagesToSummarize = append(messagesToSummarize, models.Message{
		Role:    "system",
		Content: masterProfile.SystemPrompt,
	})

	// Group summaries by level
	summariesByLevel := groupSummariesByLevel(cc.summaries)

	// Find the max level
	maxLevel := findMaxLevel(cc.summaries)

	// Add summaries from each level, with decreasing importance for higher levels
	for level := 1; level <= maxLevel; level++ {
		if summaries, ok := summariesByLevel[level]; ok {
			// Calculate weight for this level
			weight := 1.0
			for i := 1; i < level; i++ {
				weight *= float64(cfg.Summarization.SummaryWeightCoefficient)
			}

			// Add summaries with level information and weight
			for _, summary := range summaries {
				weightedPrompt := fmt.Sprintf("Level %d summary (importance weight: %.2f): %s",
					level, weight, summary.Content)

				messagesToSummarize = append(messagesToSummarize, models.Message{
					Role:    "system",
					Content: weightedPrompt,
					ID:      summary.ID,
				})
			}
		}
	}

	return messagesToSummarize
}

// getNewSummariesForMasterUpdate returns messages for summaries that aren't in the master summary
func (cc *conversationContext) getNewSummariesForMasterUpdate(ctx context.Context) []models.Message {
	usrCfg, err := GetUserConfig(cc.userID)
	if err != nil {
		util.LogWarning("Failed to get user config", logrus.Fields{
			"error": err,
		})
		return nil
	}

	// Get summary IDs already included in the master summary
	masterSummaryIDs := make(map[int]bool)
	masterSummary, err := storage.SummaryStoreInstance.GetSummary(ctx, cc.masterSummary.ID)
	if err == nil {
		for _, id := range masterSummary.SourceIds {
			masterSummaryIDs[id] = true
		}
	}

	// Group summaries by level that aren't already part of the master summary
	summariesByLevel := make(map[int][]models.Summary)
	for _, summary := range cc.summaries {
		if !masterSummaryIDs[summary.ID] {
			summariesByLevel[summary.Level] = append(summariesByLevel[summary.Level], summary)
		}
	}

	// Find the max level
	maxLevel := 0
	for level := range summariesByLevel {
		if level > maxLevel {
			maxLevel = level
		}
	}

	var newSummaryMessages []models.Message

	// Add new summaries with appropriate weighting
	for level := 1; level <= maxLevel; level++ {
		if summaries, ok := summariesByLevel[level]; ok {
			// Calculate weight for this level
			weight := 1.0
			for i := 1; i < level; i++ {
				weight *= float64(usrCfg.Summarization.SummaryWeightCoefficient)
			}

			// Add summaries with level information and weight
			for _, summary := range summaries {
				weightedPrompt := fmt.Sprintf("New level %d summary (importance weight: %.2f): %s",
					level, weight, summary.Content)

				newSummaryMessages = append(newSummaryMessages, models.Message{
					Role:    "system",
					Content: weightedPrompt,
					ID:      summary.ID,
				})
			}
		}
	}

	return newSummaryMessages
}

// groupSummariesByLevel organizes summaries into a map keyed by level
func groupSummariesByLevel(summaries []models.Summary) map[int][]models.Summary {
	result := make(map[int][]models.Summary)
	for _, s := range summaries {
		result[s.Level] = append(result[s.Level], s)
	}
	return result
}

// countSummariesAtLevel counts how many summaries exist at a specific level
func countSummariesAtLevel(summaries []models.Summary, level int) int {
	count := 0
	for _, s := range summaries {
		if s.Level == level {
			count++
		}
	}
	return count
}

// findMaxLevel returns the highest summary level in a slice of summaries
func findMaxLevel(summaries []models.Summary) int {
	max := 0
	for _, s := range summaries {
		if s.Level > max {
			max = s.Level
		}
	}
	return max
}
