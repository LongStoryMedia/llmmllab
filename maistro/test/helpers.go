package test

import (
	"encoding/csv"
	"maistro/models"
	"os"
	"strconv"
	"time"
)

func ParseMessagesCSV(filePath string) ([]models.Message, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var messages []models.Message
	for i, record := range records {
		if i == 0 { // Skip header row
			continue
		}

		id, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, err
		}

		conversationID, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, err
		}

		createdAt, err := time.Parse("2006-01-02 15:04:05.999999-07", record[4])
		if err != nil {
			return nil, err
		}

		message := models.Message{
			ID:             id,
			ConversationID: conversationID,
			Role:           record[2],
			Content:        record[3],
			CreatedAt:      &createdAt,
		}
		messages = append(messages, message)
	}

	return messages, nil
}
