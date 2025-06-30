package storage

import (
	"context"
	"fmt"
	"maistro/models"
	"maistro/util"
	"math"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/sirupsen/logrus"
)

type memoryStore struct{}

// InitMemorySchema initializes the database schema for memory storage
func (md *memoryStore) InitMemorySchema(ctx context.Context) error {
	util.LogInfo("Initializing memory schema...")

	// Create memories table and hypertable
	_, err := Pool.Exec(ctx, GetQuery("memory.init_memory_schema"))
	if err != nil {
		return fmt.Errorf("failed to create memories table: %w", err)
	}
	util.LogInfo("Created memories table")

	// Create indexes for memories
	_, err = Pool.Exec(ctx, GetQuery("memory.create_memory_indexes"))
	if err != nil {
		return fmt.Errorf("failed to create memory indexes: %w", err)
	}
	util.LogInfo("Created memory indexes")

	// Enable compression on memories
	_, err = Pool.Exec(ctx, GetQuery("memory.enable_memories_compression"))
	if err != nil {
		return fmt.Errorf("failed to enable memories compression: %w", err)
	}
	util.LogInfo("Enabled memories compression")

	// Add compression policy for memories
	_, err = Pool.Exec(ctx, GetQuery("memory.memories_compression_policy"))
	if err != nil {
		util.LogWarning("Failed to add memories compression policy", logrus.Fields{"error": err})
	}
	util.LogInfo("Added memories compression policy")

	// Add retention policy for memories
	_, err = Pool.Exec(ctx, GetQuery("memory.memories_retention_policy"))
	if err != nil {
		util.LogWarning("Failed to add memories retention policy", logrus.Fields{"error": err})
	}
	util.LogInfo("Added memories retention policy")

	_, err = Pool.Exec(ctx, GetQuery("memory.create_memory_cascade_delete_triggers"))
	if err != nil {
		util.LogWarning("Failed to create memory cascade delete trigger", logrus.Fields{"error": err})
	} else {
		util.LogInfo("Memory cascade delete trigger created successfully")
	}

	util.LogInfo("Memory schema initialized successfully")
	return nil
}

// StoreMemory stores a memory with embedding for a user
func (md *memoryStore) StoreMemory(ctx context.Context, userID, source, role string, sourceID int, embeddings [][]float32) error {
	tx, err := Pool.Begin(ctx)
	if err != nil {
		return util.HandleError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	if err := md.StoreMemoryWithTx(ctx, userID, source, role, sourceID, embeddings, tx); err != nil {
		tx.Rollback(ctx) // rollback on error
		return util.HandleError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return util.HandleError(fmt.Errorf("failed to commit transaction: %w", err))
	}
	return nil
}

// StoreMemoryWithTx stores a memory with embedding for a user within a transaction
func (md *memoryStore) StoreMemoryWithTx(ctx context.Context, userID, source, role string, sourceID int, embeddings [][]float32, tx pgx.Tx) error {
	for _, embedding := range embeddings {
		pe, _ := processEmbedding(embedding)
		embeddingStr := formatEmbeddingForPgVector(pe)
		_, err := tx.Exec(ctx, GetQuery("memory.store_memory"), userID, sourceID, source, embeddingStr, role)
		if err != nil {
			util.HandleError(fmt.Errorf("failed to store memory: %w", err))
		}
	}

	return nil
}

// DeleteMemory deletes a memory by ID
func (md *memoryStore) DeleteMemory(ctx context.Context, id, userID string) error {
	_, err := Pool.Exec(ctx, GetQuery("memory.delete_memory"), id, userID)

	if err != nil {
		return fmt.Errorf("failed to delete memory: %w", err)
	}

	return nil
}

// DeleteAllUserMemories deletes all memories for a user
func (md *memoryStore) DeleteAllUserMemories(ctx context.Context, userID string) error {
	_, err := Pool.Exec(ctx, GetQuery("memory.delete_all_user_memories"), userID)

	if err != nil {
		return fmt.Errorf("failed to delete all user memories: %w", err)
	}

	return nil
}

// SearchSimilarity searches for semantically similar messages across all conversations
func (md *memoryStore) SearchSimilarity(ctx context.Context, embeddings [][]float32, minSimilarity float32, limit int, userID *string, conversationID *int, startDate, endDate *time.Time) ([]models.Memory, error) {
	memories := make([]models.Memory, 0)
	if len(embeddings) == 0 {
		return memories, nil // No embeddings to search
	}

	for _, embedding := range embeddings {
		if len(embedding) == 0 {
			return nil, util.HandleError(fmt.Errorf("embedding vector is empty"))
		}
		rows, err := Pool.Query(ctx, GetQuery("memory.search"),
			formatEmbeddingForPgVector(embedding),
			minSimilarity,
			limit,
			userID,
			conversationID,
			startDate,
			endDate)
		if err != nil {
			return nil, util.HandleError(err)
		}
		defer rows.Close()

		var currentMem models.Memory
		var lastPairKey string = "" // Empty string as sentinel value

		for rows.Next() {
			var frag models.MemoryFragment
			var newMem models.Memory

			if err := rows.Scan(&frag.Role, &newMem.SourceID, &frag.Content, &newMem.Source, &newMem.Similarity, &newMem.ConversationID, &newMem.CreatedAt); err != nil {
				return nil, util.HandleError(err)
			}
			frag.ID = newMem.SourceID

			// Generate a pair key for this row similar to the one in SQL
			var pairKey string
			if newMem.Source == "summary" {
				pairKey = fmt.Sprintf("summary-%d", newMem.SourceID)
			} else {
				// For message pairs, the SQL query should have arranged them so user messages come first
				// If this is the first message of a pair, we'll create a new Memory
				// If it's the second message, we'll add to the current Memory
				if frag.Role == "user" {
					pairKey = fmt.Sprintf("pair-%d", newMem.SourceID)
				} else {
					// For assistant messages, use the same key as the preceding user message
					pairKey = lastPairKey
				}
			}

			// Start a new memory if this is a new pair or a summary
			if pairKey != lastPairKey || newMem.Source == "summary" {
				// Save the current memory if it exists
				if len(currentMem.Fragments) > 0 {
					memories = append(memories, currentMem)
				}

				// Create a new memory
				currentMem = models.Memory{
					SourceID:       newMem.SourceID,
					Source:         newMem.Source,
					Similarity:     newMem.Similarity,
					ConversationID: newMem.ConversationID,
					CreatedAt:      newMem.CreatedAt,
					Fragments:      []models.MemoryFragment{},
				}
			}

			// Add this fragment to the current memory
			currentMem.Fragments = append(currentMem.Fragments, frag)

			// For summaries, save immediately
			if newMem.Source == "summary" {
				memories = append(memories, currentMem)
				currentMem = models.Memory{}
				lastPairKey = ""
			} else {
				// Update the lastPairKey
				lastPairKey = pairKey

				// If we have a complete pair (2 fragments), save it
				if len(currentMem.Fragments) == 2 {
					memories = append(memories, currentMem)
					currentMem = models.Memory{}
					lastPairKey = ""
				}
			}
		}

		// Add any remaining memory (should only happen if query returns incomplete results)
		if len(currentMem.Fragments) > 0 {
			memories = append(memories, currentMem)
		}
	}

	return memories, nil
}

// Vector processing utilities (migrated from embedding.go)
// formatEmbeddingForPgVector converts a []float32 to pgvector's string format
func formatEmbeddingForPgVector(embedding []float32) string {
	strValues := make([]string, len(embedding))
	for i, val := range embedding {
		strValues[i] = fmt.Sprintf("%f", val)
	}
	return "[" + strings.Join(strValues, ",") + "]"
}

// processEmbedding adjusts an embedding to fit in 768 dimensions
// Returns the processed embedding and the original dimension
func processEmbedding(embedding []float32) ([]float32, int) {
	originalDimension := len(embedding)
	targetDimension := 768
	switch {
	case originalDimension == targetDimension:
		return embedding, originalDimension
	case originalDimension < targetDimension:
		return padVector(embedding, targetDimension), originalDimension
	case originalDimension > targetDimension:
		return reduceVector(embedding, targetDimension), originalDimension
	}
	return embedding, originalDimension
}

func padVector(vec []float32, targetDimension int) []float32 {
	result := make([]float32, targetDimension)
	copy(result, vec)
	return result
}

func reduceVector(vec []float32, targetDimension int) []float32 {
	originalDimension := len(vec)
	result := make([]float32, targetDimension)
	ratio := float64(originalDimension) / float64(targetDimension)
	for i := range targetDimension {
		startIdx := int(math.Floor(float64(i) * ratio))
		endIdx := min(int(math.Floor(float64(i+1)*ratio)), originalDimension)
		if startIdx >= endIdx {
			if i < originalDimension {
				result[i] = vec[i]
			}
			continue
		}
		var sum float32 = 0
		for j := startIdx; j < endIdx; j++ {
			sum += vec[j]
		}
		result[i] = sum / float32(endIdx-startIdx)
	}
	return normalizeVector(result)
}

func normalizeVector(vec []float32) []float32 {
	var sum float32 = 0
	for _, v := range vec {
		sum += v * v
	}
	if sum < 1e-10 {
		return vec
	}
	magnitude := float32(math.Sqrt(float64(sum)))
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = v / magnitude
	}
	return result
}
