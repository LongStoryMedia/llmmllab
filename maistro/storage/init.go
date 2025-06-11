package storage

import (
	"context"
	"errors"
	"fmt"
	"maistro/config"
	"maistro/util"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sirupsen/logrus"
)

var (
	Pool       *pgxpool.Pool
	initOnce   sync.Once = sync.Once{}
	connString string    // Store connection string for reconnection logic
)

// InitDB initializes the database connection
func InitDB(connStr string) error {
	var err error
	var config *pgxpool.Config
	connString = connStr

	initOnce.Do(func() {
		LoadQueries()
		util.LogInfo("Initializing PostgreSQL database connection...")

		config, err = pgxpool.ParseConfig(connStr)
		if err != nil {
			err = fmt.Errorf("unable to parse postgres connection string: %w", err)
			return
		}

		// Build the connection pool
		Pool, err = pgxpool.NewWithConfig(context.Background(), config)
		if err != nil {
			err = fmt.Errorf("failed to create connection pool: %w", err)
			return
		}

		// Ping the database to verify the connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err = Pool.Ping(ctx); err != nil {
			err = fmt.Errorf("failed to ping database: %w", err)
			return
		}

		// Initialize all the tables
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		err = InitializeTables(ctx)
		if err != nil {
			err = fmt.Errorf("failed to initialize tables: %w", err)
			return
		}
		util.LogInfo("Successfully initialized tables")

		// Ensure research tables are created
		err = EnsureResearchTables(ctx)
		if err != nil {
			err = fmt.Errorf("failed to ensure research tables: %w", err)
			return
		}
		util.LogInfo("Successfully ensured research tables")

		// Create default model profiles
		err = CreateDefaultProfiles(ctx)
		if err != nil {
			err = fmt.Errorf("failed to create default model profiles: %w", err)
			return
		}

		util.LogInfo("Successfully connected to PostgreSQL")
	})
	return err
}

// EnsureDBConnection checks the database connection and attempts to reconnect if necessary
func EnsureDBConnection(ctx context.Context) error {
	if Pool == nil {
		util.LogInfo("Connection pool is nil, initializing database")

		if connString == "" {
			return errors.New("connection string is empty, cannot reconnect")
		}
		return InitDB(connString)
	}

	// Check if the connection is still valid
	if err := Pool.Ping(ctx); err != nil {
		util.LogWarning("Database connection lost, attempting to reconnect", logrus.Fields{"error": err})

		// Close the old pool
		Pool.Close()

		// Reinitialize
		if connString == "" {
			return errors.New("connection string is empty, cannot reconnect")
		}
		return InitDB(connString)
	}

	return nil
}

// InitializeTables creates all necessary database tables
func InitializeTables(ctx context.Context) error {
	// Create extensions
	_, err := Pool.Exec(ctx, GetQuery("init.create_extensions"))
	if err != nil {
		return fmt.Errorf("failed to create extensions: %w", err)
	}

	// Create users table
	_, err = Pool.Exec(ctx, GetQuery("user.create_users_table"))
	if err != nil {
		return fmt.Errorf("failed to create users table: %w", err)
	}

	// Create conversations table
	_, err = Pool.Exec(ctx, GetQuery("conversation.create_conversations_table"))
	if err != nil {
		return fmt.Errorf("failed to create conversations table: %w", err)
	}

	// Create messages table
	_, err = Pool.Exec(ctx, GetQuery("message.create_messages_table"))
	if err != nil {
		return fmt.Errorf("failed to create messages table: %w", err)
	}

	// Create messages indexes
	_, err = Pool.Exec(ctx, GetQuery("message.create_messages_indexes"))
	if err != nil {
		return fmt.Errorf("failed to create message indexes: %w", err)
	}

	// Create message embeddings table
	_, err = Pool.Exec(ctx, GetQuery("message.create_message_embeddings_table"))
	if err != nil {
		return fmt.Errorf("failed to create message embeddings table: %w", err)
	}

	// Create hypertable for message_embeddings
	_, err = Pool.Exec(ctx, GetQuery("message.create_message_embeddings_hypertable"))
	if err != nil {
		return fmt.Errorf("failed to create message embeddings hypertable: %w", err)
	}

	// Enable compression for message_embeddings
	_, err = Pool.Exec(ctx, GetQuery("message.enable_message_embeddings_compression"))
	if err != nil {
		return fmt.Errorf("failed to enable message embeddings compression: %w", err)
	}

	// Add compression policy for message_embeddings
	_, err = Pool.Exec(ctx, GetQuery("message.message_embeddings_compression_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add message embeddings compression policy", logrus.Fields{"error": err})
	}

	// Add retention policy for message_embeddings
	_, err = Pool.Exec(ctx, GetQuery("message.message_embeddings_retention_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add message embeddings retention policy", logrus.Fields{"error": err})
	}

	// Create summaries table
	_, err = Pool.Exec(ctx, GetQuery("summary.create_summaries_table"))
	if err != nil {
		return fmt.Errorf("failed to create summaries table: %w", err)
	}

	// Create model profiles table
	_, err = Pool.Exec(ctx, GetQuery("modelprofile.create_model_profiles_table"))
	if err != nil {
		return fmt.Errorf("failed to create model profiles table: %w", err)
	}

	// Create model profiles index
	_, err = Pool.Exec(ctx, GetQuery("modelprofile.create_model_profiles_index"))
	if err != nil {
		return fmt.Errorf("failed to create model profiles index: %w", err)
	}

	// Create hypertable for conversations
	_, err = Pool.Exec(ctx, GetQuery("conversation.create_conversations_hypertable"))
	if err != nil {
		return fmt.Errorf("failed to create conversations hypertable: %w", err)
	}

	// Create conversations indexes
	_, err = Pool.Exec(ctx, GetQuery("conversation.create_conversations_indexes"))
	if err != nil {
		return fmt.Errorf("failed to create conversations indexes: %w", err)
	}

	// Create hypertable for messages
	_, err = Pool.Exec(ctx, GetQuery("message.create_messages_hypertable"))
	if err != nil {
		return fmt.Errorf("failed to create messages hypertable: %w", err)
	}

	// Create additional message indexes
	_, err = Pool.Exec(ctx, GetQuery("message.create_additional_messages_indexes"))
	if err != nil {
		return fmt.Errorf("failed to create additional message indexes: %w", err)
	}

	// Create hypertable for summaries
	_, err = Pool.Exec(ctx, GetQuery("summary.create_summaries_hypertable"))
	if err != nil {
		return fmt.Errorf("failed to create summaries hypertable: %w", err)
	}

	// Create summaries indexes
	_, err = Pool.Exec(ctx, GetQuery("summary.create_summaries_indexes"))
	if err != nil {
		return fmt.Errorf("failed to create summaries indexes: %w", err)
	}

	// Create user check trigger
	_, err = Pool.Exec(ctx, GetQuery("user.create_user_check_trigger"))
	if err != nil {
		return fmt.Errorf("failed to create user check trigger: %w", err)
	}

	// Create conversation update trigger
	_, err = Pool.Exec(ctx, GetQuery("conversation.create_conversation_update_trigger"))
	if err != nil {
		return fmt.Errorf("failed to create conversation update trigger: %w", err)
	}

	// Create conversation check triggers
	_, err = Pool.Exec(ctx, GetQuery("conversation.create_conversation_check_triggers"))
	if err != nil {
		return fmt.Errorf("failed to create conversation check triggers: %w", err)
	}

	// Create cascade delete trigger
	_, err = Pool.Exec(ctx, GetQuery("conversation.create_cascade_delete_trigger"))
	if err != nil {
		return fmt.Errorf("failed to create cascade delete trigger: %w", err)
	}

	// Enable compression on messages
	_, err = Pool.Exec(ctx, GetQuery("message.enable_messages_compression"))
	if err != nil {
		return fmt.Errorf("failed to enable messages compression: %w", err)
	}

	// Enable compression on conversations
	_, err = Pool.Exec(ctx, GetQuery("conversation.enable_conversations_compression"))
	if err != nil {
		return fmt.Errorf("failed to enable conversations compression: %w", err)
	}

	// Enable compression on summaries
	_, err = Pool.Exec(ctx, GetQuery("summary.enable_summaries_compression"))
	if err != nil {
		return fmt.Errorf("failed to enable summaries compression: %w", err)
	}

	// Add compression policy for messages
	_, err = Pool.Exec(ctx, GetQuery("message.messages_compression_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add messages compression policy", logrus.Fields{"error": err})
	}

	// Add compression policy for conversations
	_, err = Pool.Exec(ctx, GetQuery("conversation.conversations_compression_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add conversations compression policy", logrus.Fields{"error": err})
	}

	// Add compression policy for summaries
	_, err = Pool.Exec(ctx, GetQuery("summary.summaries_compression_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add summaries compression policy", logrus.Fields{"error": err})
	}

	// Add retention policy for conversations
	_, err = Pool.Exec(ctx, GetQuery("conversation.conversations_retention_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add conversations retention policy", logrus.Fields{"error": err})
	}

	// Add retention policy for messages
	_, err = Pool.Exec(ctx, GetQuery("message.messages_retention_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add messages retention policy", logrus.Fields{"error": err})
	}

	// Add retention policy for summaries
	_, err = Pool.Exec(ctx, GetQuery("summary.summaries_retention_policy"))
	if err != nil {
		util.LogWarning("Warning: Failed to add summaries retention policy", logrus.Fields{"error": err})
	}

	// Initialize memory schema
	(&memoryStore{}).InitMemorySchema(ctx)

	return nil
}

// EnsureResearchTables ensures that the research tables are created
func EnsureResearchTables(ctx context.Context) error {
	// Create research_tasks table
	_, err := Pool.Exec(ctx, GetQuery("research.create_research_tasks_table"))
	if err != nil {
		return fmt.Errorf("failed to create research_tasks table: %w", err)
	}

	// Create research_subtasks table
	_, err = Pool.Exec(ctx, GetQuery("research.create_research_subtasks_table"))
	if err != nil {
		return fmt.Errorf("failed to create research_subtasks table: %w", err)
	}

	return nil
}

// CreateDefaultProfiles creates the default model profiles in the database
func CreateDefaultProfiles(ctx context.Context) error {

	tx, err := Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Insert each default profile
	systemUserID := "0"

	for _, profile := range config.DefaultModelProfiles {
		util.LogInfo("Inserting default profile", logrus.Fields{
			"profileName": profile.Name,
			"profileId":   profile.ID,
		})

		_, err = tx.Exec(ctx,
			GetQuery("modelprofile.create_default_profile"),
			profile.ID.String(),
			systemUserID,
			profile.Name,
			profile.Description,
			profile.ModelName,
			profile.Parameters,
			profile.SystemPrompt,
			profile.ModelVersion,
			profile.Type,
		)
		if err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}

// PerformDatabaseMaintenance runs optimization and maintenance tasks for the database
func PerformDatabaseMaintenance(ctx context.Context) error {
	util.LogInfo("Starting database maintenance tasks...")

	// Vacuum analyze for better query planning
	_, err := Pool.Exec(ctx, `VACUUM ANALYZE`)
	if err != nil {
		return fmt.Errorf("failed to vacuum analyze: %w", err)
	}
	util.LogInfo("VACUUM ANALYZE completed")

	// Reindex tables to optimize indexes
	_, err = Pool.Exec(ctx, "REINDEX (VERBOSE, CONCURRENTLY) DATABASE ollama")
	if err != nil {
		return fmt.Errorf("failed to reindex DB: %w", err)
	}
	util.LogInfo("REINDEX completed")

	// Run TimescaleDB-specific maintenance
	_, err = Pool.Exec(ctx, `SELECT run_job(j.id) FROM timescaledb_information.jobs j WHERE j.proc_name = 'policy_refresh'`)
	if err != nil {
		util.LogWarning("Note: TimescaleDB policy refresh failed (may be normal if no jobs)", logrus.Fields{"error": err})
	} else {
		util.LogInfo("TimescaleDB policy refresh completed")
	}

	util.LogInfo("Database maintenance tasks completed successfully")
	return nil
}

// EnsureUser creates a user if they don't exist
func EnsureUser(ctx context.Context, userID string) error {
	_, err := Pool.Exec(ctx, GetQuery("user.ensure_user"), userID)
	return err
}
