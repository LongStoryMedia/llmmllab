package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"maistro/config"
	"maistro/models"
	"maistro/util"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

const modelProfileKeyPrefix = "proxyllama:modelprofile:"
const modelProfilesListKeyPrefix = "proxyllama:modelprofiles:"

func getModelProfileCacheKey(profileID uuid.UUID) string {
	return cacheKey(modelProfileKeyPrefix, profileID)
}

func getModelProfilesListCacheKey(userID string) string {
	return cacheKey(modelProfilesListKeyPrefix, userID)
}

// modelProfileStore implements the ModelProfileStore interface
type modelProfileStore struct{}

// GetModelProfileFromCache tries to get a single model profile from Redis
func (mps *modelProfileStore) GetModelProfileFromCache(ctx context.Context, profileID uuid.UUID) (*models.ModelProfile, bool) {
	// Early return if cache is disabled
	if !IsStorageCacheEnabled() {
		return nil, false
	}

	// Safety check for nil context
	if ctx == nil {
		util.LogWarning("nil context passed to GetModelProfileFromCache")
		return nil, false
	}

	// Check for zero UUID
	if profileID == uuid.Nil {
		util.LogWarning("nil UUID passed to GetModelProfileFromCache")
		return nil, false
	}

	key := getModelProfileCacheKey(profileID)

	// Use a timeout for Redis operations to prevent hanging
	redisCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	// Add error handling around Redis operations
	result := redisClient.Get(redisCtx, key)
	if result.Err() != nil {
		// Don't log regular cache misses as they're expected
		if result.Err().Error() != "redis: nil" {
			util.HandleError(fmt.Errorf("redis error in GetModelProfileFromCache: %w", result.Err()))
		}
		return nil, false
	}

	data, err := result.Bytes()
	if err != nil {
		util.HandleError(fmt.Errorf("error getting bytes from Redis for profile: %w", err))
		return nil, false
	}

	var profile models.ModelProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		util.HandleError(fmt.Errorf("error unmarshaling profile data: %w", err))
		return nil, false
	}

	return &profile, true
}

// CacheModelProfile stores a single model profile in Redis
func (mps *modelProfileStore) CacheModelProfile(ctx context.Context, profile *models.ModelProfile) error {
	if !IsStorageCacheEnabled() || profile == nil {
		return nil
	}
	conf := config.GetConfig(nil)
	data, err := json.Marshal(profile)
	if err != nil {
		return err
	}
	key := getModelProfileCacheKey(profile.ID)
	ttl := time.Duration(conf.Redis.MessageTtl) * time.Second
	return redisClient.Set(ctx, key, data, ttl).Err()
}

// InvalidateModelProfileCache removes a single model profile from Redis
func (mps *modelProfileStore) InvalidateModelProfileCache(ctx context.Context, profileID uuid.UUID) {
	if !IsStorageCacheEnabled() {
		return
	}
	key := getModelProfileCacheKey(profileID)
	redisClient.Del(ctx, key)
}

// GetModelProfilesListFromCache tries to get all model profiles for a user from Redis
func (mps *modelProfileStore) GetModelProfilesListFromCache(ctx context.Context, userID string) ([]*models.ModelProfile, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}
	key := getModelProfilesListCacheKey(userID)
	data, err := redisClient.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}
	var profiles []*models.ModelProfile
	if err := json.Unmarshal(data, &profiles); err != nil {
		return nil, false
	}
	return profiles, true
}

// CacheModelProfilesList stores all model profiles for a user in Redis
func (mps *modelProfileStore) CacheModelProfilesList(ctx context.Context, userID string, profiles []*models.ModelProfile) error {
	if !IsStorageCacheEnabled() || profiles == nil {
		return nil
	}
	conf := config.GetConfig(nil)
	data, err := json.Marshal(profiles)
	if err != nil {
		return err
	}
	key := getModelProfilesListCacheKey(userID)
	ttl := time.Duration(conf.Redis.MessageTtl) * time.Second
	return redisClient.Set(ctx, key, data, ttl).Err()
}

// InvalidateModelProfilesListCache removes the model profiles list for a user from Redis
func (mps *modelProfileStore) InvalidateModelProfilesListCache(ctx context.Context, userID string) {
	if !IsStorageCacheEnabled() {
		return
	}
	key := getModelProfilesListCacheKey(userID)
	redisClient.Del(ctx, key)
}

// CreateModelProfile creates a new model profile
func (mps *modelProfileStore) CreateModelProfile(ctx context.Context, profile *models.ModelProfile) (uuid.UUID, error) {
	userID := profile.UserID
	name := profile.Name
	description := profile.Description
	modelName := profile.ModelName
	parameters := profile.Parameters
	systemPrompt := profile.SystemPrompt
	modelVersion := profile.ModelVersion
	profileType := profile.Type

	// Convert parameters to JSON
	parametersJSON, err := json.Marshal(parameters)
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to marshal parameters: %w", err)
	}

	var profileID uuid.UUID
	err = Pool.QueryRow(ctx, GetQuery("modelprofile.create_profile"),
		userID, name, description, modelName, parametersJSON,
		systemPrompt, modelVersion, profileType).Scan(&profileID)

	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to create model profile: %w", err)
	}

	// Invalidate the model profiles cache for this user
	mps.InvalidateModelProfilesListCache(ctx, userID)

	return profileID, nil
}

// GetModelProfile retrieves a model profile by ID
func (mps *modelProfileStore) GetModelProfile(ctx context.Context, profileID uuid.UUID) (*models.ModelProfile, error) {
	tx, err := Pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	mp, err := mps.GetModelProfileWithTx(ctx, profileID, tx)
	if err != nil {
		tx.Rollback(ctx) // Rollback on error
		return nil, util.HandleError(fmt.Errorf("failed to get model profile: %w", err))
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to commit transaction: %w", err))
	}
	// Return the profile retrieved from the transaction
	return mp, nil
}

// GetModelProfileWithTx retrieves a model profile by ID with transaction
func (mps *modelProfileStore) GetModelProfileWithTx(ctx context.Context, profileID uuid.UUID, tx pgx.Tx) (*models.ModelProfile, error) {
	// Special case for nil UUID - look for a default profile based on type
	if profileID == uuid.Nil {
		util.LogWarning("nil UUID passed to GetModelProfile, using primary profile as default")
		return &config.DefaultPrimaryProfile, nil
	}

	// Try to get from cache first
	if profile, found := mps.GetModelProfileFromCache(ctx, profileID); found {
		return profile, nil
	}

	var mp models.ModelProfile
	var parametersJSON []byte

	util.LogInfo("Fetching model profile from database", nil)
	err := tx.QueryRow(ctx, GetQuery("modelprofile.get_profile_by_id"), profileID).Scan(
		&mp.ID, &mp.UserID, &mp.Name, &mp.Description, &mp.ModelName,
		&parametersJSON, &mp.SystemPrompt, &mp.ModelVersion, &mp.Type,
		&mp.CreatedAt, &mp.UpdatedAt)
	if err != nil {
		return nil, util.HandleError(fmt.Errorf("failed to get model profile: %w", err))
	}

	// Parse parameters JSON
	if err := json.Unmarshal(parametersJSON, &mp.Parameters); err != nil {
		// If JSON parsing fails, initialize an empty map
		mp.Parameters = models.ModelParameters{}
		util.LogWarning("Failed to parse parameters JSON", nil)
	}

	// Cache for future use
	if err := mps.CacheModelProfile(ctx, &mp); err != nil {
		// Just log, don't fail on cache error
		util.LogWarning("Failed to cache model profile", nil)
	}

	return &mp, nil
}

// UpdateModelProfile updates an existing model profile
func (mps *modelProfileStore) UpdateModelProfile(ctx context.Context, profile *models.ModelProfile) error {
	profileID := profile.ID
	name := profile.Name
	description := profile.Description
	modelName := profile.ModelName
	parameters := profile.Parameters
	systemPrompt := profile.SystemPrompt
	modelVersion := profile.ModelVersion
	profileType := profile.Type

	// Convert parameters to JSON
	parametersJSON, err := json.Marshal(parameters)
	if err != nil {
		return fmt.Errorf("failed to marshal parameters: %w", err)
	}

	var updateTime time.Time
	err = Pool.QueryRow(ctx, GetQuery("modelprofile.update_profile"),
		profileID, name, description, modelName, parametersJSON,
		systemPrompt, modelVersion, profileType).Scan(&updateTime)
	if err != nil {
		return fmt.Errorf("failed to update model profile: %w", err)
	}

	// Get user ID to invalidate their cache
	var userID string
	err = Pool.QueryRow(ctx, GetQuery("modelprofile.get_user_id"), profileID).Scan(&userID)
	if err != nil {
		util.LogWarning("Could not fetch user_id for profile", nil)
	} else {
		// Invalidate the model profiles cache for this user
		mps.InvalidateModelProfilesListCache(ctx, userID)
	}

	// Invalidate the individual profile cache
	mps.InvalidateModelProfileCache(ctx, profileID)

	return nil
}

// DeleteModelProfile deletes a model profile
func (mps *modelProfileStore) DeleteModelProfile(ctx context.Context, profileID uuid.UUID) error {
	// Get user ID to invalidate their cache
	var userID string
	err := Pool.QueryRow(ctx, GetQuery("modelprofile.get_user_id"), profileID).Scan(&userID)
	if err != nil {
		util.LogWarning("Could not fetch user_id for profile", nil)
		// Continue with deletion anyway
	}

	_, err = Pool.Exec(ctx, GetQuery("modelprofile.delete_profile"), profileID)
	if err != nil {
		return fmt.Errorf("failed to delete model profile: %w", err)
	}

	// Invalidate caches
	if userID != "" {
		mps.InvalidateModelProfilesListCache(ctx, userID)
	}
	mps.InvalidateModelProfileCache(ctx, profileID)

	return nil
}

// ListModelProfilesByUser gets all model profiles for a specific user
func (mps *modelProfileStore) ListModelProfilesByUser(ctx context.Context, userID string) ([]*models.ModelProfile, error) {
	// Try to get from cache first
	if profiles, found := mps.GetModelProfilesListFromCache(ctx, userID); found {
		return profiles, nil
	}

	// Not in cache, get from database
	rows, err := Pool.Query(ctx, GetQuery("modelprofile.list_profiles_by_user"), userID)
	if err != nil {
		return nil, fmt.Errorf("failed to query model profiles: %w", err)
	}
	defer rows.Close()

	profiles := make([]*models.ModelProfile, 0)
	for rows.Next() {
		var mp models.ModelProfile
		var parametersJSON []byte

		// Scan values from the database row
		if err := rows.Scan(
			&mp.ID,
			&mp.UserID,
			&mp.Name,
			&mp.Description,
			&mp.ModelName,
			&parametersJSON,
			&mp.SystemPrompt,
			&mp.ModelVersion,
			&mp.Type,
			&mp.CreatedAt,
			&mp.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan model profile row: %w", err)
		}

		// Parse parameters JSON
		if err := json.Unmarshal(parametersJSON, &mp.Parameters); err != nil {
			// If JSON parsing fails, initialize an empty map
			mp.Parameters = models.ModelParameters{}
			util.LogWarning("Failed to parse parameters JSON", nil)
		}

		profiles = append(profiles, &mp)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating model profile rows: %w", err)
	}

	// Cache for future use
	if err := mps.CacheModelProfilesList(ctx, userID, profiles); err != nil {
		// Just log, don't fail on cache error
		util.LogWarning("Failed to cache model profiles for user", nil)
	}

	return profiles, nil
}
