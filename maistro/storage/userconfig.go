package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"maistro/config"
	"maistro/models"
	"maistro/util"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

const userConfigKeyPrefix = "proxyllama:userconfig:"

// getUserConfigCacheKey constructs the Redis key for a user's config
func getUserConfigCacheKey(userID string) string {
	return userConfigKeyPrefix + userID
}

// userConfigStore implements the UserConfigStore interface
type userConfigStore struct{}

// getUserConfigFromCache tries to get user config from Redis
func (ucs *userConfigStore) getUserConfigFromCache(ctx context.Context, userID string) (*models.UserConfig, bool) {
	if !IsStorageCacheEnabled() {
		return nil, false
	}
	key := getUserConfigCacheKey(userID)
	data, err := redisClient.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}
	var userConfig models.UserConfig
	if err := json.Unmarshal(data, &userConfig); err != nil {
		return nil, false
	}
	return &userConfig, true
}

// cacheUserConfig stores user config in Redis
func (ucs *userConfigStore) cacheUserConfig(ctx context.Context, userID string, cfg *models.UserConfig) {
	if !IsStorageCacheEnabled() || cfg == nil {
		return
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		return
	}
	ttl := 24 * time.Hour // Cache user config for 24h, adjust as needed
	key := getUserConfigCacheKey(userID)
	util.LogInfo("Caching user config", map[string]interface{}{
		"userID": userID,
		"ttl":    ttl,
	})
	redisClient.Set(ctx, key, data, ttl)
}

// GetUserConfig retrieves user configuration from database
func (ucs *userConfigStore) GetUserConfig(ctx context.Context, userID string) (*models.UserConfig, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	// Ensure user exists
	if err := EnsureUser(ctx, userID); err != nil {
		util.HandleError(err)
		return nil, err
	}

	// Try to get from cache first
	usrCfg, found := ucs.getUserConfigFromCache(ctx, userID)
	if found {
		util.LogDebug("User config found in cache", logrus.Fields{
			"userID":  userID,
			"config":  *usrCfg,
			"summary": *(*usrCfg).Summarization,
		})
		return usrCfg, nil
	}

	// Parse JSON into config struct
	var usrConfig models.UserConfig
	err := Pool.QueryRow(ctx, GetQuery("user.get_config"), userID).Scan(&usrConfig)
	if err != nil {
		if err == sql.ErrNoRows || strings.Contains(err.Error(), "cannot scan NULL into *models.UserConfig") {
			// No rows found, or empty config
			util.LogWarning("No user config found, setting to default", logrus.Fields{"userID": userID})
			usrConfig = models.UserConfig{UserID: userID}
		} else {
			util.HandleError(err)
			return nil, err
		}
	} else {
		// Successfully retrieved user config
		util.LogInfo("User config retrieved from database", logrus.Fields{"userID": userID})
	}

	// Ensure all required fields have values by merging with defaults
	config.MergeWithDefaultConfig(&usrConfig)

	// Cache for future use
	ucs.cacheUserConfig(ctx, userID, &usrConfig)

	return &usrConfig, nil
}

// UpdateUserConfig saves user configuration to database
func (ucs *userConfigStore) UpdateUserConfig(ctx context.Context, userID string, cfg *models.UserConfig) error {
	// Convert config to JSON
	configJson, err := json.Marshal(cfg)
	if err != nil {
		return err
	}

	// Update in database
	_, err = Pool.Exec(ctx, GetQuery("user.update_config"), configJson, userID)
	if err != nil {
		return err
	}

	// Update the cache
	ucs.cacheUserConfig(ctx, userID, cfg)

	return nil
}

// GetAllUsers retrieves all users from the database
func (ucs *userConfigStore) GetAllUsers(ctx context.Context) ([]models.User, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	rows, err := Pool.Query(ctx, "SELECT id, created_at, username FROM users")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []models.User
	for rows.Next() {
		var u models.User
		if err := rows.Scan(&u.ID, &u.CreatedAt, &u.Username); err != nil {
			return nil, err
		}
		users = append(users, u)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return users, nil
}
