package context

import (
	"context"
	"maistro/config"
	"maistro/storage"
	"sync"
	"time"

	"maistro/util"
)

// userConfigCacheEntry holds a user config and its expiry
type userConfigCacheEntry struct {
	config    *config.UserConfig
	expiresAt time.Time
}

var (
	userConfigCache      = make(map[string]userConfigCacheEntry)
	userConfigCacheMutex sync.RWMutex
	userConfigCacheTTL   = 30 * time.Minute

	userConfigCacheHits      uint64
	userConfigCacheMisses    uint64
	userConfigCacheEvictions uint64

	userConfigJanitorStop = make(chan struct{})
)

func init() {
	go userConfigJanitor()
}

// userConfigJanitor periodically removes expired entries
func userConfigJanitor() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			userConfigCacheMutex.Lock()
			now := time.Now()
			for k, v := range userConfigCache {
				if now.After(v.expiresAt) {
					delete(userConfigCache, k)
					userConfigCacheEvictions++
				}
			}
			userConfigCacheMutex.Unlock()
		case <-userConfigJanitorStop:
			return
		}
	}
}

// UpdateUserConfig provides a wrapper around the config package's UpdateUserConfig
func UpdateUserConfig(userConfig config.UserConfig) {
	conf := config.GetConfig(nil)
	conf.UpdateUserConfig(userConfig)
}

// GetUserConfig returns the user config from cache, falling back to storage
func GetUserConfig(userID string) (*config.UserConfig, error) {
	userConfigCacheMutex.RLock()
	entry, ok := userConfigCache[userID]
	userConfigCacheMutex.RUnlock()
	if ok && entry.config != nil && time.Now().Before(entry.expiresAt) {
		userConfigCacheHits++
		return entry.config, nil
	}
	userConfigCacheMisses++
	// Fallback to storage
	util.LogDebug("User config not found in cache, retrieving from storage", map[string]interface{}{"userID": userID})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ucFromStorage, err := storage.UserConfigStoreInstance.GetUserConfig(ctx, userID)
	if err != nil {
		return nil, err
	}
	SetUserConfig(ucFromStorage)
	return ucFromStorage, nil
}

// SetUserConfig updates the cache and storage
func SetUserConfig(userConfig *config.UserConfig) error {
	if userConfig == nil {
		util.LogWarning("SetUserConfig called with nil userConfig")
		return nil
	}
	userConfigCacheMutex.Lock()
	userConfigCache[userConfig.UserID] = userConfigCacheEntry{
		config:    userConfig,
		expiresAt: time.Now().Add(userConfigCacheTTL),
	}
	userConfigCacheMutex.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return storage.UserConfigStoreInstance.UpdateUserConfig(ctx, userConfig.UserID, userConfig)
}

// InvalidateCachedUserConfig removes a user config from the cache
func InvalidateCachedUserConfig(userID string) {
	userConfigCacheMutex.Lock()
	if _, ok := userConfigCache[userID]; ok {
		delete(userConfigCache, userID)
		userConfigCacheEvictions++
	}
	userConfigCacheMutex.Unlock()
}

// GetUserConfigCacheStats returns cache hit/miss/eviction stats
func GetUserConfigCacheStats() (hits, misses, evictions, size uint64) {
	userConfigCacheMutex.RLock()
	size = uint64(len(userConfigCache))
	userConfigCacheMutex.RUnlock()
	return userConfigCacheHits, userConfigCacheMisses, userConfigCacheEvictions, size
}

// StopUserConfigJanitor stops the janitor goroutine (for clean shutdown)
func StopUserConfigJanitor() {
	close(userConfigJanitorStop)
}
