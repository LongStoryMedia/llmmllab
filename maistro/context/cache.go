package context

import (
	"context"
	"fmt"
	"sync"
	"time"

	"maistro/util"
)

type cacheEntry struct {
	context   ConversationContext
	expiresAt time.Time
}

type cacheKey struct {
	userID         string
	conversationID int
}

type CacheProvider interface {
	Get(userID string, conversationID int) (ConversationContext, bool)
	Set(convContext ConversationContext)
	Remove(userID string, conversationID int)
	Clear()
	Size() int
	Stop()
}

type InMemoryCache struct {
	entries         map[cacheKey]cacheEntry
	mutex           sync.RWMutex
	ttl             time.Duration
	janitorInterval time.Duration
	stopJanitor     chan bool
}

var (
	Cache           CacheProvider
	defaultTTL      = 30 * time.Minute
	janitorCycleTTL = 5 * time.Minute
)

func InitCache(ttl time.Duration) CacheProvider {
	if ttl == 0 {
		ttl = defaultTTL
	}
	util.LogInfo("Initializing in-memory cache", map[string]interface{}{"ttl": ttl})
	cache := NewInMemoryCache(ttl)
	Cache = cache
	return cache
}

func NewInMemoryCache(ttl time.Duration) *InMemoryCache {
	cache := &InMemoryCache{
		entries:         make(map[cacheKey]cacheEntry),
		ttl:             ttl,
		janitorInterval: janitorCycleTTL,
		stopJanitor:     make(chan bool),
	}
	go cache.startJanitor()
	return cache
}

func (c *InMemoryCache) startJanitor() {
	ticker := time.NewTicker(c.janitorInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.cleanExpired()
		case <-c.stopJanitor:
			return
		}
	}
}

func (c *InMemoryCache) Stop() {
	c.stopJanitor <- true
}

func (c *InMemoryCache) cleanExpired() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	now := time.Now()
	for k, v := range c.entries {
		if now.After(v.expiresAt) {
			delete(c.entries, k)
		}
	}
}

func (c *InMemoryCache) Get(userID string, conversationID int) (ConversationContext, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	key := cacheKey{userID: userID, conversationID: conversationID}
	entry, found := c.entries[key]

	if !found {
		return nil, false
	}

	if time.Now().After(entry.expiresAt) {
		return nil, false
	}

	return entry.context, true
}

func (c *InMemoryCache) Set(convContext ConversationContext) {
	if convContext == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()
	key := cacheKey{
		userID:         convContext.GetUserID(),
		conversationID: convContext.GetConversationID(),
	}
	c.entries[key] = cacheEntry{
		context:   convContext,
		expiresAt: time.Now().Add(c.ttl),
	}
}

func (c *InMemoryCache) Remove(userID string, conversationID int) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	key := cacheKey{userID: userID, conversationID: conversationID}
	delete(c.entries, key)
}

func (c *InMemoryCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.entries = make(map[cacheKey]cacheEntry)
}

func (c *InMemoryCache) Size() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	return len(c.entries)
}

func GetCache() CacheProvider {
	if Cache == nil {
		ttl := defaultTTL
		InitCache(ttl)
	}
	return Cache
}

func GetCachedConversation(userID string, conversationID int) (ConversationContext, error) {
	cache := GetCache()

	if cachedContext, found := cache.Get(userID, conversationID); found {
		return cachedContext, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	convContext, err := GetOrCreateConversation(ctx, userID, &conversationID)

	if err != nil {
		return nil, fmt.Errorf("failed to retrieve conversation: %w", err)
	}

	cache.Set(convContext)

	return convContext, nil
}

func InvalidateConversation(userID string, conversationID int) {
	GetCache().Remove(userID, conversationID)
}
