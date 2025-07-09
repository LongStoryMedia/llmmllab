package mq

import (
	"maistro/models"
	"maistro/util"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	// HandlerTimeoutDuration is the time after which a handler is considered expired
	HandlerTimeoutDuration = 45 * time.Minute
)

// handlerRegistration stores information about a registered handler
type handlerRegistration struct {
	handler    ResultHandler
	registered time.Time
	timeout    time.Duration
}

// handlerRegistry is a thread-safe registry for result handlers with timeout
type handlerRegistry struct {
	handlers map[CorrelationID]handlerRegistration
	mu       sync.Mutex
	done     chan struct{}
}

// newHandlerRegistry creates a new handler registry with timeout management
func newHandlerRegistry() *handlerRegistry {
	r := &handlerRegistry{
		handlers: make(map[CorrelationID]handlerRegistration),
		done:     make(chan struct{}),
	}

	// Start the cleanup goroutine
	go r.cleanupExpiredHandlers()
	return r
}

// register adds a handler for a correlation ID with a timeout
func (r *handlerRegistry) register(correlationID CorrelationID, handler ResultHandler, timeout time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.handlers[correlationID] = handlerRegistration{
		handler:    handler,
		registered: time.Now(),
		timeout:    timeout,
	}

	util.LogDebug("Registered result handler with timeout", logrus.Fields{
		"requestId": correlationID,
		"timeout":   timeout,
	})
}

// deregister removes a handler for a correlation ID
func (r *handlerRegistry) deregister(correlationID CorrelationID) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.handlers, correlationID)
}

// handle processes a result for a specific correlation ID
func (r *handlerRegistry) handle(correlationID CorrelationID, result models.InferenceQueueMessage, err error) bool {
	r.mu.Lock()
	reg, exists := r.handlers[correlationID]
	if exists {
		delete(r.handlers, correlationID)
	}
	r.mu.Unlock()

	if !exists {
		return false
	}

	// Execute the handler
	reg.handler(correlationID, result, err)
	return true
}

// cleanupExpiredHandlers periodically removes expired handlers
func (r *handlerRegistry) cleanupExpiredHandlers() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			r.mu.Lock()
			now := time.Now()
			for id, reg := range r.handlers {
				if now.Sub(reg.registered) > reg.timeout {
					util.LogWarning("Result handler expired", logrus.Fields{
						"requestId": id,
						"duration":  now.Sub(reg.registered),
					})

					// Execute the handler with a timeout error
					go reg.handler(id, models.InferenceQueueMessage{}, util.NewError("result handler timed out"))

					// Remove the expired handler
					delete(r.handlers, id)
				}
			}
			r.mu.Unlock()
		case <-r.done:
			return
		}
	}
}

// close stops the cleanup goroutine
func (r *handlerRegistry) close() {
	close(r.done)
}

// Global registry instance
var handlerReg *handlerRegistry

// initHandlerRegistry initializes the global handler registry
func initHandlerRegistry() {
	if handlerReg == nil {
		handlerReg = newHandlerRegistry()
	}
}

// closeHandlerRegistry closes the handler registry
func closeHandlerRegistry() {
	if handlerReg != nil {
		handlerReg.close()
	}
}

// logRegisteredHandlers logs all currently registered handlers for debugging
func (r *handlerRegistry) logRegisteredHandlers() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.handlers) == 0 {
		util.LogInfo("No handlers currently registered")
		return
	}

	util.LogInfo("Currently registered handlers", logrus.Fields{
		"count": len(r.handlers),
	})

	// Log each registered handler
	for id, reg := range r.handlers {
		util.LogInfo("Registered handler", logrus.Fields{
			"requestId":    id,
			"registeredAt": reg.registered,
			"duration":     time.Since(reg.registered),
			"timeout":      reg.timeout,
		})
	}
}

// handleWithUUIDPart tries to find a handler based on the UUID part of a correlation ID
// This is used when correlation IDs have different prefixes but the same UUID
func (r *handlerRegistry) handleWithUUIDPart(uuidPart string, result models.InferenceQueueMessage) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	util.LogInfo("Looking for handler with UUID part", logrus.Fields{
		"uuidPart": uuidPart,
	})

	// Look for any correlation ID that ends with the UUID part
	for id, reg := range r.handlers {
		// Check if this correlation ID contains the UUID part
		if strings.Contains(string(id), uuidPart) {
			util.LogInfo("Found handler with matching UUID part", logrus.Fields{
				"correlationId": id,
				"uuidPart":      uuidPart,
			})

			// Execute the handler and remove it from the registry
			go reg.handler(id, result, nil)
			delete(r.handlers, id)
			return true
		}
	}

	return false
}
