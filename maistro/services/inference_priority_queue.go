package svc

import (
	"container/heap"
	"encoding/json"
	"errors"
	"maistro/models"
	"maistro/util"
	"net/http"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

type DispatchFunc func(args ...any) InferenceResponse

// InferenceResponse is sent on the ResponseChan when a request completes
// Result is the return value from Dispatch, Err is any error
type InferenceResponse struct {
	Result any
	Err    error
}

// InferenceRequest represents a queued inference job
// Priority: lower value = higher priority
// RequiredMemory: in bytes
// DeviceType: "gpu" or "cpu" (optional)
type InferenceRequest struct {
	Priority       int
	RequiredMemory float32 // bytes
	EnqueueTime    time.Time
	Dispatch       DispatchFunc
	DispatchArgs   []any
	ResponseChan   chan InferenceResponse // Optional: if set, result is sent here
	ReadyTime      time.Time              // Time when this request is ready to be processed (for delayed processing)
}

// PriorityQueue implements heap.Interface and holds InferenceRequests
// Lower Priority value means higher priority
// FIFO for same priority

type PriorityQueue []*InferenceRequest

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool {
	// First check if both requests are ready to be processed
	now := time.Now()
	iReady := now.After(pq[i].ReadyTime)
	jReady := now.After(pq[j].ReadyTime)

	// If one is ready and the other isn't, the ready one has higher priority
	if iReady && !jReady {
		return true
	}
	if !iReady && jReady {
		return false
	}

	// If both are ready or both are not ready, use standard priority comparison
	if pq[i].Priority == pq[j].Priority {
		return pq[i].EnqueueTime.Before(pq[j].EnqueueTime)
	}
	return pq[i].Priority < pq[j].Priority
}
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }
func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*InferenceRequest))
}
func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// InferenceScheduler manages the queue and dispatches requests

type InferenceScheduler struct {
	queue PriorityQueue
	lock  sync.Mutex
	cond  *sync.Cond
	stop  chan struct{}
	// Add a callback or channel for dispatching jobs as needed
}

func NewInferenceScheduler() *InferenceScheduler {
	s := &InferenceScheduler{
		queue: make(PriorityQueue, 0),
		stop:  make(chan struct{}),
	}
	s.cond = sync.NewCond(&s.lock)
	go s.run()
	return s
}

func NewResponseChan() chan InferenceResponse {
	return make(chan InferenceResponse, 1) // Buffered to avoid blocking
}

func (s *InferenceScheduler) Enqueue(req *InferenceRequest) {
	s.lock.Lock()

	// Set ReadyTime based on priority - higher priority requests are processed sooner
	// Priority 0 = process immediately, Priority 1 = delay by 1 second, etc.
	req.ReadyTime = time.Now().Add(time.Duration(req.Priority) * time.Second)

	// add based on priority and enqueue time
	heap.Push(&s.queue, req)
	s.cond.Signal()
	s.lock.Unlock()
}

func (s *InferenceScheduler) Stop() {
	close(s.stop)
	s.cond.Broadcast()
}

// run is the main scheduling loop

func (s *InferenceScheduler) run() {
	// conf := config.GetConfig(nil)
	for {
		select {
		case <-s.stop:
			return
		default:
			// Continue processing
		}

		// Check if there are tasks to process
		s.lock.Lock()
		if s.queue.Len() == 0 {
			s.cond.Wait() // Wait for signal that queue has items
			s.lock.Unlock()
			continue
		}
		s.lock.Unlock()

		// Fetch device memory stats
		stats, err := fetchDeviceStats("http://192.168.0.71:8000/resources/malloc")
		if err != nil {
			util.HandleError(err)
			time.Sleep(500 * time.Millisecond)
			continue
		}

		// Process the queue with the current memory information
		s.processQueueWithAvailableMemory(stats)

		// Sleep a bit to avoid hammering the API
		time.Sleep(100 * time.Millisecond)
	}
}

// processQueueWithAvailableMemory examines the queue and dispatches tasks that can fit in available memory
func (s *InferenceScheduler) processQueueWithAvailableMemory(stats map[string]models.DevStats) {
	// Create a memory availability map
	availableMemory := make(map[string]float32)
	for device, stat := range stats {
		availableMemory[device] = util.Mb2b(stat.MemFree)
	}

	// Keep track of total available memory across all devices
	totalAvailableMemory := float32(0)
	for _, mem := range availableMemory {
		totalAvailableMemory += mem
	}

	s.lock.Lock()
	defer s.lock.Unlock()

	// No requests to process
	if s.queue.Len() == 0 {
		util.LogInfo("No inference requests to process")
		s.cond.Wait() // Wait for new requests to be enqueued
		s.lock.Unlock()
		util.LogInfo("Resuming processing after waiting for new requests")
		return
	}

	// Track which requests to process
	toDispatch := make([]*InferenceRequest, 0)
	now := time.Now()

	// First pass: identify which requests can be satisfied with current memory
	for i := 0; i < s.queue.Len(); i++ {
		req := s.queue[i]

		// Skip if request is not ready to process yet (delay based on priority)
		if now.Before(req.ReadyTime) {
			util.LogDebug("Request not ready yet", logrus.Fields{
				"priority": req.Priority,
				"readyIn":  req.ReadyTime.Sub(now).Seconds(),
			})
			continue
		}

		// Check if any single device has enough memory
		canFit := false
		for device, mem := range availableMemory {
			util.LogDebug("Fetched device stats", logrus.Fields{
				"mem":      mem,
				"device":   device,
				"required": req.RequiredMemory,
			})
			if mem >= req.RequiredMemory {
				// This device has enough memory
				canFit = true
				// Reserve this memory
				availableMemory[device] -= req.RequiredMemory
				totalAvailableMemory -= req.RequiredMemory
				break
			}
		}

		if canFit {
			toDispatch = append(toDispatch, req)
		}
	}

	// Second pass: remove items from queue and dispatch them
	for _, req := range toDispatch {
		// Find and remove from the queue
		for i := 0; i < s.queue.Len(); i++ {
			if s.queue[i] == req {
				heap.Remove(&s.queue, i)
				break
			}
		}

		// Dispatch in a separate goroutine to avoid blocking
		go s.dispatch(req)
	}

	if len(toDispatch) > 0 {
		util.LogInfo("Dispatched inference requests", logrus.Fields{
			"count":           len(toDispatch),
			"remaining_queue": s.queue.Len(),
		})
	}
}

func (s *InferenceScheduler) dispatch(req *InferenceRequest) {
	req.ResponseChan <- req.Dispatch(req.DispatchArgs...)
}

// fetchDeviceStats fetches /resources/malloc and parses the result
func fetchDeviceStats(url string) (map[string]models.DevStats, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, errors.New("non-200 response from inference API")
	}
	var result struct {
		Devices map[string]models.DevStats `json:"devices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Devices, nil
}
