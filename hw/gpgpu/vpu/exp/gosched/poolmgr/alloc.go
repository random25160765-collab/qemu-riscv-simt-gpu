package poolmgr

import (
	"fmt"
	"sync"
	"time"
)

// ── Allocator ──────────────────────────────────────────────

type AllocStats struct {
	Total    int
	Idle     int
	Busy     int
	Error    int
	Uptime   time.Duration
	Dispatch int64
}

type Allocator struct {
	free     chan *VPUInstance
	all      []*VPUInstance
	mu       sync.Mutex
	dispatch int64
}

func NewAllocator(instances []*VPUInstance) *Allocator {
	a := &Allocator{
		free: make(chan *VPUInstance, len(instances)),
		all:  instances,
	}
	for _, v := range instances {
		if v.Status == "idle" {
			a.free <- v
		}
	}
	return a
}

func (a *Allocator) Alloc() (*VPUInstance, error) {
	v := <-a.free
	if v.Status != "idle" {
		a.free <- v // put back, try again
		return nil, fmt.Errorf("vpu %d: not idle", v.ID)
	}
	a.mu.Lock()
	a.dispatch++
	a.mu.Unlock()
	return v, nil
}

func (a *Allocator) Free(v *VPUInstance) {
	if v.Status != "idle" {
		v.Status = "idle"
	}
	a.free <- v
}

func (a *Allocator) Stats() AllocStats {
	a.mu.Lock()
	defer a.mu.Unlock()
	s := AllocStats{Total: len(a.all), Dispatch: a.dispatch}
	for _, v := range a.all {
		switch v.Status {
		case "idle": s.Idle++
		case "busy": s.Busy++
		default: s.Error++
		}
	}
	return s
}

func (a *Allocator) Close() {
	for _, v := range a.all {
		v.Close()
	}
}
