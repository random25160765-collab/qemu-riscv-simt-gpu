package main

import (
	"sort"
	"sync"
	"time"
)

// ── Per-warp timing ───────────────────────────────────────

type WarpStats struct {
	EngineTime  time.Duration // total time in engine_exec
	BarrierTime time.Duration // total time waiting at barrier
	Barriers    int           // how many barriers hit
}

// ── Global stats ──────────────────────────────────────────

type Stats struct {
	mu           sync.Mutex
	TotalWarps   int
	EngineTime   time.Duration
	BarrierTime  time.Duration
	BarrierWaits int
	latencies    []time.Duration // per-warp total engine time
	LatencyP50   time.Duration
	LatencyP95   time.Duration
	LatencyP99   time.Duration
	LatencyMax   time.Duration
}

var globalStats = &Stats{latencies: make([]time.Duration, 0, 32768)}

func RecordWarp(ws WarpStats) {
	s := globalStats
	s.mu.Lock()
	s.TotalWarps++
	s.EngineTime += ws.EngineTime
	s.BarrierTime += ws.BarrierTime
	s.BarrierWaits += ws.Barriers
	s.latencies = append(s.latencies, ws.EngineTime)
	s.mu.Unlock()
}

func GetStats() *Stats {
	s := globalStats
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.latencies) > 0 {
		sort.Slice(s.latencies, func(i, j int) bool {
			return s.latencies[i] < s.latencies[j]
		})
		n := len(s.latencies)
		s.LatencyP50 = s.latencies[n*50/100]
		s.LatencyP95 = s.latencies[n*95/100]
		s.LatencyP99 = s.latencies[n*99/100]
		s.LatencyMax = s.latencies[n-1]
	}
	return s
}
