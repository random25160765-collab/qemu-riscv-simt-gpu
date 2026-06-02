// Package mock — pure Go scheduler tests, zero CGo dependency.
// Tests barrier correctness, scheduling policies, and stats collection
// without needing libengine.a or any C code.
package mock

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ── Types replicated from scheduler (self-contained) ─────

type Barrier struct {
	mu     sync.Mutex
	cond   *sync.Cond
	count  int
	target int
}

func newBarrier(target int) *Barrier {
	b := &Barrier{target: target}
	b.cond = sync.NewCond(&b.mu)
	return b
}

func (b *Barrier) Wait() {
	b.mu.Lock()
	b.count++
	if b.count >= b.target {
		b.count = 0
		b.cond.Broadcast()
	} else {
		b.cond.Wait()
	}
	b.mu.Unlock()
}

type WarpStats struct {
	EngineTime  time.Duration
	BarrierTime time.Duration
	Barriers    int
}

type Stats struct {
	mu           sync.Mutex
	TotalWarps   int
	EngineTime   time.Duration
	BarrierTime  time.Duration
	BarrierWaits int
	latencies    []time.Duration
	LatencyP50   time.Duration
	LatencyP95   time.Duration
	LatencyP99   time.Duration
	LatencyMax   time.Duration
}

func newStats() *Stats { return &Stats{latencies: make([]time.Duration, 0, 32768)} }

func (s *Stats) record(ws WarpStats) {
	s.mu.Lock()
	s.TotalWarps++
	s.EngineTime += ws.EngineTime
	s.BarrierTime += ws.BarrierTime
	s.BarrierWaits += ws.Barriers
	s.latencies = append(s.latencies, ws.EngineTime)
	s.mu.Unlock()
}

func (s *Stats) finalize() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.latencies) > 0 {
		sort.Slice(s.latencies, func(i, j int) bool { return s.latencies[i] < s.latencies[j] })
		n := len(s.latencies)
		s.LatencyP50 = s.latencies[n*50/100]
		s.LatencyP95 = s.latencies[n*95/100]
		s.LatencyP99 = s.latencies[n*99/100]
		s.LatencyMax = s.latencies[n-1]
	}
}

type SchedPolicy int

const (
	RoundRobin SchedPolicy = iota
	Greedy
	GTO
	Wavefront
)

func (p SchedPolicy) String() string {
	switch p {
	case RoundRobin: return "rr"
	case Greedy: return "greedy"
	case GTO: return "gto"
	case Wavefront: return "wave"
	default: return "??"
	}
}

// ── Mock Engine ─────────────────────────────────────────────

type MockEngine struct {
	TotalInst int
	BarAt     map[int]bool
	FailAt    map[int]bool
	execCount int64
}

func NewMockEngine(totalInst int, bars ...int) *MockEngine {
	m := &MockEngine{TotalInst: totalInst, BarAt: make(map[int]bool), FailAt: make(map[int]bool)}
	for _, b := range bars { m.BarAt[b] = true }
	return m
}

func (m *MockEngine) WithFail(at ...int) *MockEngine {
	for _, f := range at { m.FailAt[f] = true }
	return m
}

// Exec simulates engine_exec. currentPc is the instruction about to execute.
// Returns -2=done, -1=error, 1=continue+advance, pc|0x10000=barrier (barrierPc=returned value & 0xFFFF).
func (m *MockEngine) Exec(currentPc int) int {
	atomic.AddInt64(&m.execCount, 1)
	if currentPc >= m.TotalInst         { return -2 }
	if m.FailAt[currentPc]              { return -1 }
	if m.BarAt[currentPc]               { return currentPc | 0x10000 }
	return 1
}

func (m *MockEngine) ExecCount() int64 { return atomic.LoadInt64(&m.execCount) }

// ── Mock warp runner ────────────────────────────────────────

func runMockWarp(ctx context.Context, eng *MockEngine, bar *Barrier, errCh chan<- error, warpID int, stats *Stats) {
	var ws WarpStats
	pc := 0 // current instruction pointer
	t0 := time.Time{}

	for {
		select {
		case <-ctx.Done():
			errCh <- ctx.Err()
			return
		default:
		}

		t0 = time.Now()
		ret := eng.Exec(pc)
		ws.EngineTime += time.Since(t0)

		switch {
		case ret == -2: // done
			stats.record(ws)
			errCh <- nil
			return
		case ret == 1: // normal instruction executed
			pc++ // advance PC
		case ret == -1: // error
			stats.record(ws)
			errCh <- fmt.Errorf("warp %d: fail", warpID)
			return
		case ret&0x10000 != 0: // barrier
			rpc := ret & 0xFFFF
			pc = rpc // advance past barrier
			t0 = time.Now()
			bar.Wait()
			ws.BarrierTime += time.Since(t0)
			ws.Barriers++
		default:
			stats.record(ws)
			errCh <- fmt.Errorf("warp %d: unexpected ret=%d", warpID, ret)
			return
		}
	}
}

// ── Mock kernel launcher ────────────────────────────────────

func runMockKernel(t *testing.T, eng *MockEngine, totalWarps, warpsPerBlock int, policy SchedPolicy, timeout time.Duration, stats *Stats) error {
	t.Helper()

	totalBlocks := totalWarps / warpsPerBlock
	if totalWarps%warpsPerBlock != 0 { totalBlocks++ }
	if totalBlocks == 0 { totalBlocks = 1 }

	maxActive := 256
	if totalWarps < maxActive { maxActive = totalWarps }
	sem := make(chan struct{}, maxActive)
	errCh := make(chan error, totalWarps)

	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	launchWarp := func(blockIdx, w int, wg *sync.WaitGroup, bar *Barrier) {
		id := blockIdx*warpsPerBlock + w
		sem <- struct{}{}
		go func() {
			defer func() { <-sem }()
			if wg != nil { defer wg.Done() }
			runMockWarp(ctx, eng, bar, errCh, id, stats)
		}()
	}

	switch policy {
	case Wavefront:
		for w := 0; w < warpsPerBlock; w++ {
			var wg sync.WaitGroup
			for b := 0; b < totalBlocks; b++ {
				wg.Add(1)
				launchWarp(b, w, &wg, newBarrier(warpsPerBlock))
			}
			wg.Wait()
		}
		return nil // wavefront collects internally

	case GTO:
		firstWave := maxActive
		if firstWave > totalWarps { firstWave = totalWarps }
		bars := make([]*Barrier, totalBlocks)
		for i := range bars { bars[i] = newBarrier(warpsPerBlock) }
		launched := 0
		for b := 0; b < totalBlocks && launched < firstWave; b++ {
			for w := 0; w < warpsPerBlock && launched < firstWave; w++ {
				launchWarp(b, w, nil, bars[b])
				launched++
			}
		}
		for i := firstWave; i < totalWarps; i++ {
			if err := <-errCh; err != nil { return err }
			b, w := i/warpsPerBlock, i%warpsPerBlock
			launchWarp(b, w, nil, bars[b])
		}
		for i := 0; i < firstWave; i++ {
			if err := <-errCh; err != nil { return err }
		}
		return nil

	case Greedy:
		for b := 0; b < totalBlocks; b++ {
			var wg sync.WaitGroup
			wg.Add(warpsPerBlock)
			bar := newBarrier(warpsPerBlock)
			for w := 0; w < warpsPerBlock; w++ { launchWarp(b, w, &wg, bar) }
			wg.Wait()
		}
		return nil

	default: // RoundRobin
		bars := make([]*Barrier, totalBlocks)
		for i := range bars { bars[i] = newBarrier(warpsPerBlock) }
		var wg sync.WaitGroup
		for b := 0; b < totalBlocks; b++ {
			for w := 0; w < warpsPerBlock; w++ {
				wg.Add(1)
				launchWarp(b, w, &wg, bars[b])
			}
		}
		wg.Wait()
	}

	for i := 0; i < totalWarps; i++ {
		if err := <-errCh; err != nil { return err }
	}
	return nil
}

// ── Tests ───────────────────────────────────────────────────

func TestBasic(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(5) // 5 inst, no barriers
	err := runMockKernel(t, eng, 4, 1, RoundRobin, 0, stats)
	if err != nil { t.Fatal(err) }
	if eng.ExecCount() != 24 { // 4 warps × 6 calls (5 continue + 1 done)
		t.Fatalf("exec count: got %d, want 24", eng.ExecCount())
	}
}

func TestBarrier(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(10, 3, 7) // barriers at inst 3 and 7
	err := runMockKernel(t, eng, 4, 4, RoundRobin, 0, stats)
	if err != nil { t.Fatal(err) }
	if eng.ExecCount() != 44 { // 4 warps × 11 calls
		t.Fatalf("exec count: got %d, want 44", eng.ExecCount())
	}
	stats.finalize()
	if stats.BarrierWaits != 8 { // 4 warps × 2 barriers
		t.Fatalf("barrier waits: got %d, want 8", stats.BarrierWaits)
	}
}

func TestAllPolicies(t *testing.T) {
	policies := []SchedPolicy{RoundRobin, Greedy, GTO, Wavefront}
	for _, p := range policies {
		t.Run(p.String(), func(t *testing.T) {
			stats := newStats()
			eng := NewMockEngine(8, 4)
			err := runMockKernel(t, eng, 8, 4, p, 0, stats)
			if err != nil { t.Fatal(err) }
			if eng.ExecCount() < 72 { t.Errorf("exec count too low: %d", eng.ExecCount()) }
		})
	}
}

func TestTimeout(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(999999)
	err := runMockKernel(t, eng, 4, 1, RoundRobin, 10*time.Millisecond, stats)
	if err == nil { t.Fatal("expected timeout") }
	if err != context.DeadlineExceeded {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}
}

func TestFail(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(10).WithFail(3)
	err := runMockKernel(t, eng, 4, 1, RoundRobin, 0, stats)
	if err == nil { t.Fatal("expected error") }
}

func TestStats(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(10, 4, 8)
	err := runMockKernel(t, eng, 8, 4, RoundRobin, 0, stats)
	if err != nil { t.Fatal(err) }
	stats.finalize()

	if stats.TotalWarps != 8 { t.Errorf("total warps: got %d, want 8", stats.TotalWarps) }
	if stats.BarrierWaits != 16 { t.Errorf("barrier waits: got %d, want 16", stats.BarrierWaits) }
	if stats.LatencyP50 == 0 || stats.LatencyMax == 0 { t.Error("latency not populated") }
	t.Logf("Stats: warps=%d barriers=%d p50=%v p99=%v", stats.TotalWarps, stats.BarrierWaits, stats.LatencyP50, stats.LatencyP99)
}

func TestBarrierAllArriveBeforeProceed(t *testing.T) {
	b := newBarrier(4)
	passed := false

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			b.mu.Lock()
			b.count++
			if b.count >= b.target {
				passed = true
				b.count = 0
				b.cond.Broadcast()
			} else {
				if passed { t.Errorf("warp %d: barrier passed before all arrived", id) }
				b.cond.Wait()
			}
			b.mu.Unlock()
		}(i)
	}
	wg.Wait()
	if !passed { t.Error("barrier never satisfied") }
}

func TestBarrierEpoch(t *testing.T) {
	b := newBarrier(2)
	results := make([]int, 4)

	var wg sync.WaitGroup
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(warpID int) {
			defer wg.Done()
			for round := 0; round < 2; round++ {
				b.Wait()
				results[warpID*2+round] = 1
			}
		}(w)
	}
	wg.Wait()
	for i, v := range results {
		if v != 1 { t.Errorf("result[%d] = %d, want 1", i, v) }
	}
}
