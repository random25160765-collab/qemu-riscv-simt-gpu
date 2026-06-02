package mock

import (
	"context"
	"testing"
	
)

func TestSingleWarp(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(5) // 5 inst, no bars
	bar := newBarrier(1)
	errCh := make(chan error, 1)

	go runMockWarp(context.Background(), eng, bar, errCh, 0, stats)

	err := <-errCh
	if err != nil { t.Fatal(err) }
	if eng.ExecCount() != 6 { t.Fatalf("exec count: %d", eng.ExecCount()) }
	t.Log("ok")
}

func TestFourWarpsOneBlock(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(10, 3, 7) // barriers at 3 and 7
	bar := newBarrier(4)
	errCh := make(chan error, 4)

	for i := 0; i < 4; i++ {
		go runMockWarp(context.Background(), eng, bar, errCh, i, stats)
	}

	for i := 0; i < 4; i++ {
		if err := <-errCh; err != nil { t.Fatal(err) }
	}
	if eng.ExecCount() != 44 { t.Fatalf("exec count: %d", eng.ExecCount()) }
	stats.finalize()
	if stats.BarrierWaits != 8 { t.Fatalf("barrier waits: %d", stats.BarrierWaits) }
	t.Log("ok")
}

func TestKernelRR(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(5)
	err := runMockKernel(t, eng, 4, 1, RoundRobin, 0, stats)
	if err != nil { t.Fatal(err) }
	t.Logf("exec=%d", eng.ExecCount())
}
