package mock
import (
	"context"
	"testing"
)
func TestSyncWarp(t *testing.T) {
	stats := newStats()
	eng := NewMockEngine(3) // 3 inst: pc goes 0,1,2,done
	errCh := make(chan error, 1)
	// run synchronously
	runMockWarp(context.Background(), eng, newBarrier(1), errCh, 0, stats)
	if err := <-errCh; err != nil { t.Fatal(err) }
	if eng.ExecCount() != 4 { t.Fatalf("exec: %d", eng.ExecCount()) }
	t.Log("ok")
}
