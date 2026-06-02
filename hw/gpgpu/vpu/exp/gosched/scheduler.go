package main

/*
#include <stdlib.h>
#include <string.h>
#include "../core/engine_bridge.h"
*/
import "C"

import (
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// ── SoA memory pool ────────────────────────────────────────

type soaSet struct {
	warp                                          *C.GPGPUWarp
	ctx                                           *C.EngineContext
	gpr, fpr, vpr, pc, mh, fcsr, mma unsafe.Pointer
	stk                                           *C.SIMTFrame
	sdepth                                        *C.int
}

const soaPoolSize = 256

var soaPool = make(chan *soaSet, soaPoolSize)

func init() {
	for i := 0; i < soaPoolSize; i++ {
		set := &soaSet{}
		set.warp = (*C.GPGPUWarp)(C.calloc(1, C.bridge_sizeof_gpgpu_warp()))
		set.ctx  = (*C.EngineContext)(C.calloc(1, C.ulong(unsafe.Sizeof(C.EngineContext{}))))
		set.gpr  = C.calloc(1, C.ulong(soaGprSz))
		set.fpr  = C.calloc(1, C.ulong(soaFprSz))
		set.vpr  = C.calloc(1, C.ulong(soaVprSz))
		set.pc   = C.calloc(1, C.ulong(soaPcSz))
		set.mh   = C.calloc(1, C.ulong(soaPcSz))
		set.fcsr = C.calloc(1, C.ulong(soaPcSz))
		set.mma  = C.calloc(1, C.ulong(soaMmaSz))
		set.stk  = (*C.SIMTFrame)(C.calloc(1, C.ulong(simtStkSz)))
		set.sdepth = (*C.int)(C.calloc(1, 4))
		soaPool <- set
	}
}

func soaGet() *soaSet  { return <-soaPool }
func soaPut(s *soaSet) { soaPool <- s }

// ── WarpSlot pool ──────────────────────────────────────────

type WarpSlot struct {
	gpr, fpr, vpr, pc, mh, fcsr, mma unsafe.Pointer
	stk                               *C.SIMTFrame
	sdepth                            C.int
	active                            uint32
	resumePc                          int
}

var slotPool = make(chan *WarpSlot, soaPoolSize*2)

func init() {
	for i := 0; i < soaPoolSize*2; i++ {
		s := &WarpSlot{}
		s.gpr  = C.calloc(1, C.ulong(soaGprSz))
		s.fpr  = C.calloc(1, C.ulong(soaFprSz))
		s.vpr  = C.calloc(1, C.ulong(soaVprSz))
		s.pc   = C.calloc(1, C.ulong(soaPcSz))
		s.mh   = C.calloc(1, C.ulong(soaPcSz))
		s.fcsr = C.calloc(1, C.ulong(soaPcSz))
		s.mma  = C.calloc(1, C.ulong(soaMmaSz))
		s.stk  = (*C.SIMTFrame)(C.calloc(1, C.ulong(simtStkSz)))
		slotPool <- s
	}
}

func slotGet() *WarpSlot  { return <-slotPool }
func slotPut(s *WarpSlot) { slotPool <- s }

func (s *WarpSlot) save(gpr, fpr, vpr, pc, mh, fcsr, mma unsafe.Pointer,
	stk *C.SIMTFrame, sdepth C.int, active uint32, rpc int) {
	C.memcpy(s.gpr, gpr, C.ulong(soaGprSz))
	C.memcpy(s.fpr, fpr, C.ulong(soaFprSz))
	C.memcpy(s.vpr, vpr, C.ulong(soaVprSz))
	C.memcpy(s.pc, pc, C.ulong(soaPcSz))
	C.memcpy(s.mh, mh, C.ulong(soaPcSz))
	C.memcpy(s.fcsr, fcsr, C.ulong(soaPcSz))
	C.memcpy(s.mma, mma, C.ulong(soaMmaSz))
	C.memcpy(unsafe.Pointer(s.stk), unsafe.Pointer(stk), C.ulong(simtStkSz))
	s.sdepth = sdepth
	s.active = active
	s.resumePc = rpc
}

func (s *WarpSlot) restore(gpr, fpr, vpr, pc, mh, fcsr, mma unsafe.Pointer,
	stk *C.SIMTFrame, sdepth *C.int, active *uint32, rpc *int) {
	C.memcpy(gpr, s.gpr, C.ulong(soaGprSz))
	C.memcpy(fpr, s.fpr, C.ulong(soaFprSz))
	C.memcpy(vpr, s.vpr, C.ulong(soaVprSz))
	C.memcpy(pc, s.pc, C.ulong(soaPcSz))
	C.memcpy(mh, s.mh, C.ulong(soaPcSz))
	C.memcpy(fcsr, s.fcsr, C.ulong(soaPcSz))
	C.memcpy(mma, s.mma, C.ulong(soaMmaSz))
	C.memcpy(unsafe.Pointer(stk), unsafe.Pointer(s.stk), C.ulong(simtStkSz))
	*sdepth = s.sdepth
	*active = s.active
	*rpc = s.resumePc
}

// ── Barrier ────────────────────────────────────────────────

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

// ── Warp runner ────────────────────────────────────────────

func runWarp(
	ctx context.Context,
	code *C.ThOp, tcount int,
	soa *soaSet,
	bar *Barrier,
	errCh chan<- error,
) {
	gpr, fpr, vpr, pc := soa.gpr, soa.fpr, soa.vpr, soa.pc
	mh, fcsr := soa.mh, soa.fcsr
	mma, stk := soa.mma, soa.stk
	eCtx, sdepth := soa.ctx, soa.sdepth
	vl := (*C.uint32_t)(C.calloc(1, 4))
	*vl = 32 // VL=32 by default
	defer C.free(unsafe.Pointer(vl))

	resumePc := C.int(-1)
	slot := slotGet()
	defer slotPut(slot)

	var ws WarpStats
	var t0 time.Time

	for {
		select {
		case <-ctx.Done():
			errCh <- ctx.Err()
			return
		default:
		}

		t0 = time.Now()
		ret := C.engine_exec(code, C.int(tcount), eCtx,
			(*C.uint32_t)(gpr), (*C.uint32_t)(fpr), (*C.uint32_t)(vpr), vl,
			(*C.uint32_t)(pc), (*C.uint32_t)(mh), (*C.uint32_t)(fcsr),
			stk, sdepth, resumePc, (*C.float)(mma))
		ws.EngineTime += time.Since(t0)

		if ret == 0 {
			RecordWarp(ws)
			errCh <- nil
			return
		}
		if ret < 0 {
			RecordWarp(ws)
			errCh <- fmt.Errorf("warp %d: illegal instruction", eCtx.warp_id)
			return
		}
		if ret&0x10000 != 0 {
			rpc := int(ret & 0xFFFF)
			slot.save(gpr, fpr, vpr, pc, mh, fcsr, mma, stk, *sdepth, uint32(eCtx.active), rpc)

			t0 = time.Now()
			bar.Wait()
			ws.BarrierTime += time.Since(t0)
			ws.Barriers++

			var active uint32
			slot.restore(gpr, fpr, vpr, pc, mh, fcsr, mma, stk, sdepth, &active, &rpc)
			eCtx.active = C.uint32_t(active)
			resumePc = C.int(rpc)
			continue
		}
		RecordWarp(ws)
		errCh <- fmt.Errorf("warp %d: unexpected return %d", eCtx.warp_id, ret)
		return
	}
}

// ── Scheduling policies ───────────────────────────────────

type SchedPolicy int

const (
	SchedRoundRobin SchedPolicy = iota // fair: semaphore-limited concurrency
	SchedGreedy                        // blocks sequentially, warp-parallel per block
	SchedGTO                           // Greedy-Then-Oldest: first wave greedy, rest FIFO
	SchedWavefront                     // one warp per block per wave, round-robin across blocks
)

func (p SchedPolicy) String() string {
	switch p {
	case SchedRoundRobin: return "rr"
	case SchedGreedy: return "greedy"
	case SchedGTO: return "gto"
	case SchedWavefront: return "wave"
	default: return "??"
	}
}

// ── Kernel launch descriptor ──────────────────────────────

type KernelLaunch struct {
	KernAddr uint32
	GridDim  [3]uint32
	BlockDim [3]uint32
	ShmSize  uint32
	Policy   SchedPolicy
	Timeout  time.Duration
}

// ── Kernel runner ──────────────────────────────────────────

func RunKernel(s *C.GPGPUState, launch KernelLaunch, ctx context.Context) error {
	s.kernel.kernel_addr = C.uint64_t(launch.KernAddr)
	s.kernel.grid_dim[0] = C.uint32_t(launch.GridDim[0])
	s.kernel.grid_dim[1] = C.uint32_t(launch.GridDim[1])
	s.kernel.grid_dim[2] = C.uint32_t(launch.GridDim[2])
	s.kernel.block_dim[0] = C.uint32_t(launch.BlockDim[0])
	s.kernel.block_dim[1] = C.uint32_t(launch.BlockDim[1])
	s.kernel.block_dim[2] = C.uint32_t(launch.BlockDim[2])

	kernSize := uint32(s.kern_size)
	if kernSize == 0 { kernSize = 4096 }

	code, tcount := Predecode(s, launch.KernAddr, kernSize)
	if code == nil { return fmt.Errorf("predecode failed") }
	defer FreeCode(code)

	tpb := launch.BlockDim[0] * launch.BlockDim[1] * launch.BlockDim[2]
	warpsPerBlock := int((tpb + 31) / 32)
	totalBlocks := int(launch.GridDim[0] * launch.GridDim[1] * launch.GridDim[2])
	totalWarps := totalBlocks * warpsPerBlock

	maxActive := soaPoolSize
	if totalWarps < maxActive { maxActive = totalWarps }
	sem := make(chan struct{}, maxActive)
	errCh := make(chan error, totalWarps)

	// ── collect block descriptors ──
	type blockDesc struct {
		id        [3]uint32
		linear    uint32
		shm       unsafe.Pointer
	}
	blocks := make([]blockDesc, totalBlocks)
	idx := 0
	for z := uint32(0); z < launch.GridDim[2]; z++ {
		for y := uint32(0); y < launch.GridDim[1]; y++ {
			for x := uint32(0); x < launch.GridDim[0]; x++ {
				var shm unsafe.Pointer
				if launch.ShmSize > 0 {
					shm = C.calloc(1, C.ulong(launch.ShmSize))
				}
				blocks[idx] = blockDesc{
					id:     [3]uint32{x, y, z},
					linear: z*launch.GridDim[0]*launch.GridDim[1] + y*launch.GridDim[0] + x,
					shm:    shm,
				}
				idx++
			}
		}
	}

	// ── helper: launch one warp ──
	launchWarp := func(b blockDesc, w int, wg *sync.WaitGroup, bar *Barrier) {
		tidBase := uint32(w * 32)
		nThr := tpb - tidBase
		if nThr > 32 { nThr = 32 }

		sem <- struct{}{}
		go func(w int, tidBase, nThr uint32) {
			defer func() { <-sem }()
			if wg != nil { defer wg.Done() }

			soa := soaGet()
			defer soaPut(soa)

			InitWarp(soa.warp, launch.KernAddr, tidBase, b.id, nThr, uint32(w), b.linear)
			AosToSoa(soa.warp, soa.gpr, soa.fpr, soa.vpr, soa.pc, soa.mh, soa.fcsr)
			*soa.sdepth = 0
			*soa.ctx = MakeContext(s, uint32(soa.warp.active_mask), b.shm, launch.ShmSize,
				tidBase, b.id, uint32(w))

			runWarp(ctx, code, tcount, soa, bar, errCh)

			SoaToAos(soa.warp, soa.gpr, soa.fpr, soa.vpr, soa.pc, soa.mh, soa.fcsr)
		}(w, tidBase, nThr)
	}

	switch launch.Policy {

	case SchedWavefront:
		// 每轮: 所有 block 的 warp w 同时执行, barrier 同步后进入 warp w+1
		for w := 0; w < warpsPerBlock; w++ {
			var wg sync.WaitGroup
			bars := make([]*Barrier, totalBlocks)
			for bi := range blocks {
				bars[bi] = newBarrier(warpsPerBlock)
			}
			for bi, b := range blocks {
				wg.Add(1)
				launchWarp(b, w, &wg, bars[bi])
			}
			wg.Wait()
		}
		return nil // Wavefront collects internally, skip outer loop

	case SchedGTO:
		// Greedy phase: semaphore 满 → 第一批 warps 并发
		// Oldest phase: 剩余 warps 按 warp ID 升序 (oldest=smallest ID)
		bars := make([]*Barrier, totalBlocks)
		for bi := range bars { bars[bi] = newBarrier(warpsPerBlock) }

		firstWave := maxActive
		if firstWave > totalWarps { firstWave = totalWarps }

		// Phase 1: greedy — 前 firstWave 个 warp 并发
		launched := 0
		for bi := range blocks {
			for w := 0; w < warpsPerBlock && launched < firstWave; w++ {
				launchWarp(blocks[bi], w, nil, bars[bi])
				launched++
			}
		}
		// Phase 2: oldest-first — 等一个完成 → 发射下一个 oldest
		for i := firstWave; i < totalWarps; i++ {
			if err := <-errCh; err != nil { return err }
			bi := i / warpsPerBlock
			w := i % warpsPerBlock
			launchWarp(blocks[bi], w, nil, bars[bi])
		}
		// drain remaining firstWave results
		for i := 0; i < firstWave; i++ {
			if err := <-errCh; err != nil { return err }
		}
		return nil // GTO collects internally, skip outer loop

	case SchedGreedy:
		// 每个 block 内部 warp 并行, block 间串行
		for _, b := range blocks {
			var wg sync.WaitGroup
			wg.Add(warpsPerBlock)
			bar := newBarrier(warpsPerBlock)
			for w := 0; w < warpsPerBlock; w++ {
				launchWarp(b, w, &wg, bar)
			}
			wg.Wait()
			if b.shm != nil { C.free(b.shm) }
		}

	default: // SchedRoundRobin
		// 所有 warps 并发, semaphore 限流
		bars := make([]*Barrier, totalBlocks)
		for bi := range bars { bars[bi] = newBarrier(warpsPerBlock) }
		var shmWg sync.WaitGroup
		for _, b := range blocks {
			shmWg.Add(warpsPerBlock)
			for w := 0; w < warpsPerBlock; w++ {
				launchWarp(b, w, &shmWg, bars[b.linear%uint32(totalBlocks)])
			}
		}
		// cleanup shm when per-block warps are done
		go func() {
			shmWg.Wait()
			for _, b := range blocks {
				if b.shm != nil { C.free(b.shm) }
			}
		}()
	}

	// collect results
	for i := 0; i < totalWarps; i++ {
		if err := <-errCh; err != nil {
			return err
		}
	}
	return nil
}
