package main

/*
#include <stdlib.h>
#include <string.h>
#include "../state.h"
*/
import "C"

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"os"
	"time"
	"unsafe"
)

func init() {
	go func() { http.ListenAndServe(":6060", nil) }()
}

func main() {
	policyName := flag.String("policy", "rr", "scheduling policy: rr|greedy|gto|wave")
	timeoutMs := flag.Int("timeout", 0, "warp timeout in ms (0=disabled)")
	sbFlag := flag.Bool("scoreboard", false, "enable register scoreboard hazard detection")
	flag.Parse()

	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <kernel.bin> [grid x y z block x y z]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "  pprof: http://localhost:6060/debug/pprof/\n")
		os.Exit(1)
	}

	policy := parsePolicy(*policyName)
	timeout := time.Duration(*timeoutMs) * time.Millisecond

	args := flag.Args()
	kernPath := args[0]
	gridX, gridY, gridZ := uint32(1), uint32(1), uint32(1)
	blockX, blockY, blockZ := uint32(1), uint32(1), uint32(1)

	if len(args) >= 7 {
		gridX = parseUint(args[1]); gridY = parseUint(args[2]); gridZ = parseUint(args[3])
		blockX = parseUint(args[4]); blockY = parseUint(args[5]); blockZ = parseUint(args[6])
	}

	s := AllocGPGPUState()
	defer FreeGPGPUState(s)
	s.warp_size = 32
	s.global_status = 1

	cfgPath := C.CString("gpu_config.lua")
	LoadConfig(&s.cfg, C.GoString(cfgPath))
	C.free(unsafe.Pointer(cfgPath))

	s.vram_size = C.uint64_t(s.cfg.vram_mb) * 1024 * 1024
	s.vram_ptr = (*C.uint8_t)(C.calloc(1, C.ulong(s.vram_size)))
	if s.vram_ptr == nil {
		fmt.Fprintln(os.Stderr, "VRAM alloc failed")
		os.Exit(1)
	}

	fmt.Printf("VRAM: %d MB | %d CU x %d warps/CU x %d lanes/warp | policy=%s timeout=%v sb=%v\n\n",
		s.cfg.vram_mb, s.cfg.num_cus, s.cfg.warps_per_cu, s.warp_size, policy, timeout, *sbFlag)

	data, err := os.ReadFile(kernPath)
	if err != nil || len(data) == 0 {
		fmt.Fprintf(os.Stderr, "Failed to load kernel: %s (%v)\n", kernPath, err)
		os.Exit(1)
	}
	kernAddr := uint32(0x500000)
	C.memcpy(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr))+uintptr(kernAddr)),
		unsafe.Pointer(&data[0]), C.size_t(len(data)))
	s.kern_size = C.uint32_t(len(data))

	launch := KernelLaunch{
		KernAddr:   kernAddr,
		GridDim:    [3]uint32{gridX, gridY, gridZ},
		BlockDim:   [3]uint32{blockX, blockY, blockZ},
		Policy:     policy,
		Timeout:    timeout,
		Scoreboard: *sbFlag,
	}

	fmt.Printf("Launch: grid=(%d,%d,%d) block=(%d,%d,%d)\n",
		gridX, gridY, gridZ, blockX, blockY, blockZ)

	t0 := time.Now()
	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	err = RunKernel(s, launch, ctx)
	elapsed := time.Since(t0)

	if err != nil {
		fmt.Fprintf(os.Stderr, "Kernel error: %v\n", err)
		os.Exit(1)
	}

	stats := GetStats()
	fmt.Printf("\nDone in %v\n", elapsed.Round(time.Microsecond))
	fmt.Printf("Warps: %d total | Engine time: %v avg %v/warp\n",
		stats.TotalWarps, stats.EngineTime.Round(time.Microsecond),
		stats.EngineTime/time.Duration(stats.TotalWarps))
	fmt.Printf("Latency: p50=%v p95=%v p99=%v max=%v\n",
		stats.LatencyP50.Round(time.Microsecond),
		stats.LatencyP95.Round(time.Microsecond),
		stats.LatencyP99.Round(time.Microsecond),
		stats.LatencyMax.Round(time.Microsecond))
	if stats.BarrierWaits > 0 {
		fmt.Printf("Barriers: %d waits, avg wait %v\n",
			stats.BarrierWaits, stats.BarrierTime/time.Duration(stats.BarrierWaits))
	}
	if stats.ScoreboardStalls > 0 {
		fmt.Printf("Scoreboard: %d stalls (%.1f per warp)\n",
			stats.ScoreboardStalls, float64(stats.ScoreboardStalls)/float64(stats.TotalWarps))
	}
}

func parsePolicy(s string) SchedPolicy {
	switch s {
	case "greedy": return SchedGreedy
	case "gto": return SchedGTO
	case "wave": return SchedWavefront
	default: return SchedRoundRobin
	}
}

func parseUint(s string) uint32 {
	var v uint32
	fmt.Sscanf(s, "%d", &v)
	return v
}
