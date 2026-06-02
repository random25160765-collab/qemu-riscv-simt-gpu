package main

/*
#cgo CFLAGS: -I.. -I../core -I../vpu -I../tcu -I../proto -I/usr/include/lua5.4
#cgo LDFLAGS: -L.. -lengine -lm -lrt -llua5.4 -ldl
#include <stdlib.h>
#include <string.h>
#include "../core/engine_bridge.h"
#include "../state.h"
#include "../config.h"
*/
import "C"

import (
	"context"
	"fmt"
	"os"
	"unsafe"
)

// sizeof constants — computed from Go side to avoid CGo static linkage issues
const (
	soaGprSz  = 32 * 32 * 4  // uint32_t[32*32]
	soaVprSz  = 32 * 32 * 4  // uint32_t[32*32] vector
	soaFprSz  = 32 * 32 * 4
	soaPcSz   = 32 * 4       // uint32_t[32]
	soaMmaSz  = 8 * 32 * 4   // float[8*32]
	simtStkSz = 32 * 8       // SIMTFrame[32] = {int32_t, uint32_t} × 32
)

// ── memory allocation ──────────────────────────────────────

// AllocGPGPUState allocates a zero-filled GPGPUState on the C heap.
func AllocGPGPUState() *C.GPGPUState {
	return (*C.GPGPUState)(C.calloc(1, C.bridge_sizeof_gpgpu_state()))
}

// FreeGPGPUState releases a GPGPUState and its VRAM.
func FreeGPGPUState(s *C.GPGPUState) {
	if s.vram_ptr != nil {
		C.free(unsafe.Pointer(s.vram_ptr))
	}
	C.free(unsafe.Pointer(s))
}

// AllocWarp allocates a zero-filled GPGPUWarp on the C heap.
func AllocWarp() *C.GPGPUWarp {
	return (*C.GPGPUWarp)(C.calloc(1, C.bridge_sizeof_gpgpu_warp()))
}

// AllocSoA returns freshly allocated SoA arrays for a single warp.
func AllocSoA() (gpr, fpr, vpr, pc, mh, fcsr, mma unsafe.Pointer, stk *C.SIMTFrame) {
	gpr  = C.calloc(1, soaGprSz)
	fpr  = C.calloc(1, soaFprSz)
	vpr  = C.calloc(1, soaVprSz)
	pc   = C.calloc(1, soaPcSz)
	mh   = C.calloc(1, soaPcSz)
	fcsr = C.calloc(1, soaPcSz)
	mma  = C.calloc(1, soaMmaSz)
	stk  = (*C.SIMTFrame)(C.calloc(1, simtStkSz))
	return
}

// FreeSoA releases all SoA arrays.
func FreeSoA(gpr, fpr, vpr, pc, mh, fcsr, mma unsafe.Pointer, stk *C.SIMTFrame) {
	C.free(gpr); C.free(fpr); C.free(vpr); C.free(pc)
	C.free(mh); C.free(fcsr); C.free(mma)
	C.free(unsafe.Pointer(stk))
}

// ── C function wrappers ────────────────────────────────────

func Predecode(s *C.GPGPUState, kernAddr uint32, kernSize uint32) (*C.ThOp, int) {
	var count C.int
	code := C.bridge_predecode(s, C.uint32_t(kernAddr), C.uint32_t(kernSize), &count)
	return code, int(count)
}

func FreeCode(code *C.ThOp) {
	C.free(unsafe.Pointer(code))
}

func AosToSoa(warp *C.GPGPUWarp,
	gpr, fpr, vpr, pc, mh, fcsr unsafe.Pointer) {
	C.bridge_aos_to_soa(warp,
		(*C.uint32_t)(gpr), (*C.uint32_t)(fpr), (*C.uint32_t)(vpr),
		(*C.uint32_t)(pc), (*C.uint32_t)(mh), (*C.uint32_t)(fcsr))
}

func SoaToAos(warp *C.GPGPUWarp,
	gpr, fpr, vpr, pc, mh, fcsr unsafe.Pointer) {
	C.bridge_soa_to_aos(warp,
		(*C.uint32_t)(gpr), (*C.uint32_t)(fpr), (*C.uint32_t)(vpr),
		(*C.uint32_t)(pc), (*C.uint32_t)(mh), (*C.uint32_t)(fcsr))
}

func InitWarp(warp *C.GPGPUWarp, kernAddr uint32, tidBase uint32,
	blockID [3]uint32, nThreads uint32, warpID uint32, blkLinear uint32) {
	var bid [3]C.uint32_t
	bid[0] = C.uint32_t(blockID[0])
	bid[1] = C.uint32_t(blockID[1])
	bid[2] = C.uint32_t(blockID[2])
	C.bridge_init_warp(warp, C.uint32_t(kernAddr), C.uint32_t(tidBase),
		&bid[0], C.uint32_t(nThreads), C.uint32_t(warpID), C.uint32_t(blkLinear))
}

// ExecWarp calls engine_exec once. Returns the raw C int.
func ExecWarp(code *C.ThOp, tcount int, ctx *C.EngineContext,
	gpr, fpr, vpr unsafe.Pointer, vl *C.uint32_t,
	pc, mh, fcsr unsafe.Pointer,
	stk *C.SIMTFrame, sdepth *C.int, resumePc int,
	mma unsafe.Pointer) int {
	return int(C.engine_exec(code, C.int(tcount), ctx,
		(*C.uint32_t)(gpr), (*C.uint32_t)(fpr), (*C.uint32_t)(vpr), vl,
		(*C.uint32_t)(pc), (*C.uint32_t)(mh), (*C.uint32_t)(fcsr),
		stk, sdepth, C.int(resumePc),
		(*C.float)(mma)))
}

func LoadConfig(cfg *C.vpu_config_t, path string) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	C.vpu_config_load(cfg, cpath)
}

// ── VRAM allocator helpers ─────────────────────────────────

const (
	PtrSlotA = C.PTR_SLOT_A
	PtrSlotB = C.PTR_SLOT_B
	PtrSlotC = C.PTR_SLOT_C
	PtrSlotD = C.PTR_SLOT_D
)

func VramAllocInit(s *C.GPGPUState)          { C.bridge_vram_alloc_init(s) }
func VramAllocReset(s *C.GPGPUState)         { C.bridge_vram_alloc_reset(s) }
func VramAlloc(s *C.GPGPUState, size int) uint32 { return uint32(C.bridge_vram_alloc(s, C.size_t(size))) }
func VramPtrRead(s *C.GPGPUState, slot int) uint32 { return uint32(C.bridge_vram_ptr_read(s, C.int(slot))) }
func VramPtrWrite(s *C.GPGPUState, slot int, addr uint32) { C.bridge_vram_ptr_write(s, C.int(slot), C.uint32_t(addr)) }
func VramWrite8(s *C.GPGPUState, addr uint32, data unsafe.Pointer, n int) {
	C.memcpy(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr))+uintptr(addr)), data, C.size_t(n))
}
func VramRead8(s *C.GPGPUState, addr uint32) unsafe.Pointer {
	return unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr)) + uintptr(addr))
}
func VramF32(s *C.GPGPUState, addr uint32, idx int) *float32 {
	return (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr)) + uintptr(addr) + uintptr(idx*4)))
}
func VramU32(s *C.GPGPUState, addr uint32, idx int) *uint32 {
	return (*uint32)(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr)) + uintptr(addr) + uintptr(idx*4)))
}
func VramU8(s *C.GPGPUState, addr uint32) *uint8 {
	return (*uint8)(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr)) + uintptr(addr)))
}

func KernSize(s *C.GPGPUState) uint32           { return uint32(s.kern_size) }
func SetKernSize(s *C.GPGPUState, sz uint32)    { s.kern_size = C.uint32_t(sz) }
func VramSize(s *C.GPGPUState) uint64            { return uint64(s.vram_size) }
func SetVramSize(s *C.GPGPUState, sz uint64)     { s.vram_size = C.uint64_t(sz) }
func VramPtr(s *C.GPGPUState) *C.uint8_t         { return s.vram_ptr }
func SetVramPtr(s *C.GPGPUState, p unsafe.Pointer) { s.vram_ptr = (*C.uint8_t)(p) }

// ── test helpers (needs CGo, must be in non-test file) ──────

const KernAddr = 0x500000

func GPUStateInit() *C.GPGPUState {
	s := AllocGPGPUState()
	s.warp_size = 32
	s.global_status = 1
	s.cfg.num_cus = 0
	s.cfg.warps_per_cu = 1
	SetVramSize(s, 16*1024*1024)
	SetVramPtr(s, C.calloc(1, C.ulong(VramSize(s))))
	if VramPtr(s) == nil {
		return nil
	}
	VramAllocInit(s)
	return s
}

func GPUStateLoadKernel(s *C.GPGPUState, path string) error {
	data, err := os.ReadFile(path)
	if err != nil || len(data) == 0 {
		return fmt.Errorf("load %s: %v", path, err)
	}
	C.memcpy(unsafe.Pointer(uintptr(unsafe.Pointer(VramPtr(s)))+KernAddr),
		unsafe.Pointer(&data[0]), C.size_t(len(data)))
	SetKernSize(s, uint32(len(data)))
	return nil
}

func GPUStateRun(s *C.GPGPUState, gx, gy, gz, bx, by, bz uint32) error {
	return RunKernel(s, KernelLaunch{
		KernAddr: KernAddr,
		GridDim:  [3]uint32{gx, gy, gz},
		BlockDim: [3]uint32{bx, by, bz},
	}, context.Background())
}

func AllocMap(s *C.GPGPUState, slot int, bytes int) uint32 {
	addr := VramAlloc(s, bytes)
	VramPtrWrite(s, slot, addr)
	return addr
}

func PtrRead(s *C.GPGPUState, slot int) uint32 { return VramPtrRead(s, slot) }

// ── helpers ────────────────────────────────────────────────

func MakeContext(s *C.GPGPUState, active uint32, shm unsafe.Pointer, shmSize uint32,
	tidBase uint32, blockID [3]uint32, warpID uint32) C.EngineContext {
	var ctx C.EngineContext
	ctx.s = s
	ctx.active = C.uint32_t(active)
	ctx.shm = (*C.uint8_t)(shm)
	ctx.shm_size = C.uint32_t(shmSize)
	ctx.thread_id[0] = C.uint32_t(tidBase)
	ctx.thread_id[1] = 0
	ctx.thread_id[2] = 0
	ctx.block_id[0] = C.uint32_t(blockID[0])
	ctx.block_id[1] = C.uint32_t(blockID[1])
	ctx.block_id[2] = C.uint32_t(blockID[2])
	ctx.warp_id = C.uint32_t(warpID)
	ctx.thread_mask = C.uint32_t(active)
	return ctx
}
