package poolmgr

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// syscall numbers for eventfd (not in stdlib syscall)
const (
	sysEventfd2      = 290                // SYS_EVENTFD2 on x86_64
	efdCloExec       = 0o2000000          // EFD_CLOEXEC
	efdNonBlock      = 0o4000             // EFD_NONBLOCK
	efdSemaphore     = 0o1                // EFD_SEMAPHORE
)

func eventfd(initval uint, flags int) (int, error) {
	r1, _, e := syscall.Syscall(sysEventfd2, uintptr(initval), uintptr(flags), 0)
	if e != 0 { return -1, e }
	return int(r1), nil
}

// protocol constants from iface.h
const (
	vpuCmdNop      uint32 = 0
	vpuCmdRegWrite uint32 = 1
	vpuCmdDispatch uint32 = 3
	vpuCmdReset    uint32 = 4

	vpuCtrlSize = 16 // sizeof(VPUCtrl)

	vpuEnvDoorbell = "VPU_DOORBELL_FD"
	vpuEnvComplete = "VPU_COMPLETE_FD"
	vpuEnvError    = "VPU_ERROR_FD"
	vpuEnvVramSize = "VPU_VRAM_SIZE"
)

// ── VPU control block (matches VPUCtrl in main.c) ──────────

type vpuCtrl struct {
	cmd  uint32
	data [3]uint32
}

func (c *vpuCtrl) asPtr() uintptr { return uintptr(unsafe.Pointer(c)) }

// ── VPUInstance ────────────────────────────────────────────

type VPUInstance struct {
	ID     int
	Status string // idle, busy, error

	vram []byte // mmap'd VRAM shm (via /dev/shm file)
	ctrl []byte // mmap'd ctrl shm

	vramFile *os.File
	ctrlFile *os.File

	doorbellFD int
	completeFD int
	errorFD    int

	cmd *exec.Cmd
	mu  sync.Mutex
}

// NewVPU creates a VPU instance using /dev/shm files as shared memory.
func NewVPU(id int, vramMB int, daemonPath string) (*VPUInstance, error) {
	v := &VPUInstance{ID: id, Status: "init"}

	vramSize := vramMB * 1024 * 1024

	// 1. Create VRAM shared memory via /dev/shm file + mmap
	var err error
	v.vramFile, v.vram, err = createShmFile("vpu_vram", id, vramSize)
	if err != nil {
		return nil, fmt.Errorf("vram shm: %w", err)
	}

	// 2. Create CTRL shared memory
	v.ctrlFile, v.ctrl, err = createShmFile("vpu_ctrl", id, vpuCtrlSize)
	if err != nil {
		v.closeShm()
		return nil, fmt.Errorf("ctrl shm: %w", err)
	}

	// 3. Create eventfds
	v.doorbellFD, err = eventfd(0, efdCloExec|efdNonBlock)
	if err != nil {
		v.closeShm()
		return nil, fmt.Errorf("doorbell eventfd: %w", err)
	}
	v.completeFD, err = eventfd(0, efdCloExec)
	if err != nil {
		v.closeShm(); v.closeEventfds()
		return nil, fmt.Errorf("complete eventfd: %w", err)
	}
	v.errorFD, err = eventfd(0, efdCloExec)
	if err != nil {
		v.closeShm(); v.closeEventfds()
		return nil, fmt.Errorf("error eventfd: %w", err)
	}

	// 4. Fork VPU daemon with eventfd FDs passed via ExtraFiles
	v.cmd = exec.Command(daemonPath)
	v.cmd.Env = append(os.Environ(),
		fmt.Sprintf("%s=%d", vpuEnvDoorbell, v.doorbellFD),
		fmt.Sprintf("%s=%d", vpuEnvComplete, v.completeFD),
		fmt.Sprintf("%s=%d", vpuEnvError, v.errorFD),
		fmt.Sprintf("%s=%d", vpuEnvVramSize, vramSize),
	)
	v.cmd.ExtraFiles = []*os.File{
		os.NewFile(uintptr(v.doorbellFD), "doorbell"),
		os.NewFile(uintptr(v.completeFD), "complete"),
		os.NewFile(uintptr(v.errorFD), "error"),
	}
	v.cmd.Stdout = os.Stdout
	v.cmd.Stderr = os.Stderr

	if err := v.cmd.Start(); err != nil {
		v.closeShm()
		v.closeEventfds()
		return nil, fmt.Errorf("start daemon: %w", err)
	}

	v.Status = "idle"
	return v, nil
}

// ── shm helpers ─────────────────────────────────────────────

func createShmFile(prefix string, id, size int) (*os.File, []byte, error) {
	name := fmt.Sprintf("/dev/shm/%s_%d_%d", prefix, os.Getpid(), id)
	f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, nil, err
	}
	os.Remove(name) // unlink immediately — stays alive while mmap'd

	if err := syscall.Ftruncate(int(f.Fd()), int64(size)); err != nil {
		f.Close()
		return nil, nil, err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, size,
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	return f, data, nil
}

func (v *VPUInstance) closeShm() {
	if v.vram != nil {
		syscall.Munmap(v.vram)
		v.vram = nil
	}
	if v.ctrl != nil {
		syscall.Munmap(v.ctrl)
		v.ctrl = nil
	}
	if v.vramFile != nil {
		v.vramFile.Close()
		v.vramFile = nil
	}
	if v.ctrlFile != nil {
		v.ctrlFile.Close()
		v.ctrlFile = nil
	}
}

func (v *VPUInstance) closeEventfds() {
	if v.doorbellFD >= 0 {
		syscall.Close(v.doorbellFD)
		v.doorbellFD = -1
	}
	if v.completeFD >= 0 {
		syscall.Close(v.completeFD)
		v.completeFD = -1
	}
	if v.errorFD >= 0 {
		syscall.Close(v.errorFD)
		v.errorFD = -1
	}
}

// ── Dispatch ────────────────────────────────────────────────

func (v *VPUInstance) Dispatch(kernData []byte, kernAddr uint32,
	grid, block [3]uint32) (int32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.Status != "idle" {
		return -1, fmt.Errorf("vpu %d: not idle (%s)", v.ID, v.Status)
	}
	v.Status = "busy"

	// 1. Write kernel to VRAM (if provided)
	if kernData != nil {
		copy(v.vram[kernAddr:], kernData)
	}

	// 2. Set kernel params via REG_WRITE
	v.regWrite(0x0300, kernAddr) // KERNEL_ADDR_LO
	v.regWrite(0x0310, grid[0])  // GRID_DIM_X
	v.regWrite(0x0314, grid[1])  // GRID_DIM_Y
	v.regWrite(0x0318, grid[2])  // GRID_DIM_Z
	v.regWrite(0x031C, block[0]) // BLOCK_DIM_X
	v.regWrite(0x0320, block[1]) // BLOCK_DIM_Y
	v.regWrite(0x0324, block[2]) // BLOCK_DIM_Z

	// 3. Send DISPATCH
	return v.sendDispatch()
}

func (v *VPUInstance) regWrite(offset, value uint32) {
	ctrl := (*vpuCtrl)(unsafe.Pointer(&v.ctrl[0]))
	ctrl.data[0] = offset
	ctrl.data[1] = value
	ctrl.cmd = vpuCmdRegWrite
	v.doorbell()
	v.waitCmdDone(ctrl)
}

func (v *VPUInstance) sendDispatch() (int32, error) {
	ctrl := (*vpuCtrl)(unsafe.Pointer(&v.ctrl[0]))
	ctrl.data[0] = 0
	ctrl.cmd = vpuCmdDispatch

	v.doorbell()

	// block until complete or error eventfd fires
	maxFD := v.completeFD
	if v.errorFD > maxFD { maxFD = v.errorFD }
	rfds := &syscall.FdSet{}
	fdBitset(rfds, v.completeFD)
	fdBitset(rfds, v.errorFD)
	_, err := syscall.Select(maxFD+1, rfds, nil, nil, nil)
	if err != nil {
		v.Status = "error"
		return -1, fmt.Errorf("select: %w", err)
	}

	if fdIsSet(rfds, v.errorFD) {
		drainEventfd(v.errorFD)
		v.Status = "error"
		return -1, fmt.Errorf("vpu %d: error signaled", v.ID)
	}

	drainEventfd(v.completeFD)

	ret := int32(ctrl.data[0])
	v.Status = "idle"
	return ret, nil
}

func (v *VPUInstance) doorbell() {
	var val uint64 = 1
	syscall.Write(v.doorbellFD, (*[8]byte)(unsafe.Pointer(&val))[:])
}

// fd set helpers
func fdBitset(set *syscall.FdSet, fd int) {
	set.Bits[fd/64] |= 1 << (uint(fd) % 64)
}
func fdIsSet(set *syscall.FdSet, fd int) bool {
	return set.Bits[fd/64]&(1<<(uint(fd)%64)) != 0
}
func drainEventfd(fd int) {
	var buf [8]byte
	syscall.Read(fd, buf[:])
}

func (v *VPUInstance) waitCmdDone(ctrl *vpuCtrl) {
	for ctrl.cmd != vpuCmdNop {
		runtime.Gosched()
	}
}

// ── Close ──────────────────────────────────────────────────

func (v *VPUInstance) Close() {
	if v.cmd != nil && v.cmd.Process != nil {
		v.cmd.Process.Signal(syscall.SIGTERM)
		v.cmd.Wait()
	}
	v.closeEventfds()
	v.closeShm()
}
