package main
import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func TestDebugScal(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	A := AllocMap(s, PtrSlotA, 4)
	Alpha := AllocMap(s, PtrSlotD, 4)
	AllocMap(s, PtrSlotB, 4)
	*VramF32(s, A, 0) = 4.0
	*VramF32(s, Alpha, 0) = 0.75

	if err := GPUStateLoadKernel(s, "../kernels/scal_mul.bin"); err != nil {
		t.Fatal(err)
	}

	// Dump everything pre-run
	fmt.Printf("[pre-run] ptrs: A=0x%x B=0x%x C=0x%x D=0x%x\n",
		VramPtrRead(s,0), VramPtrRead(s,1), VramPtrRead(s,2), VramPtrRead(s,3))
	fmt.Printf("[pre-run] data: A[0]=%.4f D[0]=%.4f B[0]=%.4f\n",
		*VramF32(s, VramPtrRead(s,0), 0),
		*VramF32(s, VramPtrRead(s,3), 0),
		*VramF32(s, VramPtrRead(s,1), 0))
	fmt.Printf("[pre-run] kernel@0x500000: inst[0]=0x%08x inst[1]=0x%08x\n",
		*(*uint32)(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr)) + 0x500000)),
		*(*uint32)(unsafe.Pointer(uintptr(unsafe.Pointer(s.vram_ptr)) + 0x500004)))
	fmt.Printf("[pre-run] tcount=%d kernSize=%d\n", s.kern_size, s.kern_size)

	if err := GPUStateRun(s, 1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	// Dump post-run
	fmt.Printf("[post-run] B[0]=%.4f (expect 3.0)\n", *VramF32(s, VramPtrRead(s,1), 0))

	result := *VramF32(s, PtrRead(s, PtrSlotB), 0)
	if math.Abs(float64(result-3.0)) > 1e-5 {
		t.Fatalf("got %.4f exp 3.0", result)
	}
}
