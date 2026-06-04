package main

import (
	"math"
	"testing"
	"unsafe"
)

// ── func tests ──────────────────────────────────────────────

func TestVecmul(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	// Match C test order: alloc data first, then load kernel
	A := AllocMap(s,PtrSlotA, 2048*4)
	B := AllocMap(s,PtrSlotB, 2048*4)
	Caddr := AllocMap(s,PtrSlotC, 2048*4)
	for i := 0; i < 2048; i++ {
		*VramF32(s, A, i) = float32(i + 1)
		*VramF32(s, B, i) = 2.0
	}

	if err := GPUStateLoadKernel(s, "../kernels/vecmul.bin"); err != nil {
		t.Fatal(err)
	}

	if err := GPUStateRun(s,1, 1, 1, 2048, 1, 1); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2048; i++ {
		got := *VramF32(s, Caddr, i)
		exp := float32(i+1) * 2.0
		if math.Abs(float64(got-exp)) > 1e-5 {
			t.Fatalf("[%d] got %.6f exp %.6f", i, got, exp)
		}
	}
}

func TestSaxpy(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	N := uint32(16384)
	*VramU32(s, 0, 0) = N
	*VramF32(s, 4, 0) = 2.5

	Xbase := AllocMap(s,PtrSlotA, int(N)*4)
	Ybase := AllocMap(s,PtrSlotB, int(N)*4)

	// simple LCG for deterministic "random"
	rng := uint32(42)
	randF := func() float32 {
		rng = rng*1103515245 + 12345
		return float32(int32(rng&0x7FFFFFFF)) / float32(0x7FFFFFFF)*100 - 50
	}
	x := make([]float32, N)
	y := make([]float32, N)
	for i := uint32(0); i < N; i++ {
		x[i] = randF()
		y[i] = randF()
	}
	VramWrite8(s, Xbase, unsafe.Pointer(&x[0]), int(N)*4)
	VramWrite8(s, Ybase, unsafe.Pointer(&y[0]), int(N)*4)

	GPUStateLoadKernel(s, "../kernels/saxpy.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	Yout := PtrRead(s,PtrSlotB)
	errors := 0
	for i := uint32(0); i < N && errors < 10; i++ {
		got := *VramF32(s, Yout, int(i))
		exp := 2.5*x[i] + y[i]
		if math.Abs(float64(got-exp)) > 1e-4*math.Abs(float64(exp)) {
			errors++
		}
	}
	if errors > 0 {
		t.Fatalf("%d errors", errors)
	}
}

func TestRV32M(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	pairs := [][2]int32{{10, 3}, {10, -3}, {-5, -2}, {100, 0}, {int32(-2147483648), 2}, {-1, -1}}
	*VramU32(s, 0, 0) = 6
	In := AllocMap(s,PtrSlotA, len(pairs)*8)
	AllocMap(s,PtrSlotC, 6*32)
	for i, p := range pairs {
		*VramU32(s, In, i*2) = uint32(p[0])
		*VramU32(s, In, i*2+1) = uint32(p[1])
	}

	GPUStateLoadKernel(s, "../kernels/rv32m.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	Out := PtrRead(s,PtrSlotC)
	for i, p := range pairs {
		a, b := p[0], p[1]
		o := (*[8]uint32)(unsafe.Pointer(VramU8(s, Out + uint32(i*32))))
		ex := [8]uint32{
			uint32(a * b),
			uint32((int64(a) * int64(b)) >> 32),
			uint32((int64(a) * int64(uint32(b))) >> 32),
			uint32((uint64(uint32(a)) * uint64(uint32(b))) >> 32),
			divS32(a,b),
			divU32(a,b),
			remS32(a,b),
			remU32(a,b),
		}
		for j := 0; j < 8; j++ {
			if o[j] != ex[j] {
				t.Fatalf("case %d[%d]: got 0x%x exp 0x%x", i, j, o[j], ex[j])
			}
		}
	}
}


func TestRV32F(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	fa, fb, fc := float32(3), float32(2), float32(4)
	i32 := int32(-5)
	u32 := uint32(7)
	In := AllocMap(s,PtrSlotA, 20)
	AllocMap(s,PtrSlotC, 26*4)
	*VramF32(s, In, 0) = fa
	*VramF32(s, In, 1) = fb
	*VramF32(s, In, 2) = fc
	*VramU32(s, In, 3) = uint32(i32)
	*VramU32(s, In, 4) = u32

	GPUStateLoadKernel(s, "../kernels/rv32f.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	Out := PtrRead(s,PtrSlotC)
	cf := func(idx int, exp float32) {
		got := *VramF32(s, Out, idx)
		if math.Abs(float64(got-exp)) > 1e-4*math.Abs(float64(exp)) && math.Abs(float64(got-exp)) > 1e-5 {
			t.Errorf("fo[%d]: got %.6f exp %.6f", idx, got, exp)
		}
	}
	ci := func(idx int, exp uint32) {
		if got := *VramU32(s, Out, idx); got != exp {
			t.Errorf("io[%d]: got 0x%x exp 0x%x", idx, got, exp)
		}
	}
	cf(0, fa+fb); cf(1, fa-fb); cf(2, fa*fb); cf(3, fa/fb)
	cf(4, float32(math.Sqrt(float64(fa))))
	cf(5, fa*fb+fc); cf(6, fa*fb-fc); cf(7, -(fa*fb-fc)); cf(8, -(fa*fb+fc))
	cf(9, 3); cf(10, -3); cf(11, 3); cf(12, 2); cf(13, 3)
	ci(14, 0); ci(15, 0); ci(16, 1); ci(17, 1)
	ci(18, 3); ci(19, 3)
	cf(20, -5); cf(21, 7)
	ci(22, 7); ci(23, 0x40400000); ci(24, 0x40); ci(25, 0x02)
}

func TestMemAccess(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	in := []uint8{0x7F, 0x80, 0xFF, 0x00, 0x34, 0x12, 0x78, 0x56}
	In := AllocMap(s,PtrSlotA, len(in))
	AllocMap(s,PtrSlotC, 12*4)
	VramWrite8(s, In, unsafe.Pointer(&in[0]), len(in))

	GPUStateLoadKernel(s, "../kernels/mem_access.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	Out := PtrRead(s,PtrSlotC)
	exp := [12]uint32{127, uint32(0xFFFFFF80), 0xFFFFFFFF, 128, 255, 0x1234, 0x5678, 0x1234, 0x00FF807F, 0x42, 0x4321, 0x00FF807F}
	for i, e := range exp {
		if got := *VramU32(s, Out, i); got != e {
			t.Errorf("[%d] got 0x%x exp 0x%x", i, got, e)
		}
	}
}

func TestMatmul(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	*VramU32(s, 0, 0) = 2
	A := AllocMap(s,PtrSlotA, 4)
	B := AllocMap(s,PtrSlotB, 4)
	AllocMap(s,PtrSlotC, 4)
	*VramF32(s, A, 0) = 1; *VramF32(s, A, 1) = 2
	*VramF32(s, B, 0) = 3; *VramF32(s, B, 1) = 4

	GPUStateLoadKernel(s, "../kernels/matmul.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	result := *VramF32(s, PtrRead(s,PtrSlotC), 0)
	if math.Abs(float64(result-11)) > 1e-3 {
		t.Fatalf("got %.3f exp 11", result)
	}
}

func TestDotProduct(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	N := uint32(32768)
	*VramU32(s, 0, 0) = N
	A := AllocMap(s,PtrSlotA, int(N)*4)
	B := AllocMap(s,PtrSlotB, int(N)*4)
	for i := uint32(0); i < N; i++ {
		*VramF32(s, A, int(i)) = float32(i%100) * 0.01
		*VramF32(s, B, int(i)) = float32((i+1)%100) * 0.01
	}

	GPUStateLoadKernel(s, "../kernels/dot_product.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	var sum float32
	for i := uint32(0); i < N; i++ {
		sum += float32(i%100) * 0.01 * float32((i+1)%100) * 0.01
	}
	result := *VramF32(s, 4, 0)
	if math.Abs(float64(result-sum)) > 0.01 {
		t.Fatalf("got %.4f exp %.4f", result, sum)
	}
}

func TestMemcpy(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	N := uint32(131072)
	*VramU32(s, 0, 0) = N
	A := AllocMap(s,PtrSlotA, int(N)*4)
	AllocMap(s,PtrSlotB, int(N)*4)
	for i := uint32(0); i < N; i++ {
		*VramU32(s, A, int(i)) = i
	}

	GPUStateLoadKernel(s, "../kernels/memcpy.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	Bout := PtrRead(s,PtrSlotB)
	for i := uint32(0); i < N; i++ {
		if *VramU32(s, Bout, int(i)) != i {
			t.Fatalf("[%d] mismatch", i)
		}
	}
}

func TestScalMul(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	// C test: PTR_SLOT_A=input, PTR_SLOT_D=alpha, PTR_SLOT_B=output
	A := AllocMap(s,PtrSlotA, 4)
	Alpha := AllocMap(s,PtrSlotD, 4)
	AllocMap(s,PtrSlotB, 4)
	*VramF32(s, A, 0) = 4.0
	*VramF32(s, Alpha, 0) = 0.75

	GPUStateLoadKernel(s, "../kernels/scal_mul.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	result := *VramF32(s, PtrRead(s,PtrSlotB), 0)
	if math.Abs(float64(result-3.0)) > 1e-5 {
		t.Fatalf("got %.4f exp 3.0", result)
	}
}

func TestGELU(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	// C test: PTR_SLOT_A=input, PTR_SLOT_B=output
	A := AllocMap(s,PtrSlotA, 4)
	AllocMap(s,PtrSlotB, 4)
	*VramF32(s, A, 0) = 0.5

	GPUStateLoadKernel(s, "../kernels/gelu.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	result := *VramF32(s, PtrRead(s,PtrSlotB), 0)
	exp := 0.5 * (1.0 / (1.0 + math.Exp(-1.702*0.5)))
	if math.Abs(float64(result-float32(exp))) > 1e-3 {
		t.Fatalf("got %.4f exp %.4f", result, exp)
	}
}

func TestSoftmax(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	N := 64
	A := AllocMap(s,PtrSlotA, N*4)
	AllocMap(s,PtrSlotB, N*4) // B slot
	for i := 0; i < N; i++ {
		*VramF32(s, A, i) = (float32(i) - 32.0) * 0.1
	}

	GPUStateLoadKernel(s, "../kernels/softmax.bin")

	if err := GPUStateRun(s,1, 1, 1, uint32(N), 1, 1); err != nil {
		t.Fatal(err)
	}

	Bout := PtrRead(s,PtrSlotB)
	var sum float32
	for i := 0; i < N; i++ {
		sum += *VramF32(s, Bout, i)
	}
	if math.Abs(float64(sum-1.0)) > 0.02 {
		t.Fatalf("sum=%.4f exp ~1.0", sum)
	}
}

func TestSoftmaxNorm(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	A := AllocMap(s,PtrSlotA, 4)
	AllocMap(s,PtrSlotC, 4)
	*VramF32(s, A, 0) = float32(math.Exp(1))
	*VramF32(s, 0x400000, 0) = float32(math.Exp(1))

	GPUStateLoadKernel(s, "../kernels/softmax_norm.bin")

	if err := GPUStateRun(s,1, 1, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}

	result := *VramF32(s, PtrRead(s,PtrSlotC), 0)
	if math.Abs(float64(result-1.0)) > 0.01 {
		t.Fatalf("got %.4f exp 1.0", result)
	}
}

func TestConv2d(t *testing.T) {
	s := GPUStateInit()
	defer FreeGPGPUState(s)

	in := [9]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	kr := [4]float32{1, 0, 0, 1}
	A := AllocMap(s,PtrSlotA, 9*4)
	B := AllocMap(s,PtrSlotB, 4*4)
	AllocMap(s,PtrSlotC, 4)
	VramWrite8(s, A, unsafe.Pointer(&in[0]), 9*4)
	VramWrite8(s, B, unsafe.Pointer(&kr[0]), 4*4)

	GPUStateLoadKernel(s, "../kernels/conv2d.bin")

	if err := GPUStateRun(s,3, 3, 1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}

	result := *VramF32(s, PtrRead(s,PtrSlotC), 0)
	if math.Abs(float64(result-6.0)) > 1e-4 {
		t.Fatalf("got %.4f exp 6.0", result)
	}
}

func divS32(a, b int32) uint32 { if b != 0 { return uint32(a / b) }; return 0xFFFFFFFF }
func divU32(a, b int32) uint32 { if b != 0 { return uint32(uint32(a) / uint32(b)) }; return 0xFFFFFFFF }
func remS32(a, b int32) uint32 { if b != 0 { return uint32(a % b) }; return uint32(a) }
func remU32(a, b int32) uint32 { if b != 0 { return uint32(uint32(a) % uint32(b)) }; return uint32(a) }
