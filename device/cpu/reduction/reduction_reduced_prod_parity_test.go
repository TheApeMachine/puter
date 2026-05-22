//go:build amd64

package reduction

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2ReductionAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2ReductionAvailable() bool {
	return cpu.X86.HasSSE2
}

func f16SSE2ReductionAvailable() bool {
	return cpu.X86.HasSSE2 && cpu.X86.HasAVX
}

func bf16SSE2ReductionAvailable() bool {
	return cpu.X86.HasSSE2 && cpu.X86.HasAVX
}

func TestProdBF16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runProdBF16Parity(t, ProdBF16AVX512)
}

func TestProdBF16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runProdBF16Parity(t, ProdBF16AVX2)
}

func TestProdBF16SSE2Parity(t *testing.T) {
	if !bf16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for bf16 widen")
	}

	runProdBF16Parity(t, ProdBF16SSE2)
}

func TestProdFP16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runProdFP16Parity(t, ProdFP16AVX512)
}

func TestProdFP16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runProdFP16Parity(t, ProdFP16AVX2)
}

func TestProdFP16SSE2Parity(t *testing.T) {
	if !f16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for VCVTPH2PS")
	}

	runProdFP16Parity(t, ProdFP16SSE2)
}

func runProdBF16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given bf16 prod SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ProdBF16Generic for N=%d", length), func() {
				values := randomReductionBF16Slice(length, 0x5100+int64(length))
				want := ProdBF16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertProdF32Parity(testingObject, got, want)
			})
		}
	})
}

func runProdFP16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given fp16 prod SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ProdFP16Generic for N=%d", length), func() {
				values := randomReductionFP16Slice(length, 0x5200+int64(length))
				want := ProdFP16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertProdF32Parity(testingObject, got, want)
			})
		}
	})
}

func randomReductionBF16Slice(length int, seed int64) []uint16 {
	floats := randomReductionFloat32Slice(length, seed)
	values := make([]uint16, length)

	for index, value := range floats {
		values[index] = uint16(dtype.NewBfloat16FromFloat32(value))
	}

	return values
}

func randomReductionFP16Slice(length int, seed int64) []uint16 {
	floats := randomReductionFloat32Slice(length, seed)
	values := make([]uint16, length)

	for index, value := range floats {
		values[index] = uint16(dtype.Fromfloat32(value))
	}

	return values
}

func TestProdBF16NativeParity(t *testing.T) {
	convey.Convey("Given Prod dispatch for bf16", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ProdBF16Generic for N=%d", length), func() {
				values := randomReductionBF16Slice(length, 0x5300+int64(length))
				want := ProdBF16Generic(&values[0], length)
				got := Prod(unsafe.Pointer(&values[0]), length, dtype.BFloat16)

				assertProdF32Parity(t, got, want)
			})
		}
	})
}

func TestMinBF16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runMinBF16Parity(t, MinBF16AVX512)
}

func TestMinBF16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runMinBF16Parity(t, MinBF16AVX2)
}

func TestMinBF16SSE2Parity(t *testing.T) {
	if !sse2ReductionAvailable() {
		t.Skip("SSE2 required")
	}

	runMinBF16Parity(t, MinBF16SSE2)
}

func TestMaxBF16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runMaxBF16Parity(t, MaxBF16AVX512)
}

func TestMaxBF16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runMaxBF16Parity(t, MaxBF16AVX2)
}

func TestMaxBF16SSE2Parity(t *testing.T) {
	if !sse2ReductionAvailable() {
		t.Skip("SSE2 required")
	}

	runMaxBF16Parity(t, MaxBF16SSE2)
}

func TestL1NormBF16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runL1NormBF16Parity(t, L1NormBF16AVX512)
}

func TestL1NormBF16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runL1NormBF16Parity(t, L1NormBF16AVX2)
}

func TestL1NormBF16SSE2Parity(t *testing.T) {
	if !bf16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for bf16 widen")
	}

	runL1NormBF16Parity(t, L1NormBF16SSE2)
}

func TestMinFP16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runMinFP16Parity(t, MinFP16AVX512)
}

func TestMinFP16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runMinFP16Parity(t, MinFP16AVX2)
}

func TestMinFP16SSE2Parity(t *testing.T) {
	if !f16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for VCVTPH2PS")
	}

	runMinFP16Parity(t, MinFP16SSE2)
}

func TestMaxFP16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runMaxFP16Parity(t, MaxFP16AVX512)
}

func TestMaxFP16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runMaxFP16Parity(t, MaxFP16AVX2)
}

func TestMaxFP16SSE2Parity(t *testing.T) {
	if !f16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for VCVTPH2PS")
	}

	runMaxFP16Parity(t, MaxFP16SSE2)
}

func TestL1NormFP16AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	runL1NormFP16Parity(t, L1NormFP16AVX512)
}

func TestL1NormFP16AVX2Parity(t *testing.T) {
	if !avx2ReductionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runL1NormFP16Parity(t, L1NormFP16AVX2)
}

func TestL1NormFP16SSE2Parity(t *testing.T) {
	if !f16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for VCVTPH2PS")
	}

	runL1NormFP16Parity(t, L1NormFP16SSE2)
}

func runMinBF16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given bf16 min SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MinBF16Generic for N=%d", length), func() {
				values := randomReductionBF16Slice(length, 0x6100+int64(length))
				want := MinBF16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertMinMaxF32Parity(testingObject, got, want, length)
			})
		}
	})
}

func runMaxBF16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given bf16 max SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaxBF16Generic for N=%d", length), func() {
				values := randomReductionBF16Slice(length, 0x6200+int64(length))
				want := MaxBF16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertMinMaxF32Parity(testingObject, got, want, length)
			})
		}
	})
}

func runL1NormBF16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given bf16 l1 norm SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match L1NormBF16Generic for N=%d", length), func() {
				values := randomReductionBF16Slice(length, 0x6300+int64(length))
				want := L1NormBF16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertSumF32Parity(testingObject, got, want, length)
			})
		}
	})
}

func runMinFP16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given fp16 min SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MinFP16Generic for N=%d", length), func() {
				values := randomReductionFP16Slice(length, 0x6400+int64(length))
				want := MinFP16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertMinMaxF32Parity(testingObject, got, want, length)
			})
		}
	})
}

func runMaxFP16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given fp16 max SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaxFP16Generic for N=%d", length), func() {
				values := randomReductionFP16Slice(length, 0x6500+int64(length))
				want := MaxFP16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertMinMaxF32Parity(testingObject, got, want, length)
			})
		}
	})
}

func runL1NormFP16Parity(
	testingObject *testing.T,
	runSIMD func(*uint16, int) float32,
) {
	convey.Convey("Given fp16 l1 norm SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match L1NormFP16Generic for N=%d", length), func() {
				values := randomReductionFP16Slice(length, 0x6600+int64(length))
				want := L1NormFP16Generic(&values[0], length)
				got := runSIMD(&values[0], length)

				assertSumF32Parity(testingObject, got, want, length)
			})
		}
	})
}

func assertMinMaxF32Parity(
	testingTB *testing.T,
	got, want float32,
	length int,
) {
	testingTB.Helper()

	if math.Float32bits(got) != math.Float32bits(want) {
		testingTB.Fatalf("N=%d got=%g want=%g", length, got, want)
	}
}
