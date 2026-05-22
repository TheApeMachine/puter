//go:build amd64

package optimizer

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2OptimizerAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2OptimizerAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestAdamStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given AdamStepSlicesAVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match adamStepSlicesScalar for N=%d", length), func() {
				runAdamOptimizerParityCase(t, length, adamStepSlicesAVX2)
			})
		}
	})
}

func TestAdamWStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given AdamWStepSlicesAVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match adamWStepSlicesScalar for N=%d", length), func() {
				runAdamWOptimizerParityCase(t, length, adamWStepSlicesAVX2)
			})
		}
	})
}

func TestSgdStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given SgdStepSlicesAVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match sgdStepSlicesScalar for N=%d", length), func() {
				runSgdOptimizerParityCase(t, length, sgdStepSlicesAVX2)
			})
		}
	})
}

func TestAdamStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given AdamStepSlicesSSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match adamStepSlicesScalar for N=%d", length), func() {
				runAdamOptimizerParityCase(t, length, adamStepSlicesSSE2)
			})
		}
	})
}

func TestAdamWStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given AdamWStepSlicesSSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match adamWStepSlicesScalar for N=%d", length), func() {
				runAdamWOptimizerParityCase(t, length, adamWStepSlicesSSE2)
			})
		}
	})
}

func TestSgdStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given SgdStepSlicesSSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match sgdStepSlicesScalar for N=%d", length), func() {
				runSgdOptimizerParityCase(t, length, sgdStepSlicesSSE2)
			})
		}
	})
}

type adamStepFn func(
	AdamConfig,
	[]float32, []float32, []float32, []float32, []float32,
)

type adamWStepFn func(
	AdamWConfig,
	[]float32, []float32, []float32, []float32, []float32,
)

type sgdStepFn func(
	SGDConfig,
	[]float32, []float32, []float32, []float32,
)

func runAdamOptimizerParityCase(
	testingObject *testing.T,
	length int,
	step adamStepFn,
) {
	config := DefaultAdamConfig()
	params := randFloat32Slice(length, 0x3100+int64(length))
	grad := randFloat32Slice(length, 0x3101+int64(length))
	firstSIMD := randFloat32Slice(length, 0x3102+int64(length))
	secondSIMD := randFloat32Slice(length, 0x3103+int64(length))
	firstScalar := append([]float32(nil), firstSIMD...)
	secondScalar := append([]float32(nil), secondSIMD...)
	outSIMD := make([]float32, length)
	outScalar := make([]float32, length)

	step(config, params, grad, firstSIMD, secondSIMD, outSIMD)
	adamStepSlicesScalar(config, params, grad, firstScalar, secondScalar, outScalar)

	parity.AssertFloat32SlicesWithinULP(testingObject, outSIMD, outScalar, optimizerAVX512MaxULP)
	parity.AssertFloat32SlicesWithinULP(testingObject, firstSIMD, firstScalar, optimizerAVX512MaxULP)
	parity.AssertFloat32SlicesWithinULP(testingObject, secondSIMD, secondScalar, optimizerAVX512MaxULP)
}

func runAdamWOptimizerParityCase(
	testingObject *testing.T,
	length int,
	step adamWStepFn,
) {
	config := DefaultAdamWConfig()
	params := randFloat32Slice(length, 0x3200+int64(length))
	grad := randFloat32Slice(length, 0x3201+int64(length))
	firstSIMD := randFloat32Slice(length, 0x3202+int64(length))
	secondSIMD := randFloat32Slice(length, 0x3203+int64(length))
	firstScalar := append([]float32(nil), firstSIMD...)
	secondScalar := append([]float32(nil), secondSIMD...)
	outSIMD := make([]float32, length)
	outScalar := make([]float32, length)

	step(config, params, grad, firstSIMD, secondSIMD, outSIMD)
	adamWStepSlicesScalar(config, params, grad, firstScalar, secondScalar, outScalar)

	parity.AssertFloat32SlicesWithinULP(testingObject, outSIMD, outScalar, optimizerAVX512MaxULP)
	parity.AssertFloat32SlicesWithinULP(testingObject, firstSIMD, firstScalar, optimizerAVX512MaxULP)
	parity.AssertFloat32SlicesWithinULP(testingObject, secondSIMD, secondScalar, optimizerAVX512MaxULP)
}

func runSgdOptimizerParityCase(
	testingObject *testing.T,
	length int,
	step sgdStepFn,
) {
	config := DefaultSGDConfig()
	params := randFloat32Slice(length, 0x3300+int64(length))
	grad := randFloat32Slice(length, 0x3301+int64(length))
	momentumSIMD := randFloat32Slice(length, 0x3302+int64(length))
	momentumScalar := append([]float32(nil), momentumSIMD...)
	outSIMD := make([]float32, length)
	outScalar := make([]float32, length)

	step(config, params, grad, momentumSIMD, outSIMD)
	sgdStepSlicesScalar(config, params, grad, momentumScalar, outScalar)

	parity.AssertFloat32SlicesWithinULP(testingObject, outSIMD, outScalar, optimizerAVX512MaxULP)
	parity.AssertFloat32SlicesWithinULP(testingObject, momentumSIMD, momentumScalar, optimizerAVX512MaxULP)
}

func TestAdamaxStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runTwoStateOptimizerAVX512Parity(t, DefaultAdamaxConfig(), adamaxStepSlicesAVX2, adamaxStepSlicesScalar)
}

func TestAdagradStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultAdagradConfig(), adagradStepSlicesAVX2, adagradStepSlicesScalar)
}

func TestRMSpropStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultRMSpropConfig(), rmspropStepSlicesAVX2, rmspropStepSlicesScalar)
}

func TestLionStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultLionConfig(), lionStepSlicesAVX2, lionStepSlicesScalar)
}

func TestLARSStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultLARSConfig(), larsStepSlicesAVX2, larsStepSlicesScalar)
}

func TestLBFGSStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given LBFGSStepSlicesAVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match lbfgsStepSlicesScalar for N=%d", length), func() {
				config := DefaultLBFGSConfig()
				params := randFloat32Slice(length, 0x41B0+int64(length))
				grad := randFloat32Slice(length, 0x41B1+int64(length))
				outSIMD := make([]float32, length)
				outScalar := make([]float32, length)

				lbfgsStepSlicesAVX2(config, params, grad, outSIMD)
				lbfgsStepSlicesScalar(config, params, grad, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outSIMD, outScalar, optimizerAVX512MaxULP)
			})
		}
	})
}

func TestHebbianStepSlicesAVX2Parity(t *testing.T) {
	if !avx2OptimizerAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runHebbianOptimizerParity(t, hebbianStepSlicesAVX2)
}

func TestAdamaxStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	runTwoStateOptimizerAVX512Parity(t, DefaultAdamaxConfig(), adamaxStepSlicesSSE2, adamaxStepSlicesScalar)
}

func TestAdagradStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultAdagradConfig(), adagradStepSlicesSSE2, adagradStepSlicesScalar)
}

func TestRMSpropStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultRMSpropConfig(), rmspropStepSlicesSSE2, rmspropStepSlicesScalar)
}

func TestLionStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultLionConfig(), lionStepSlicesSSE2, lionStepSlicesScalar)
}

func TestLARSStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultLARSConfig(), larsStepSlicesSSE2, larsStepSlicesScalar)
}

func TestLBFGSStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given LBFGSStepSlicesSSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match lbfgsStepSlicesScalar for N=%d", length), func() {
				config := DefaultLBFGSConfig()
				params := randFloat32Slice(length, 0x51B0+int64(length))
				grad := randFloat32Slice(length, 0x51B1+int64(length))
				outSIMD := make([]float32, length)
				outScalar := make([]float32, length)

				lbfgsStepSlicesSSE2(config, params, grad, outSIMD)
				lbfgsStepSlicesScalar(config, params, grad, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outSIMD, outScalar, optimizerAVX512MaxULP)
			})
		}
	})
}

func TestHebbianStepSlicesSSE2Parity(t *testing.T) {
	if !sse2OptimizerAvailable() {
		t.Skip("SSE2 required")
	}

	runHebbianOptimizerParity(t, hebbianStepSlicesSSE2)
}

func runHebbianOptimizerParity(
	testingObject *testing.T,
	runSIMD func(HebbianConfig, []float32, []float32, []float32, []float32, int),
) {
	cases := []struct {
		postDim int
		preDim  int
	}{
		{1, 1}, {1, 7}, {7, 64}, {64, 64}, {8, 1024},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("post=%d_pre=%d", testCase.postDim, testCase.preDim)

		testingObject.Run(label, func(t *testing.T) {
			config := DefaultHebbianConfig()
			elementCount := testCase.postDim * testCase.preDim
			weights := randFloat32Slice(elementCount, 0x41E0)
			post := randFloat32Slice(testCase.postDim, 0x41E1)
			pre := randFloat32Slice(testCase.preDim, 0x41E2)
			outSIMD := make([]float32, elementCount)
			outScalar := make([]float32, elementCount)

			runSIMD(config, weights, post, pre, outSIMD, testCase.preDim)
			hebbianStepSlicesScalar(config, weights, post, pre, outScalar, testCase.preDim)

			parity.AssertFloat32SlicesWithinULP(t, outSIMD, outScalar, optimizerAVX512MaxULP)
		})
	}
}
