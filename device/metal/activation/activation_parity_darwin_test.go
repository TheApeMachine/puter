//go:build darwin && cgo

package activation

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestActivationStandardUnaryMetalParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given Metal standard unary activation kernels", t, func() {
		type unaryCase struct {
			name      string
			maxULPF32 int
			maxULPRed int
			reference func(dtype.DType) parity.UnaryReference
			kernel    StandardKernel
		}

		cases := []unaryCase{
			{
				name:      "ReLU",
				maxULPF32: 1,
				maxULPRed: 2,
				reference: parity.ReferenceReLU,
				kernel:    StandardReLU,
			},
			{
				name:      "Exp",
				maxULPF32: 2,
				maxULPRed: 3,
				reference: parity.ReferenceExp,
				kernel:    StandardExp,
			},
			{
				name:      "Gelu",
				maxULPF32: 2,
				maxULPRed: 3,
				reference: parity.ReferenceGelu,
				kernel:    StandardGelu,
			},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, storageDType := range []dtype.DType{
					dtype.Float32,
					dtype.Float16,
					dtype.BFloat16,
				} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, count := range parity.Lengths {
							convey.Convey(fmt.Sprintf("N=%d", count), func() {
								source := parity.RandomUnaryInput(count, 0x4D00+int64(count))
								want := parity.ComputeUnaryReference(
									source,
									storageDType,
									testCase.reference(storageDType),
								)

								sourceTensor := harness.UploadVector(source, storageDType)
								destinationTensor := harness.UploadVector(make([]float32, count), storageDType)
								defer sourceTensor.Close()
								defer destinationTensor.Close()

								if err := DispatchStandardUnaryRefs(
									harness.ContextRef(),
									destinationTensor.Ref(),
									sourceTensor.Ref(),
									storageDType,
									testCase.kernel,
									uint32(count),
								); err != nil {
									t.Fatalf("dispatch %s: %v", testCase.name, err)
								}

								got := harness.DownloadFloat32(destinationTensor, storageDType)
								maxULP := reducedMaxULP(storageDType, testCase.maxULPF32, testCase.maxULPRed)
								parity.AssertFloat32SlicesWithinULP(t, got, want, maxULP)
							})
						}
					})
				}
			})
		}
	})
}

func BenchmarkActivationMetalReLU(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	source := make([]float32, count)

	for index := range source {
		source[index] = rand.Float32()*4.0 - 2.0
	}

	sourceTensor := harness.UploadVector(source, dtype.Float32)
	destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchStandardUnaryRefs(
			harness.ContextRef(),
			destinationTensor.Ref(),
			sourceTensor.Ref(),
			dtype.Float32,
			StandardReLU,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkActivationMetalGelu(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	source := make([]float32, count)

	for index := range source {
		source[index] = rand.Float32()*4.0 - 2.0
	}

	sourceTensor := harness.UploadVector(source, dtype.Float32)
	destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchStandardUnaryRefs(
			harness.ContextRef(),
			destinationTensor.Ref(),
			sourceTensor.Ref(),
			dtype.Float32,
			StandardGelu,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}

func reducedMaxULP(format dtype.DType, float32MaxULP int, reducedMaxULP int) int {
	if format == dtype.Float32 {
		return float32MaxULP
	}

	return reducedMaxULP
}
