//go:build xla

package activation_test

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/xla"
	"github.com/theapemachine/puter/device/xla/activation"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestActivationXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA standard unary activation kernels", t, func() {
		type unaryCase struct {
			name      string
			maxULPF32 int
			maxULPRed int
			reference func(dtype.DType) xlaparity.UnaryReference
			kernel    activation.StandardKernel
		}

		cases := []unaryCase{
			{name: "ReLU", maxULPF32: 1, maxULPRed: 2, reference: xlaparity.ReferenceReLU, kernel: activation.StandardReLU},
			{name: "Exp", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceExp, kernel: activation.StandardExp},
			{name: "Log", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceLog, kernel: activation.StandardLog},
			{name: "Sigmoid", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceSigmoid, kernel: activation.StandardSigmoid},
			{name: "Tanh", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceTanh, kernel: activation.StandardTanh},
			{name: "Silu", maxULPF32: 3, maxULPRed: 4, reference: xlaparity.ReferenceSilu, kernel: activation.StandardSilu},
			{name: "Gelu", maxULPF32: 8, maxULPRed: 4, reference: xlaparity.ReferenceGelu, kernel: activation.StandardGelu},
			{name: "GeluTanh", maxULPF32: 4, maxULPRed: 4, reference: xlaparity.ReferenceGeluTanh, kernel: activation.StandardGeluTanh},
			{name: "ELU", maxULPF32: 3, maxULPRed: 4, reference: xlaparity.ReferenceELU, kernel: activation.StandardELU},
			{name: "CELU", maxULPF32: 3, maxULPRed: 4, reference: xlaparity.ReferenceCELU, kernel: activation.StandardCELU},
			{name: "SELU", maxULPF32: 4, maxULPRed: 4, reference: xlaparity.ReferenceSELU, kernel: activation.StandardSELU},
			{name: "Softplus", maxULPF32: 3, maxULPRed: 4, reference: xlaparity.ReferenceSoftplus, kernel: activation.StandardSoftplus},
			{name: "Mish", maxULPF32: 4, maxULPRed: 4, reference: xlaparity.ReferenceMish, kernel: activation.StandardMish},
			{name: "Softsign", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceSoftsign, kernel: activation.StandardSoftsign},
			{name: "HardSigmoid", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceHardSigmoid, kernel: activation.StandardHardSigmoid},
			{name: "HardSwish", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceHardSwish, kernel: activation.StandardHardSwish},
			{name: "HardTanh", maxULPF32: 1, maxULPRed: 2, reference: xlaparity.ReferenceHardTanh, kernel: activation.StandardHardTanh},
			{name: "HardGelu", maxULPF32: 2, maxULPRed: 3, reference: xlaparity.ReferenceHardGelu, kernel: activation.StandardHardGelu},
			{name: "QuickGelu", maxULPF32: 3, maxULPRed: 4, reference: xlaparity.ReferenceQuickGelu, kernel: activation.StandardQuickGelu},
			{name: "TanhShrink", maxULPF32: 3, maxULPRed: 4, reference: xlaparity.ReferenceTanhShrink, kernel: activation.StandardTanhShrink},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, storageDType := range xlaparity.FloatParityDTypes {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, count := range xlaparity.Lengths {
							convey.Convey(fmt.Sprintf("N=%d", count), func() {
								source := xlaparity.RandomUnaryInput(count, 0x4E00+int64(count))
								wantBytes := xlaparity.ComputeUnaryReferenceBytes(
									source,
									storageDType,
									testCase.reference(storageDType),
								)

								sourceTensor := harness.UploadVector(source, storageDType)
								destinationTensor := harness.UploadVector(make([]float32, count), storageDType)
								defer sourceTensor.Close()
								defer destinationTensor.Close()

								runStandardUnary(
									harness.Backend(),
									destinationTensor,
									sourceTensor,
									storageDType,
									testCase.kernel,
								)

								if storageDType == dtype.Float32 || storageDType == dtype.Float64 {
									got := harness.DownloadFloat32(destinationTensor, storageDType)
									want := xlaparity.DecodeFloat32Vector(wantBytes, storageDType)
									maxULP := testCase.maxULPRed

									if storageDType == dtype.Float32 {
										maxULP = testCase.maxULPF32
									}

									xlaparity.AssertFloat32SlicesWithinULP(t, got, want, maxULP)
								}

								if storageDType != dtype.Float32 && storageDType != dtype.Float64 {
									gotBytes := harness.DownloadBytes(destinationTensor)
									xlaparity.AssertEncodedSlicesEqual(t, gotBytes, wantBytes)
								}
							})
						}
					})
				}
			})
		}
	})
}

func runStandardUnary(
	backend *xla.Backend,
	destinationTensor, sourceTensor *xla.DeviceTensor,
	format dtype.DType,
	kernel activation.StandardKernel,
) {
	backend.Activation.DispatchStandardUnary(
		unsafe.Pointer(destinationTensor),
		unsafe.Pointer(sourceTensor),
		format,
		kernel,
	)
}

func BenchmarkActivationXLAReLU(b *testing.B) {
	harness := xla.NewParityHarness(b)
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
		runStandardUnary(
			harness.Backend(),
			destinationTensor,
			sourceTensor,
			dtype.Float32,
			activation.StandardReLU,
		)
	}
}

func BenchmarkActivationXLAGelu(b *testing.B) {
	harness := xla.NewParityHarness(b)
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
		runStandardUnary(
			harness.Backend(),
			destinationTensor,
			sourceTensor,
			dtype.Float32,
			activation.StandardGelu,
		)
	}
}
