//go:build cuda

package activation

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestActivationCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA activation kernels", t, func() {
		type unaryCase struct {
			name      string
			maxULPF32 int
			maxULPRed int
			reference func(dtype.DType) parity.UnaryReference
			kernel    StandardKernel
		}

		unaryCases := []unaryCase{
			{
				name:      "ReLU",
				maxULPF32: 1,
				maxULPRed: 2,
				reference: parity.ReferenceReLU,
				kernel:    StandardReLU,
			},
		}

		for _, testCase := range unaryCases {
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
								wantBytes := parity.ComputeUnaryReferenceBytes(
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

								if storageDType == dtype.Float32 {
									got := harness.DownloadFloat32(destinationTensor, storageDType)
									want := parity.DecodeFloat32Vector(wantBytes, storageDType)
									parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULPF32)
								}

								if storageDType != dtype.Float32 {
									harness.Sync()
									gotBytes := destinationTensor.ReadBytes()
									parity.AssertEncodedSlicesEqual(t, gotBytes, wantBytes)
								}
							})
						}
					})
				}
			})
		}

		type slopeCase struct {
			name      string
			maxULPF32 int
			maxULPRed int
			operation string
			param     float32
			reference func(dtype.DType) parity.SlopeParamReference
		}

		slopeCases := []slopeCase{
			{
				name:      "Snake",
				maxULPF32: 3,
				maxULPRed: 4,
				operation: "snake",
				param:     0.5,
				reference: func(format dtype.DType) parity.SlopeParamReference {
					return parity.ReferenceSnake(format, 0.5)
				},
			},
		}

		for _, testCase := range slopeCases {
			convey.Convey(testCase.name, func() {
				for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, count := range parity.Lengths {
							convey.Convey(fmt.Sprintf("N=%d", count), func() {
								source := parity.RandomUnaryInput(count, 0x5100+int64(count))
								wantBytes := parity.ComputeSlopeParamReferenceBytes(
									source,
									storageDType,
									testCase.reference(storageDType),
									testCase.param,
								)

								sourceTensor := harness.UploadVector(source, storageDType)
								destinationTensor := harness.UploadVector(make([]float32, count), storageDType)
								defer sourceTensor.Close()
								defer destinationTensor.Close()

								if err := DispatchUnaryParamRefs(
									harness.ContextRef(),
									destinationTensor.Ref(),
									sourceTensor.Ref(),
									storageDType,
									testCase.operation,
									testCase.param,
									uint32(count),
								); err != nil {
									t.Fatalf("dispatch %s: %v", testCase.name, err)
								}

								if storageDType == dtype.Float32 {
									got := harness.DownloadFloat32(destinationTensor, storageDType)
									want := parity.DecodeFloat32Vector(wantBytes, storageDType)
									parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULPF32)
								}

								if storageDType != dtype.Float32 {
									harness.Sync()
									parity.AssertEncodedSlicesEqual(t, destinationTensor.ReadBytes(), wantBytes)
								}
							})
						}
					})
				}
			})
		}

		type dualCase struct {
			name      string
			maxULPF32 int
			maxULPRed int
			operation string
			param0    float32
			param1    float32
			reference func(dtype.DType) parity.DualParamReference
		}

		dualCases := []dualCase{
			{
				name:      "HardTanhRange",
				maxULPF32: 1,
				maxULPRed: 2,
				operation: "hard_tanh_range",
				param0:    -0.5,
				param1:    0.5,
				reference: func(format dtype.DType) parity.DualParamReference {
					return parity.ReferenceHardTanhRange(format, -0.5, 0.5)
				},
			},
			{
				name:      "RReLU",
				maxULPF32: 1,
				maxULPRed: 2,
				operation: "rrelu",
				param0:    0.1,
				param1:    0.3,
				reference: func(format dtype.DType) parity.DualParamReference {
					return parity.ReferenceRReLU(format, 0.1, 0.3)
				},
			},
		}

		for _, testCase := range dualCases {
			convey.Convey(testCase.name, func() {
				for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, count := range parity.Lengths {
							convey.Convey(fmt.Sprintf("N=%d", count), func() {
								source := parity.RandomUnaryInput(count, 0x6100+int64(count))
								wantBytes := parity.ComputeDualParamReferenceBytes(
									source,
									storageDType,
									testCase.reference(storageDType),
									testCase.param0,
									testCase.param1,
								)

								sourceTensor := harness.UploadVector(source, storageDType)
								destinationTensor := harness.UploadVector(make([]float32, count), storageDType)
								defer sourceTensor.Close()
								defer destinationTensor.Close()

								if err := DispatchDualParamRefs(
									harness.ContextRef(),
									destinationTensor.Ref(),
									sourceTensor.Ref(),
									storageDType,
									testCase.operation,
									testCase.param0,
									testCase.param1,
									uint32(count),
								); err != nil {
									t.Fatalf("dispatch %s: %v", testCase.name, err)
								}

								if storageDType == dtype.Float32 {
									got := harness.DownloadFloat32(destinationTensor, storageDType)
									want := parity.DecodeFloat32Vector(wantBytes, storageDType)
									parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULPF32)
								}

								if storageDType != dtype.Float32 {
									harness.Sync()
									parity.AssertEncodedSlicesEqual(t, destinationTensor.ReadBytes(), wantBytes)
								}
							})
						}
					})
				}
			})
		}
	})
}
