//go:build darwin && cgo

package activation

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestActivationLUTGatherMetalParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given Metal LUT gather kernels for production f16/bf16 activations", t, func() {
		cases := []struct {
			name      string
			kernel    StandardKernel
			reference func(dtype.DType) parity.UnaryReference
		}{
			{name: "Exp", kernel: StandardExp, reference: parity.ReferenceExp},
			{name: "ReLU", kernel: StandardReLU, reference: parity.ReferenceReLU},
			{name: "Gelu", kernel: StandardGelu, reference: parity.ReferenceGelu},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, count := range parity.Lengths {
							convey.Convey(fmt.Sprintf("N=%d", count), func() {
								source := parity.RandomUnaryInput(count, 0x4D10+int64(count))
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

								harness.Sync()
								gotBytes := destinationTensor.ReadBytes()
								parity.AssertEncodedSlicesEqual(t, gotBytes, wantBytes)
							})
						}
					})
				}
			})
		}
	})
}
