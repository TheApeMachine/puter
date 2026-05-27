//go:build xla

package layernorm_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceLayerNorm = cpulayernorm.New()

const xlaRMSNormEpsilon = 1e-6

func TestLayerNormXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA layer norm", t, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						rows := 4
						lastDim := count
						input := xlaparity.RandomUnaryInput(rows*lastDim, 0x4F00+int64(count))
						scale := xlaparity.RandomUnaryInput(lastDim, 0x4F10+int64(count))
						bias := xlaparity.RandomUnaryInput(lastDim, 0x4F20+int64(count))
						wantBytes := layernormReferenceBytes(input, scale, bias, rows, lastDim, storageDType)

						inputTensor := harness.UploadVector(input, storageDType)
						scaleTensor := harness.UploadVector(scale, storageDType)
						biasTensor := harness.UploadVector(bias, storageDType)
						destinationTensor := harness.UploadVector(make([]float32, rows*lastDim), storageDType)
						defer inputTensor.Close()
						defer scaleTensor.Close()
						defer biasTensor.Close()
						defer destinationTensor.Close()

						harness.Backend().LayerNorm(
							unsafe.Pointer(inputTensor),
							unsafe.Pointer(scaleTensor),
							unsafe.Pointer(biasTensor),
							unsafe.Pointer(destinationTensor),
							rows, lastDim,
							storageDType,
						)

						assertNormOutput(t, harness, destinationTensor, storageDType, wantBytes)
					})
				}
			})
		}
	})

	convey.Convey("Given XLA RMS norm", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				rows := 4
				lastDim := count
				input := xlaparity.RandomUnaryInput(rows*lastDim, 0x4F30+int64(count))
				scale := xlaparity.RandomUnaryInput(lastDim, 0x4F40+int64(count))
				wantBytes := rmsNormReferenceBytes(input, scale, rows, lastDim, dtype.Float32)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				scaleTensor := harness.UploadVector(scale, dtype.Float32)
				destinationTensor := harness.UploadVector(make([]float32, rows*lastDim), dtype.Float32)
				defer inputTensor.Close()
				defer scaleTensor.Close()
				defer destinationTensor.Close()

				harness.Backend().RMSNorm(
					device.RMSNormConfig{Epsilon: xlaRMSNormEpsilon},
					unsafe.Pointer(inputTensor),
					unsafe.Pointer(scaleTensor),
					unsafe.Pointer(destinationTensor),
					rows, lastDim,
					dtype.Float32,
				)

				assertNormOutput(t, harness, destinationTensor, dtype.Float32, wantBytes)
			})
		}
	})
}

func layernormReferenceBytes(
	input, scale, bias []float32,
	rows, lastDim int,
	format dtype.DType,
) []byte {
	inputBytes, err := xlaparity.EncodeVector(input, format)

	if err != nil {
		panic(err)
	}

	scaleBytes, err := xlaparity.EncodeVector(scale, format)

	if err != nil {
		panic(err)
	}

	biasBytes, err := xlaparity.EncodeVector(bias, format)

	if err != nil {
		panic(err)
	}

	outputBytes := make([]byte, len(inputBytes))
	referenceLayerNorm.LayerNorm(
		unsafe.Pointer(&inputBytes[0]),
		unsafe.Pointer(&scaleBytes[0]),
		unsafe.Pointer(&biasBytes[0]),
		unsafe.Pointer(&outputBytes[0]),
		rows, lastDim,
		format,
	)

	return outputBytes
}

func rmsNormReferenceBytes(
	input, scale []float32,
	rows, lastDim int,
	format dtype.DType,
) []byte {
	inputBytes, err := xlaparity.EncodeVector(input, format)

	if err != nil {
		panic(err)
	}

	scaleBytes, err := xlaparity.EncodeVector(scale, format)

	if err != nil {
		panic(err)
	}

	outputBytes := make([]byte, len(inputBytes))
	referenceLayerNorm.RMSNorm(
		device.RMSNormConfig{Epsilon: xlaRMSNormEpsilon},
		unsafe.Pointer(&inputBytes[0]),
		unsafe.Pointer(&scaleBytes[0]),
		unsafe.Pointer(&outputBytes[0]),
		rows, lastDim,
		format,
	)

	return outputBytes
}

func assertNormOutput(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	deviceTensor *xla.DeviceTensor,
	storageDType dtype.DType,
	wantBytes []byte,
) {
	testingTB.Helper()

	if storageDType == dtype.Float32 {
		got := harness.DownloadFloat32(deviceTensor, storageDType)
		want := xlaparity.DecodeFloat32Vector(wantBytes, storageDType)
		xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 2)
	}

	if storageDType != dtype.Float32 {
		gotBytes := harness.DownloadBytes(deviceTensor)
		xlaparity.AssertEncodedSlicesEqual(testingTB, gotBytes, wantBytes)
	}
}
