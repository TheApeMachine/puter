//go:build xla

package matmul_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpumatmul "github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestMatmulXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA matmul", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						rows := int64(8)
						inner := int64(count)
						cols := int64(8)

						left := xlaparity.RandomUnaryInput(int(rows*inner), 0x3100+int64(count))
						right := xlaparity.RandomUnaryInput(int(inner*cols), 0x3200+int64(count))
						wantBytes := matmulReferenceBytes(left, right, int(rows), int(inner), int(cols), storageDType)

						leftTensor := harness.UploadVector(left, storageDType)
						rightTensor := harness.UploadVector(right, storageDType)
						outputTensor := harness.UploadVector(make([]float32, int(rows*cols)), storageDType)
						defer leftTensor.Close()
						defer rightTensor.Close()
						defer outputTensor.Close()

						harness.Backend().Matmul(
							unsafe.Pointer(outputTensor),
							unsafe.Pointer(leftTensor),
							unsafe.Pointer(rightTensor),
							int(rows),
							int(inner),
							int(cols),
							storageDType,
						)

						assertMatmulOutput(t, harness, outputTensor, storageDType, wantBytes)
					})
				}
			})
		}
	})
}

func matmulReferenceBytes(
	left, right []float32,
	rows, inner, cols int,
	format dtype.DType,
) []byte {
	switch format {
	case dtype.Float32, dtype.Float64, dtype.Float16, dtype.BFloat16:
		leftBytes, err := xlaparity.EncodeVector(left, format)

		if err != nil {
			panic(err)
		}

		rightBytes, err := xlaparity.EncodeVector(right, format)

		if err != nil {
			panic(err)
		}

		outputBytes := make([]byte, rows*cols*elementByteWidth(format))
		cpumatmul.New().Matmul(
			unsafe.Pointer(&outputBytes[0]),
			unsafe.Pointer(&leftBytes[0]),
			unsafe.Pointer(&rightBytes[0]),
			rows,
			inner,
			cols,
			format,
		)

		return outputBytes
	default:
		want := make([]float32, rows*cols)
		cpumatmul.MatmulFloat32Native(want, left, right, rows, inner, cols)

		encoded, err := xlaparity.EncodeVector(want, format)

		if err != nil {
			panic(err)
		}

		return encoded
	}
}

func elementByteWidth(format dtype.DType) int {
	width, err := format.Size()

	if err != nil {
		panic(err)
	}

	return width
}

func assertMatmulOutput(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	deviceTensor *xla.DeviceTensor,
	storageDType dtype.DType,
	wantBytes []byte,
) {
	testingTB.Helper()

	if storageDType == dtype.Float32 || storageDType == dtype.Float64 {
		got := harness.DownloadFloat32(deviceTensor, storageDType)
		want := xlaparity.DecodeFloat32Vector(wantBytes, storageDType)
		maxULP := 2

		if storageDType == dtype.Float64 {
			maxULP = 3
		}

		xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
	}

	if storageDType != dtype.Float32 && storageDType != dtype.Float64 {
		gotBytes := harness.DownloadBytes(deviceTensor)
		xlaparity.AssertEncodedSlicesEqual(testingTB, gotBytes, wantBytes)
	}
}
