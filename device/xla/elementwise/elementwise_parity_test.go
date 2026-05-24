//go:build xla

package elementwise_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuelementwise "github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestElementwiseXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA elementwise Add", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						left := xlaparity.RandomUnaryInput(count, 0xE100+int64(count))
						right := xlaparity.RandomUnaryInput(count, 0xE200+int64(count))
						wantBytes := computeBinaryReferenceBytes(left, right, storageDType, cpuelementwise.New().Add)

						leftTensor := harness.UploadVector(left, storageDType)
						rightTensor := harness.UploadVector(right, storageDType)
						destinationTensor := harness.UploadVector(make([]float32, count), storageDType)
						defer leftTensor.Close()
						defer rightTensor.Close()
						defer destinationTensor.Close()

						harness.Backend().Add(
							unsafe.Pointer(destinationTensor),
							unsafe.Pointer(leftTensor),
							unsafe.Pointer(rightTensor),
							count,
							storageDType,
						)

						assertParityOutput(t, harness, destinationTensor, storageDType, wantBytes, 1, 2)
					})
				}
			})
		}
	})

	convey.Convey("Given XLA elementwise Axpy", t, func() {
		alpha := float32(0.75)

		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						vectorY := xlaparity.RandomUnaryInput(count, 0xE300+int64(count))
						vectorX := xlaparity.RandomUnaryInput(count, 0xE400+int64(count))
						wantBytes := computeAxpyReferenceBytes(vectorY, vectorX, alpha, storageDType)

						yTensor := harness.UploadVector(vectorY, storageDType)
						xTensor := harness.UploadVector(vectorX, storageDType)
						defer xTensor.Close()

						harness.Backend().Axpy(
							unsafe.Pointer(yTensor),
							unsafe.Pointer(xTensor),
							count,
							alpha,
							storageDType,
						)

						assertParityOutput(t, harness, yTensor, storageDType, wantBytes, 2, 3)
					})
				}
			})
		}
	})
}

func computeBinaryReferenceBytes(
	left, right []float32,
	format dtype.DType,
	kernel func(dst, left, right unsafe.Pointer, count int, format dtype.DType),
) []byte {
	count := len(left)
	sourceLeft, err := xlaparity.EncodeVector(left, format)

	if err != nil {
		panic(err)
	}

	sourceRight, err := xlaparity.EncodeVector(right, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, len(sourceLeft))
	kernel(
		unsafe.Pointer(&destinationBytes[0]),
		unsafe.Pointer(&sourceLeft[0]),
		unsafe.Pointer(&sourceRight[0]),
		count,
		format,
	)

	return destinationBytes
}

func computeAxpyReferenceBytes(
	vectorY, vectorX []float32,
	alpha float32,
	format dtype.DType,
) []byte {
	count := len(vectorY)
	yBytes, err := xlaparity.EncodeVector(vectorY, format)

	if err != nil {
		panic(err)
	}

	xBytes, err := xlaparity.EncodeVector(vectorX, format)

	if err != nil {
		panic(err)
	}

	cpuelementwise.New().Axpy(
		unsafe.Pointer(&yBytes[0]),
		unsafe.Pointer(&xBytes[0]),
		count,
		alpha,
		format,
	)

	return yBytes
}

func assertParityOutput(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	deviceTensor *xla.DeviceTensor,
	storageDType dtype.DType,
	wantBytes []byte,
	maxULPF32 int,
	maxULPRed int,
) {
	testingTB.Helper()

	if storageDType == dtype.Float32 || storageDType == dtype.Float64 {
		got := harness.DownloadFloat32(deviceTensor, storageDType)
		want := xlaparity.DecodeFloat32Vector(wantBytes, storageDType)
		maxULP := maxULPRed

		if storageDType == dtype.Float32 {
			maxULP = maxULPF32
		}

		xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
	}

	if storageDType != dtype.Float32 && storageDType != dtype.Float64 {
		gotBytes := harness.DownloadBytes(deviceTensor)
		xlaparity.AssertEncodedSlicesEqual(testingTB, gotBytes, wantBytes)
	}
}
