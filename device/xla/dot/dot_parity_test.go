//go:build xla

package dot_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuelementwise "github.com/theapemachine/puter/device/cpu/elementwise"
	cpureduction "github.com/theapemachine/puter/device/cpu/reduction"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestDotXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA dot product", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						left := xlaparity.RandomUnaryInput(count, 0xD100+int64(count))
						right := xlaparity.RandomUnaryInput(count, 0xD200+int64(count))
						want := dotReference(left, right, storageDType)

						leftTensor := harness.UploadVector(left, storageDType)
						rightTensor := harness.UploadVector(right, storageDType)
						defer leftTensor.Close()
						defer rightTensor.Close()

						var got float32
						harness.Backend().Dot(
							unsafe.Pointer(&got),
							unsafe.Pointer(leftTensor),
							unsafe.Pointer(rightTensor),
							count,
							storageDType,
						)

						xlaparity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, 2)
					})
				}
			})
		}
	})
}

func dotReference(left, right []float32, format dtype.DType) float32 {
	count := len(left)
	leftBytes, err := xlaparity.EncodeVector(left, format)

	if err != nil {
		panic(err)
	}

	rightBytes, err := xlaparity.EncodeVector(right, format)

	if err != nil {
		panic(err)
	}

	productBytes := make([]byte, len(leftBytes))
	cpuelementwise.New().Mul(
		unsafe.Pointer(&productBytes[0]),
		unsafe.Pointer(&leftBytes[0]),
		unsafe.Pointer(&rightBytes[0]),
		count,
		format,
	)

	var result float32
	cpureduction.New().Sum(
		unsafe.Pointer(&result),
		unsafe.Pointer(&productBytes[0]),
		count,
		format,
	)
	return result
}
