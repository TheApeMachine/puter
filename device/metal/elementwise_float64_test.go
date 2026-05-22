package metal

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalAddFloat64(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
			convey.Convey("Given Metal Float64 tensors for add", t, func() {
				kernel := lookupBinaryElementwiseKernel(t, "add", dtype.Float64)
				shape, err := tensor.NewShape([]int{elementCount})
				convey.So(err, convey.ShouldBeNil)

				leftValues := parityFloat64Values(elementCount, 3)
				rightValues := parityFloat64Values(elementCount, 7)
				expectedValues := make([]float64, elementCount)

				for index := range expectedValues {
					expectedValues[index] = leftValues[index] + rightValues[index]
				}

				left := uploadFloat64TensorForTest(t, backend, shape, leftValues)
				right := uploadFloat64TensorForTest(t, backend, shape, rightValues)
				out, err := backend.bridge.empty(shape, dtype.Float64)
				convey.So(err, convey.ShouldBeNil)

				defer func() {
					convey.So(left.Close(), convey.ShouldBeNil)
					convey.So(right.Close(), convey.ShouldBeNil)
					convey.So(out.Close(), convey.ShouldBeNil)
				}()

				err = kernel.Run(left, right, out)
				convey.So(err, convey.ShouldBeNil)

				_, actualBytes, err := backend.Download(out)
				convey.So(err, convey.ShouldBeNil)
				actualValues, err := dtypeconvert.BytesToFloat64(dtype.Float64, actualBytes)
				convey.So(err, convey.ShouldBeNil)

				for index := range expectedValues {
					if float64ULPDistance(actualValues[index], expectedValues[index]) > 0 {
						t.Fatalf(
							"float64 add mismatch at %d: got %g want %g",
							index, actualValues[index], expectedValues[index],
						)
					}
				}
			})
		})
	}
}

func parityFloat64Values(elementCount int, seed int) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = float64(index)*0.25 - float64(seed)*0.01
	}

	return values
}

func uploadFloat64TensorForTest(
	testingObject testing.TB,
	backend *Backend,
	shape tensor.Shape,
	values []float64,
) tensor.Tensor {
	testingObject.Helper()

	tensorValue, err := backend.Upload(shape, dtype.Float64, dtypeconvert.Float64ToBytes(values))
	if err != nil {
		testingObject.Fatalf("Upload Float64 failed: %v", err)
	}

	return tensorValue
}

func float64ULPDistance(actual float64, expected float64) uint64 {
	if actual == expected {
		return 0
	}

	if math.IsNaN(actual) && math.IsNaN(expected) {
		return 0
	}

	actualBits := math.Float64bits(actual)
	expectedBits := math.Float64bits(expected)

	if actualBits > expectedBits {
		return actualBits - expectedBits
	}

	return expectedBits - actualBits
}
