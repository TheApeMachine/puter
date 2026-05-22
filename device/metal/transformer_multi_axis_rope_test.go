//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

func TestRunMultiAxisRoPE(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given Metal tensors for multi-axis RoPE", testingObject, func() {
		shape := mustShapeForTest(testingObject, []int{3, 1, 8})
		values := []float32{
			1, 2, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24,
		}
		input := uploadDTypeTensorForTest(
			testingObject,
			backend,
			shape,
			dtype.Float32,
			dtypeconvert.Float32ToBytes(values),
		)
		out := emptyTensorForTest(testingObject, backend, shape, dtype.Float32)
		defer closeBenchmarkTensors(input, out)

		err := RunMultiAxisRoPE(input, out, MultiAxisRoPEConfig{
			LatentSeqLen: 2,
			LatentSide:   2,
			Base:         2000,
		})
		convey.So(err, convey.ShouldBeNil)

		actualDType, actualBytes, downloadErr := backend.Download(out)
		convey.So(downloadErr, convey.ShouldBeNil)
		convey.So(actualDType, convey.ShouldEqual, dtype.Float32)
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			expectedMultiAxisRoPE(values),
			1,
		)
	})
}

func expectedMultiAxisRoPE(values []float32) []float32 {
	expected := append([]float32(nil), values...)
	cosTheta := float32(math.Cos(1))
	sinTheta := float32(math.Sin(1))
	even := values[20]
	odd := values[21]

	expected[20] = even*cosTheta - odd*sinTheta
	expected[21] = even*sinTheta + odd*cosTheta

	return expected
}
