//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestGatedResidualMetalParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given residual, branch, and modulation tensors on Metal", testingObject, func() {
		residual := uploadRoPETensor(testingObject, backend, []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		})
		defer residual.Close()
		branch := uploadRoPETensor(testingObject, backend, []float32{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1,
			1, 1, 1,
		})
		defer branch.Close()
		modulation := uploadRoPETensor(testingObject, backend, []float32{
			0, 0, 0, 0, 0, 0, 0.5, 1, 2,
			0, 0, 0, 0, 0, 0, 3, 4, 5,
		})
		defer modulation.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 12))
		defer output.Close()

		backend.GatedResidual(
			residual.DispatchPointer(),
			branch.DispatchPointer(),
			modulation.DispatchPointer(),
			output.DispatchPointer(),
			4,
			3,
			2,
			9,
			0,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should match the scalar gated residual reference", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{
				1.5, 3, 5,
				4.5, 6, 8,
				10, 12, 14,
				13, 15, 17,
			})
		})
	})
}

func BenchmarkGatedResidualMetal(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	rows := 8192
	lastDim := 256
	rowsPerBatch := 4096
	modulationCols := lastDim * 3
	residual := uploadRoPETensor(benchmark, backend, make([]float32, rows*lastDim))
	defer residual.Close()
	branch := uploadRoPETensor(benchmark, backend, make([]float32, rows*lastDim))
	defer branch.Close()
	modulation := uploadRoPETensor(benchmark, backend, make([]float32, 2*modulationCols))
	defer modulation.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, rows*lastDim))
	defer output.Close()
	benchmark.SetBytes(int64(rows * lastDim * 4 * 3))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.GatedResidual(
			residual.DispatchPointer(),
			branch.DispatchPointer(),
			modulation.DispatchPointer(),
			output.DispatchPointer(),
			rows,
			lastDim,
			rowsPerBatch,
			modulationCols,
			0,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}
