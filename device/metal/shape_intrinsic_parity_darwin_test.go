//go:build darwin && cgo

package metal

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func TestPageWriteGatherMetalLayerParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given layered paged KV storage on Metal", testingObject, func() {
		storage := uploadRoPETensor(testingObject, backend, make([]float32, 12))
		defer storage.Close()
		values := uploadRoPETensor(testingObject, backend, []float32{7, 8})
		defer values.Close()
		pageIDs := uploadInt32MetalTensor(testingObject, backend, []int32{1, 1})
		defer pageIDs.Close()
		offsets := uploadInt32MetalTensor(testingObject, backend, []int32{0, 1})
		defer offsets.Close()

		backend.PageWrite(
			storage.DispatchPointer(),
			values.DispatchPointer(),
			pageIDs.DispatchPointer(),
			offsets.DispatchPointer(),
			storage.DispatchPointer(),
			3,
			2,
			1,
			2,
			6,
			dtype.Float32,
		)

		pageTable := uploadInt32MetalTensor(testingObject, backend, []int32{1})
		defer pageTable.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 2))
		defer output.Close()

		backend.PageGather(
			storage.DispatchPointer(),
			pageTable.DispatchPointer(),
			output.DispatchPointer(),
			3,
			2,
			1,
			2,
			6,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should gather rows written into the selected layer", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{7, 8})
		})
	})
}

func TestLastTokenMetalParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given sequence rows on Metal", testingObject, func() {
		input := uploadRoPETensor(testingObject, backend, []float32{1, 2, 3, 4, 5, 6})
		defer input.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 2))
		defer output.Close()

		backend.LastToken(
			input.DispatchPointer(),
			output.DispatchPointer(),
			3,
			8,
			8,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should copy the final row", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{5, 6})
		})
	})
}

func TestConcatMetalParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given row-wise tensors on Metal", testingObject, func() {
		left := uploadRoPETensor(testingObject, backend, []float32{1, 2, 3, 4})
		defer left.Close()
		right := uploadRoPETensor(testingObject, backend, []float32{10, 20})
		defer right.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 6))
		defer output.Close()

		backend.ConcatLastDim(
			left.DispatchPointer(),
			right.DispatchPointer(),
			output.DispatchPointer(),
			8,
			4,
			12,
			24,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should concatenate the tail of each row", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{1, 2, 10, 3, 4, 20})
		})
	})
}

func TestSliceMetalParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given a row-major tensor sliced across a middle dimension on Metal", testingObject, func() {
		input := uploadRoPETensor(testingObject, backend, []float32{
			1, 2,
			3, 4,
			5, 6,
			7, 8,
			9, 10,
			11, 12,
			13, 14,
			15, 16,
		})
		defer input.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 8))
		defer output.Close()

		backend.Slice(
			input.DispatchPointer(),
			output.DispatchPointer(),
			2,
			4,
			8,
			1,
			32,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should copy the same strided blocks as the scalar reference", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{
				3, 4,
				5, 6,
				11, 12,
				13, 14,
			})
		})
	})
}

func TestTransposeMetalParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given a row-major tensor transposed on Metal", testingObject, func() {
		input := uploadRoPETensor(testingObject, backend, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
			17, 18, 19, 20,
			21, 22, 23, 24,
		})
		defer input.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 24))
		defer output.Close()

		backend.Transpose(
			input.DispatchPointer(),
			output.DispatchPointer(),
			3,
			24,
			[]uint32{0, 2, 1},
			[]uint32{12, 4, 1},
			[]uint32{12, 3, 1},
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should match the scalar index mapping", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{
				1, 5, 9,
				2, 6, 10,
				3, 7, 11,
				4, 8, 12,
				13, 17, 21,
				14, 18, 22,
				15, 19, 23,
				16, 20, 24,
			})
		})
	})
}

func TestUpsampleNearest2DMetalParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given an NCHW tensor upsampled on Metal", testingObject, func() {
		input := uploadRoPETensor(testingObject, backend, []float32{
			1, 2,
			3, 4,
			5, 6,
			7, 8,
		})
		defer input.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 32))
		defer output.Close()

		backend.UpsampleNearest2D(
			input.DispatchPointer(),
			output.DispatchPointer(),
			2,
			2,
			2,
			4,
			4,
			32,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should replicate each source pixel inside its channel", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{
				1, 1, 2, 2,
				1, 1, 2, 2,
				3, 3, 4, 4,
				3, 3, 4, 4,
				5, 5, 6, 6,
				5, 5, 6, 6,
				7, 7, 8, 8,
				7, 7, 8, 8,
			})
		})
	})
}

func BenchmarkPageWriteGatherMetal(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	storage := uploadRoPETensor(benchmark, backend, make([]float32, 4096*16*8*64))
	defer storage.Close()
	values := uploadRoPETensor(benchmark, backend, make([]float32, 16*8*64))
	defer values.Close()
	pageIDs := uploadInt32MetalTensor(benchmark, backend, make([]int32, 16))
	defer pageIDs.Close()
	offsets := uploadInt32MetalTensor(benchmark, backend, make([]int32, 16))
	defer offsets.Close()
	pageTable := uploadInt32MetalTensor(benchmark, backend, []int32{0})
	defer pageTable.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, 16*8*64))
	defer output.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.PageWrite(
			storage.DispatchPointer(),
			values.DispatchPointer(),
			pageIDs.DispatchPointer(),
			offsets.DispatchPointer(),
			storage.DispatchPointer(),
			4096,
			16,
			8*64,
			16,
			0,
			dtype.Float32,
		)
		backend.PageGather(
			storage.DispatchPointer(),
			pageTable.DispatchPointer(),
			output.DispatchPointer(),
			4096,
			16,
			8*64,
			16,
			0,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func BenchmarkSliceMetal(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	outer := 2
	inputDimSize := 8192
	innerElements := 64
	sliceLen := 4096
	input := uploadRoPETensor(benchmark, backend, make([]float32, outer*inputDimSize*innerElements))
	defer input.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, outer*sliceLen*innerElements))
	defer output.Close()
	outBytes := outer * sliceLen * innerElements * 4
	benchmark.SetBytes(int64(outBytes))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.Slice(
			input.DispatchPointer(),
			output.DispatchPointer(),
			sliceLen,
			inputDimSize,
			innerElements*4,
			2048,
			outBytes,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func BenchmarkUpsampleNearest2DMetal(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	channels := 256
	inHeight := 128
	inWidth := 128
	outHeight := 256
	outWidth := 256
	outElements := channels * outHeight * outWidth
	input := uploadRoPETensor(benchmark, backend, make([]float32, channels*inHeight*inWidth))
	defer input.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, outElements))
	defer output.Close()
	benchmark.SetBytes(int64(outElements * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.UpsampleNearest2D(
			input.DispatchPointer(),
			output.DispatchPointer(),
			channels,
			inHeight,
			inWidth,
			outHeight,
			outWidth,
			outElements,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func BenchmarkTransposeMetal(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	count := 64 * 128 * 64
	input := uploadRoPETensor(benchmark, backend, make([]float32, count))
	defer input.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, count))
	defer output.Close()
	benchmark.SetBytes(int64(count * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.Transpose(
			input.DispatchPointer(),
			output.DispatchPointer(),
			4,
			count,
			[]uint32{0, 2, 1, 3},
			[]uint32{524288, 8192, 64, 1},
			[]uint32{524288, 4096, 64, 1},
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func BenchmarkConcatMetal(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	left := uploadRoPETensor(benchmark, backend, make([]float32, 4096*128))
	defer left.Close()
	right := uploadRoPETensor(benchmark, backend, make([]float32, 4096*128))
	defer right.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, 4096*256))
	defer output.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.ConcatLastDim(
			left.DispatchPointer(),
			right.DispatchPointer(),
			output.DispatchPointer(),
			128*4,
			128*4,
			256*4,
			4096*256*4,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func uploadInt32MetalTensor(
	testingHandle interface {
		Helper()
		Fatalf(string, ...any)
	},
	backend *Backend,
	values []int32,
) *DeviceTensor {
	testingHandle.Helper()

	shape, err := tensor.NewShape([]int{len(values)})
	if err != nil {
		testingHandle.Fatalf("uploadInt32MetalTensor: shape: %v", err)
	}

	rawBytes := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(rawBytes[index*4:], uint32(value))
	}

	resident, err := backend.Upload(shape, dtype.Int32, rawBytes)
	if err != nil {
		testingHandle.Fatalf("uploadInt32MetalTensor: upload: %v", err)
	}

	deviceTensor, ok := resident.(*DeviceTensor)
	if !ok {
		testingHandle.Fatalf("uploadInt32MetalTensor: got %T", resident)
	}

	return deviceTensor
}

func downloadFloat32MetalTensor(testingObject *testing.T, resident *DeviceTensor) []float32 {
	testingObject.Helper()

	dataType, rawBytes, err := resident.RawBytes()
	convey.So(err, convey.ShouldBeNil)

	values, err := convert.BytesToFloat32(dataType, rawBytes)
	convey.So(err, convey.ShouldBeNil)

	return values
}
