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
